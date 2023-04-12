#include <iostream>
#include <fstream>
#include <TFile.h>
#include <TChain.h>
#include <TCanvas.h>
#include <TROOT.h>
#include <stdlib.h>
#include <TStyle.h>
#include <TH2D.h>
#include <TLegend.h>
#include <TColor.h>
#include <math.h>
#include <THnSparse.h>
#include <map>
#include <string>
#include <vector>
#include <random>
#include "THnSparse.h"
#include "TF1.h"
#include "TSystem.h"
#include "TLorentzVector.h"
#include "TGraphAsymmErrors.h"
#include <iomanip>
#include <sys/stat.h>
#include <time.h>
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidcsv.h"
#include <fdeep/fdeep.hpp>
#include <fdeep/model.hpp>
#include <torch/torch.h>
#include <torch/script.h>

using namespace std;


//-------------------------------------------------------------------------
// Good luminosity section
//-------------------------------------------------------------------------
class LumiSections {
    // https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGoodLumiSectionsJSONFile
    private:
        rapidjson::Document certificate;
    public:
        void ReadFile( string certificate_file ){
            FILE *fp_cert = fopen(certificate_file.c_str(), "r"); 
            char buf_cert[0XFFFF];
            //FileReadStream(FILE *fp, char *buffer, std::size_t bufferSize)
            rapidjson::FileReadStream input_cert(fp_cert, buf_cert, sizeof(buf_cert));
            certificate.ParseStream(input_cert);
        }
    
        bool GoodLumiSection( string datasetName, int run, int lumiBlock ){
            bool good_data_event = true;
            if(datasetName.substr(0,4) == "Data"){
                good_data_event = false;
                string srun = to_string(run);
                if( certificate.HasMember(srun.c_str()) ){
                    rapidjson::Value& run_cert = certificate[srun.c_str()];
    
                    assert(run_cert.IsArray());
                    for (int iblock = 0; iblock < run_cert.Size(); iblock++) {
                        assert(run_cert[iblock].IsArray());
                        int block_start = run_cert[iblock][0].GetInt();
                        int block_end = run_cert[iblock][1].GetInt();
                        if( (lumiBlock >= block_start) && (lumiBlock <= block_end) ){
                            good_data_event = true;
                        }
                    }
                }
            }
            return good_data_event; 
        }
};



//-------------------------------------------------------------------------
// JES uncertainty
//-------------------------------------------------------------------------
class JES {
    private:
        vector<float> eta_ranges = {-5.4, -5.0, -4.4, -4.0, -3.5, -3.0, -2.8, -2.6, -2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.4, 5.0, 5.4};
        vector<float> pt_ranges = {9.0, 11.0, 13.5, 16.5, 19.5, 22.5, 26.0, 30.0, 34.5, 40.0, 46.0, 52.5, 60.0, 69.0, 79.0, 90.5, 105.5, 123.5, 143.0, 163.5, 185.0, 208.0, 232.5, 258.5, 286.0, 331.0, 396.0, 468.5, 549.5, 639.0, 738.0, 847.5, 968.5, 1102.0, 1249.5, 1412.0, 1590.5, 1787.0, 2003.0, 2241.0, 2503.0, 2790.5, 3107.0, 3455.0, 3837.0, 4257.0, 4719.0, 5226.5, 5784.0, 6538.0, 1000000};
        vector<vector<float>> values;
        
        int getIndex( vector<float> vec, float inputVar ){ // Return index in the vector
            int idx = 0;
            if( inputVar < vec[ vec.size()-1 ] ){
                for (unsigned int i = 0; i < vec.size()-1 ; i++){
                    if ( inputVar >= vec[i] && inputVar < vec[i+1]  ){
                        idx = i;
                        break;
                    }
                }
            }
            else idx = vec.size()-2;
            return idx;
        }
    public:
        void ReadFile( string JES_file ){
            ifstream in;
            in.open(JES_file);
            
            vector<float> values_line(pt_ranges.size()-1, 0);
            for(int i = 0; i < eta_ranges.size()-1; i++){
                values.push_back(values_line);
            }
            
            if (in.is_open()){
                for(int i = 0; i < eta_ranges.size()-1; i++){
                    for(int j = 0; j < pt_ranges.size()-1; j++){
                        in >> values[i][j];    
                    }
                }
                in.close();
            }else{
                cout << "Error opening JES file: " << JES_file << endl;
            }
        }
        
        float getUnc( float eta_input, float pt_input ){
            int eta_idx = getIndex( eta_ranges, eta_input ); 
            int pt_idx = getIndex( pt_ranges, pt_input ); 
            float uncertainty = values[eta_idx][pt_idx];

            return uncertainty;
        }
};


//-------------------------------------------------------------------------
// NN Model (Keras)
//-------------------------------------------------------------------------
class NN_Keras {
    private:
        fdeep::model NN_model = fdeep::load_model("Metadata/ML/Keras/model_temp.json");
        rapidjson::Document preprocessing;
    public:
        void readFiles( string model_file, string preprocessing_file ){
            NN_model = fdeep::load_model(model_file.c_str());
            FILE *fp_proc = fopen(preprocessing_file.c_str(), "r"); 
            char buf_proc[0XFFFF];
            rapidjson::FileReadStream stat_values(fp_proc, buf_proc, sizeof(buf_proc));
            preprocessing.ParseStream(stat_values);
        }
        
        float predict( vector<float> mlp_input, bool binary=true ){
            //vector<float> mlp_input{LeadingLep_pt, LepLep_pt, LepLep_deltaR, LepLep_deltaM, MET_pt, MET_LepLep_Mt, MET_LepLep_deltaPhi};
            rapidjson::Value& pp_mean = preprocessing["mean"];
            rapidjson::Value& pp_std = preprocessing["std"];
            for (int ivar = 0; ivar < mlp_input.size(); ivar++) {
                float mean_i = pp_mean[ivar].GetFloat();
                float std_i = pp_std[ivar].GetFloat();
                mlp_input[ivar] = (mlp_input[ivar] - mean_i)/std_i;
            }
            vector<fdeep::internal::tensor, allocator<fdeep::internal::tensor> > result = NN_model.predict({fdeep::tensor(fdeep::tensor_shape(static_cast<size_t>(mlp_input.size())), mlp_input)});
            const fdeep::float_vec& prediction = *result[0].as_vector();
            
            if( binary ){ 
                return 1-prediction[0];
            }else{
                return prediction[0];
            }
        }
};


//-------------------------------------------------------------------------
// NN Model (Torch)
//-------------------------------------------------------------------------
class NN_Torch {
    private:
        torch::jit::script::Module NN_model;
    public:
        void readFile( string model_file ){
            NN_model = torch::jit::load(model_file.c_str());
            torch::NoGradGuard no_grad; // ensures that autograd is off
            NN_model.eval();
        }
        
        float predict( vector<float> mlp_input, bool binary=true  ){
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(torch::tensor(mlp_input));
            at::Tensor result = NN_model.forward(inputs).toTensor();
            float prediction = result[0].item<float>();
            
            if( binary ){ 
                return 1-prediction;
            }else{
                return prediction;
            }
        }
};

