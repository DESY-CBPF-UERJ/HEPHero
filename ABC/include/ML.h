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
// NN Model (Keras)
//-------------------------------------------------------------------------
class NN_Keras {
    private:
        fdeep::model NN_model = fdeep::load_model("include/keras_initial_model.json");
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

