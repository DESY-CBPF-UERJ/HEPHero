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




