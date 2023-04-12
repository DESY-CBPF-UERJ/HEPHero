#ifndef jet_resolution_corrector_h
#define jet_resolution_corrector_h


#include <iostream>
#include <fstream>

#include "TROOT.h"
#include "TMath.h"
#include "TString.h"
#include "TFormula.h"
#include "TRandom.h"




class Resolution_Map {

    public:

        Resolution_Map(){};
        Resolution_Map(const std::map<std::string,std::pair<double,double>>& bins, const std::map<std::string,std::pair<double,double>>& vars, const TFormula& formula):
            fBins(bins),
            fVars(vars),
            fFormula(formula) {
        }

        virtual ~Resolution_Map(){};

        void AddBinnedObservable(const std::string& name, const std::pair<double,double>& bin) {

            if (fBins.find(name) == fBins.end()) fBins[name] = bin;
            else std::cout << " Error: That binned observable was already defined, skipping to add it ..." << std::endl;

        }

        void AddVariable(const std::string& name, const std::pair<double,double>& var) {

            if (fVars.find(name) == fVars.end()) fVars[name] = var;
            else std::cout << " Error: That variable was already defined, skipping to add it ..." << std::endl;

        }

        void SetFormula(const char* name, const TString& tformula, const std::vector<double>& params) {
        //void SetFormula(const char* name, const char* tformula, const std::vector<double>& params) {

            if (fFormula.IsValid()) std::cout << " Warning: A valid formula had been previously defined, replacing it ..." << std::endl;

            fFormula = TFormula(name,tformula);

            if ( fFormula.GetNpar()==params.size() ) for (int iPar=0; iPar<params.size(); iPar++) fFormula.SetParameter(iPar,params[iPar]);
            else std::cout << " Error: Inconsistent number of parameters in TFormula and vector, skipping initialization of parameters ..." << std::endl;


        }

        int GetNumberOfBins() { return fBins.size(); }

        int GetNumberOfVariables() { return fVars.size(); }

        TFormula GetFormula() { return fFormula; }

        std::map<std::string,std::pair<double,double>> GetBins() { return fBins; }

        std::map<std::string,std::pair<double,double>> GetVariables() { return fVars; }

    private:

        std::map<std::string,std::pair<double,double>> fBins;

        std::map<std::string,std::pair<double,double>> fVars;

        TFormula fFormula;

};


class ResolutionSF_Map {

    public:

        ResolutionSF_Map(){};
        ResolutionSF_Map(const std::map<std::string,std::pair<double,double>>& bins, const std::map<std::string,std::pair<double,double>>& vars, const double& sf, const double& sf_up, const double& sf_down):
            fBins(bins),
            fVars(vars),
            fSF(sf),
            fSF_up(sf_up),
            fSF_down(sf_down) {
        }

        virtual ~ResolutionSF_Map(){};

        void AddBinnedObservable(const std::string& name, const std::pair<double,double>& bin) {

            if (fBins.find(name) == fBins.end()) fBins[name] = bin;
            else std::cout << " Error: That binned observable was already defined, skipping to add it ..." << std::endl;

        }

        void AddVariable(const std::string& name, const std::pair<double,double>& var) {

            if (fVars.find(name) == fVars.end()) fVars[name] = var;
            else std::cout << " Error: That variable was already defined, skipping to add it ..." << std::endl;

        }

        void SetScaleFactor(const double& scale) { fSF=scale; }

        void SetScaleFactorUp(const double& scale) { fSF_up=scale; }

        void SetScaleFactorDown(const double& scale) { fSF_down=scale; }

        int GetNumberOfBins() { return fBins.size(); }

        int GetNumberOfVariables() { return fVars.size(); }

        double GetScaleFactor() { return fSF; }

        double GetScaleFactorUp() { return fSF_up; }

        double GetScaleFactorDown() { return fSF_down; }

        std::map<std::string,std::pair<double,double>> GetBins() { return fBins; }

        std::map<std::string,std::pair<double,double>> GetVariables() { return fVars; }

    private:

        std::map<std::string,std::pair<double,double>> fBins;

        std::map<std::string,std::pair<double,double>> fVars;

        double fSF;

        double fSF_up;

        double fSF_down;

};

class Resolution_Corrector {

    public:

        //Resolution_Corrector();
        void ReadFiles(const std::string& file_resolution, const std::string& file_sf) {

            fResolution_Stream.open(file_resolution);

            LoadResDataFromFile();

            fSF_Stream.open(file_sf);

            LoadSFDataFromFile();

        }

        void SetResolutionStream(const std::string& file) { fResolution_Stream.open(file); }

        void SetSFStream(const std::string& file) { fSF_Stream.open(file); }

        virtual ~Resolution_Corrector(){};

        void LoadResDataFromFile() {

            fResolution_Stream.seekg(0);

            std::string line;

            //// Meta Data ////
            int n_binned_observables;
            std::vector<std::string> binned_observables;
            int n_variables;
            std::vector<std::string> variables;
            std::string formula_string;
            std::string mode;

            std::getline(fResolution_Stream,line);
            line.erase(0,1).pop_back();
            std::stringstream line_stream(line);

            line_stream >> n_binned_observables;
            for (int iBin=0;iBin<n_binned_observables;iBin++) {
                std::string obs_tmp;
                line_stream >> obs_tmp;
                binned_observables.push_back(obs_tmp);
            }
            line_stream >> n_variables;
            for (int iVar=0;iVar<n_variables;iVar++) {
                std::string var_tmp;
                line_stream >> var_tmp;
                variables.push_back(var_tmp);
            }
            line_stream >> formula_string;
            line_stream >> mode;

            //// Actual Data ///
            while (std::getline(fResolution_Stream,line)) {
                std::stringstream line_stream(line);
                int n_param;
                std::vector<double> params;
                Resolution_Map resolution_map_tmp;
                for (int iBin=0;iBin<n_binned_observables;iBin++) {
                    double obs_tmp_down, obs_tmp_up;
                    line_stream >> obs_tmp_down >> obs_tmp_up;
                    resolution_map_tmp.AddBinnedObservable(binned_observables[iBin],std::make_pair(obs_tmp_down,obs_tmp_up));
                }
                line_stream >> n_param;
                n_param=n_param-2;
                for (int iVar=0;iVar<n_variables;iVar++) {
                    double var_tmp_down, var_tmp_up;
                    line_stream >> var_tmp_down >> var_tmp_up;
                    resolution_map_tmp.AddVariable(variables[iVar],std::make_pair(var_tmp_down,var_tmp_up));
                }
                for (int iPar=0;iPar<n_param;iPar++) {
                    double param_tmp;
                    line_stream >> param_tmp;
                    params.push_back(param_tmp);
                }
                resolution_map_tmp.SetFormula(mode.c_str(),TString(formula_string),params);
                //resolution_map_tmp.SetFormula(mode.c_str(),formula_string.c_str(),params);

                fResolution.push_back(resolution_map_tmp);

            }

        }

        void LoadSFDataFromFile() {

            fSF_Stream.seekg(0);

            std::string line;

            //// Meta Data ////
            int n_binned_observables;
            std::vector<std::string> binned_observables;
            int n_variables;
            std::vector<std::string> variables;

            std::getline(fSF_Stream,line);
            line.erase(0,1).pop_back();
            std::stringstream line_stream(line);

            line_stream >> n_binned_observables;
            for (int iBin=0;iBin<n_binned_observables;iBin++) {
                std::string obs_tmp;
                line_stream >> obs_tmp;
                binned_observables.push_back(obs_tmp);
            }
            line_stream >> n_variables;
            for (int iVar=0;iVar<n_variables;iVar++) {
                std::string var_tmp;
                line_stream >> var_tmp;
                variables.push_back(var_tmp);
            }

            //// Actual Data ///
            while (std::getline(fSF_Stream,line)) {
                std::stringstream line_stream(line);
                int n_param;
                double sf_tmp, sf_tmp_down, sf_tmp_up;
                ResolutionSF_Map resolutionsf_map_tmp;
                for (int iBin=0;iBin<n_binned_observables;iBin++) {
                    double obs_tmp_down, obs_tmp_up;
                    line_stream >> obs_tmp_down >> obs_tmp_up;
                    resolutionsf_map_tmp.AddBinnedObservable(binned_observables[iBin],std::make_pair(obs_tmp_down,obs_tmp_up));
                }
                for (int iVar=0;iVar<n_variables;iVar++) {
                    double var_tmp_down, var_tmp_up;
                    line_stream >> var_tmp_down >> var_tmp_up;
                    resolutionsf_map_tmp.AddVariable(variables[iVar],std::make_pair(var_tmp_down,var_tmp_up));
                }
                line_stream >> n_param;
                line_stream >> sf_tmp;
                line_stream >> sf_tmp_down;
                line_stream >> sf_tmp_up;

                resolutionsf_map_tmp.SetScaleFactor(sf_tmp);
                resolutionsf_map_tmp.SetScaleFactorUp(sf_tmp_up);
                resolutionsf_map_tmp.SetScaleFactorDown(sf_tmp_down);

                fScale_Factor.push_back(resolutionsf_map_tmp);

            }

        }

        void SetVariablesandMatching(const std::map<std::string,double>& vars, const bool& isMatched) {

            fIsMatched = isMatched;

            if (vars.size()==0) {

                std::cout << " Errror: No variable has been passed in argument, skipping to set variables ..." << std::endl;

            }
            else {

                bool allVariablesSupported = true;
                for (auto const& map : vars) {

                    auto list_iter = std::find(fAllowedVariables.begin(), fAllowedVariables.end(), map.first);
                    if ( list_iter == fAllowedVariables.end() ) {

                        std::cout << " Error: The variable " << map.first << " is not supported in this class, skipping to set variables ..." << std::endl;
                        allVariablesSupported = false;
                        continue;

                    }

                }

                bool isPtDefined = true;
                if (vars.find("GenJetPt") == vars.end() || vars.find("JetPt") == vars.end() ) {

                    std::cout << " Error: The variables GenJetPt and JetPt must be initialized, skipping to set variables ..." << std::endl;
                    isPtDefined = false;

                }

                if ( allVariablesSupported && isPtDefined ) fVariables = vars;

            }

        }

        std::vector<Resolution_Map> GetResolutionMap() { return fResolution; }

        std::vector<ResolutionSF_Map> GetResolutionSFMap() { return fScale_Factor; }

        bool GetMatching() { return fIsMatched; }

        std::map<std::string,double> GetVariables() { return fVariables; }

        double GetResolution() {

            if ( fResolution.size()==0 ) {

                std::cout << " Warning: The configuration data has not been loaded, please load that first, setting the resolution to 0." << std::endl;
                return 0.;

            }

            if ( fVariables.size()==0 ) {

                std::cout << " Warning: The event data has not been loaded, please load that first, setting the resolution to 0." << std::endl;
                return 0.;

            }

            for (int iEnt=0; iEnt<fResolution.size(); iEnt++) {

                int isRegionFound = 1;
                for (auto const& bin_map : fResolution[iEnt].GetBins()) isRegionFound *= ( bin_map.second.first <= fVariables[bin_map.first] &&  bin_map.second.second > fVariables[bin_map.first] ) ? 1 : 0;
                for (auto const& var_map : fResolution[iEnt].GetVariables()) isRegionFound *= ( var_map.second.first <= fVariables[var_map.first] &&  var_map.second.second > fVariables[var_map.first] ) ? 1 : 0;
                if (isRegionFound==1 && fResolution[iEnt].GetNumberOfVariables()==1) return fResolution[iEnt].GetFormula().Eval(fVariables[fResolution[iEnt].GetVariables().begin()->first]);

                }

            return 0.;

        }

        double GetSF(string sys_var) {

            if ( fScale_Factor.size()==0 ) {

                std::cout << " Warning: The configuration data has not been loaded, please load that first, setting the resolution to 1." << std::endl;
                return 1.;

            }

            if ( fVariables.size()==0 ) {

                std::cout << " Warning: The event data has not been loaded, please load that first, setting the resolution to 1." << std::endl;
                return 1.;

            }

            for (int iEnt=0; iEnt<fScale_Factor.size(); iEnt++) {

                int isRegionFound = 1;
                for (auto const& bin_map : fScale_Factor[iEnt].GetBins()) isRegionFound *= ( bin_map.second.first <= fVariables[bin_map.first] &&  bin_map.second.second > fVariables[bin_map.first] ) ? 1 : 0;
                for (auto const& var_map : fScale_Factor[iEnt].GetVariables()) isRegionFound *= ( var_map.second.first <= fVariables[var_map.first] &&  var_map.second.second > fVariables[var_map.first] ) ? 1 : 0;
                if (isRegionFound==1){ 
                    if( sys_var == "nominal" ) return fScale_Factor[iEnt].GetScaleFactor();
                    else if( sys_var == "down" ) return fScale_Factor[iEnt].GetScaleFactorDown();
                    else if( sys_var == "up"   ) return fScale_Factor[iEnt].GetScaleFactorUp();
                }
            }

            return 1.;

        }

        double GetCorrection(string sys_var) {

            if ( fResolution.size()==0 || fScale_Factor.size()==0 ) {

                std::cout << " Warning: The configuration data has not been loaded, please load that first, setting the correction to 1." << std::endl;
                return 1.;

            }

            if ( fVariables.size()==0 ) {

                std::cout << " Warning: The event data has not been loaded, please load that first, setting the correction to 1." << std::endl;
                return 1.;

            }

            double smear_factor = 1.;
            double jer_sf = GetSF(sys_var);
            double jer = GetResolution();

            if ( fIsMatched ) smear_factor = 1. + (jer_sf-1.)*(fVariables["JetPt"] - fVariables["GenJetPt"])/fVariables["JetPt"];
            else {
                TRandom random;
                //random.SetSeed();
                smear_factor = 1. + random.Gaus(0.,jer)*TMath::Sqrt(TMath::Max(pow(jer_sf,2)-1.,0.));
            }

            return smear_factor;

        }

    private:

        std::ifstream fResolution_Stream;

        std::ifstream fSF_Stream;

        std::vector<Resolution_Map> fResolution;

        std::vector<ResolutionSF_Map> fScale_Factor;

        std::map<std::string,double> fVariables;

        bool fIsMatched;

        const std::list<std::string> fAllowedVariables = {"JetPt", "JetEta", "JetPhi", "JetE", "JetA", "Rho", "GenJetPt"};

};


#endif
