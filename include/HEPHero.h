#ifndef HEPHERO_H
#define HEPHERO_H

#include <iostream>
#include <fstream>
#include <typeinfo>
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
#include "TRandom.h"
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
#include "CMS.h"
#include <fdeep/fdeep.hpp>
#include <fdeep/model.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include "rapidcsv.h"
#include "correction.h"
#include "btageffanalyzer.h"
#include "sf_trigger.h"
//#include "jet_resolution_corrector.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/ReaderAscii.h"
#include "HepMC3/ReaderAsciiHepMC2.h"
#include "HepMC3/Print.h"
#include "HepMC3/Units.h"
#include "HepMC3/GenRunInfo.h"
#include "HepMC3/WriterDOT.h"
#include "HepMC3/Relatives.h"
#include <highfive/H5File.hpp>
#include <boost/multi_array.hpp>
#include "RoccoR.h"
#include "RecoilCorrector.h"


using namespace std;

class HEPHero {
    public:
        static HEPHero* GetInstance( char* configFileName );
        ~HEPHero() {}
        
        bool Init();
        void RunEventLoop( int nEventsMax = -1);
        void FinishRun();
    
    private:
        static HEPHero* _instance;

        bool RunObjects();
        void PreObjects();
        
        // INSERT YOUR SELECTION HERE
        
    //=================================================================================================================
    // CMS SETUP
    //=================================================================================================================
    private:
        
        //=====HEP Tools===========================================================================
        void makeHist( string nametitle, int nbinsx, double xmin, double xmax, int nbinsy, double ymin, double ymax, string xtitle, string ytitle, string ztitle, string drawOption = "", double xAxisOffset = 1., double yAxisOffset = 1.2, double zAxisOffset = 1. ); 
        void makeHist( string nametitle, int nbins, double xmin, double xmax, string xtitle, string ytitle, string drawOption = "", double xAxisOffset = 1., double yAxisOffset = 1.2 );
        void makeSysHist( string nametitle, int nbinsx, double xmin, double xmax, int nbinsy, double ymin, double ymax, string xtitle, string ytitle, string ztitle, string drawOption = "", double xAxisOffset = 1., double yAxisOffset = 1.2, double zAxisOffset = 1. );
        void makeSysHist( string nametitle, int nbins, double xmin, double xmax, string xtitle, string ytitle, string drawOption = "", double xAxisOffset = 1., double yAxisOffset = 1.2 );
        void FillSystematic( string varName, double xvarEntry, double yvarEntry, double evtWeight );
        void FillSystematic( string varName, double varEntry, double evtWeight );
        void setStyle(double ytoff = 1.0, bool marker = true, double left_margin = 0.15); 
        void MakeEfficiencyPlot( TH1D hpass, TH1D htotal, string triggerName = "");
        void HDF_insert( string varname, int* variable );
        void HDF_insert( string varname, float* variable );
        void HDF_insert( string varname, double* variable );
        void HDF_insert( string varname, bool* variable );
        void HDF_insert( string varname, unsigned int* variable );
        void HDF_insert( string varname, unsigned char* variable );
        void HDF_insert( string varname, vector<int>* variable );
        void HDF_insert( string varname, vector<float>* variable );
        void HDF_insert( string varname, vector<double>* variable );
        void HDF_fill();
        void HDF_write();
        
        
        //=====GEN Tools===========================================================================
        void calculate_gen_variables();
        void plot_events( vector<int> events );
        void WriteGenCutflowInfo();
        float part_mass( int barcode );
        //void VerticalSys();
        //void Weight_corrections();
        
        
        //=====CMS Tools===========================================================================
        void GetMCMetadata();
        void WriteCutflowInfo();
        bool PileupJet( int iJet );
        void HEMissue();
        bool METFilters();
        float GetElectronWeight( string sysID );
        float GetMuonWeight( string sysID );
        float GetJetPUIDWeight( string sysID );
        float GetBTagWeight( string sysID, string sysFlavor = "", string sysType = "" );
        float GetPileupWeight( float Pileup_nTrueInt, string sysType );
        float GetTriggerWeight( string sysID );
        float GetPrefiringWeight( string sysID );
        float GetTopPtWeight();
        float GetVJetsHTWeight();
        void JESvariation();
        void JERvariation();
        void PDFtype();
        bool Jet_GenJet_match( int ijet, float deltaR_cut );
        void Jet_lep_overlap( float deltaR_cut );
        bool Lep_jet_overlap( int ilep, string type );
        bool ElectronID( int iobj, int WP );
        bool MuonID( int iobj, int WP );
        bool MuonISO( int iobj, int WP );
        bool JetBTAG( int iobj, int WP );
        void VerticalSysSizes();
        void VerticalSys();
        void Weight_corrections();
        bool MC_processing();

    
    private:
        HEPHero() {}
        HEPHero( char* configFileName );
        
        map<string, double>         _cutFlow;
        map<string, TH1D>           _histograms1D;
        map<string, TH2D>           _histograms2D;
        map<string, string>         _histograms1DDrawOptions;
        map<string, string>         _histograms2DDrawOptions;
        
        vector<string>              _inputFileNames;
        string                      _inputTreeName;
        TChain*                     _inputTree;
        string                      _outputFileName;
        string                      _outputDirectory;
        TFile*                      _outputFile;
        TTree*                      _outputTree;
        
        string                      _sysFileName;
        TFile*                      _sysFile;
        map<string, TH1D>           _systematics1D;
        map<string, string>         _systematics1DDrawOptions;
        map<string, TH2D>           _systematics2D;
        map<string, string>         _systematics2DDrawOptions;
        ofstream                    _sysFileJson;
        int                         _Universe;
        int                         _sysID_lateral;
        string                      _sysName_lateral;
        string                      _SysSubSource;
        vector<int>                 _sysIDs_vertical;
        vector<string>              _sysNames_vertical;
        random_device               _rand;
        
        vector<string>              _plotFormats;
        ofstream                    _CutflowFile;
        string                      _CutflowFileName;
        int                         _NumberEntries;
        int                         _EventPosition;
        string                      _SELECTION;
        string                      _ANALYSIS;
        string                      _datasetName;
        string                      _Files;
        int                         _FilesID;
        int                         _DatasetID;
        bool                        _applyEventWeights;
        vector<long double>         _StatisticalError;
        bool                        _Show_Timer;
        bool                        _Get_Image_in_EPS;    
        bool                        _Get_Image_in_PNG;    
        bool                        _Get_Image_in_PDF;
        time_t                      _begin;
        time_t                      _end;
        bool                        _check;
        int                         _NumMaxEvents;
        string                      _Redirector;
        string                      _Machines;
        
        HighFive::File*                             _hdf_file;
        map<string, int*>                           _hdf_int;
        map<string, float*>                         _hdf_float;
        map<string, double*>                        _hdf_double;
        map<string, bool*>                          _hdf_bool;
        map<string, unsigned int*>                  _hdf_uint;
        map<string, unsigned char*>                 _hdf_uchar;
        map<string, vector<int>*>                   _hdf_intVec;
        map<string, vector<float>*>                 _hdf_floatVec;
        map<string, vector<double>*>                _hdf_doubleVec;
        map<string, vector<int>>                    _hdf_evtVec_int;
        map<string, vector<float>>                  _hdf_evtVec_float;
        map<string, vector<double>>                 _hdf_evtVec_double;
        map<string, vector<int>>                    _hdf_evtVec_bool;
        map<string, vector<int>>                    _hdf_evtVec_uint;
        map<string, vector<int>>                    _hdf_evtVec_uchar;
        map<string, vector< vector<int>>>           _hdf_evtVec_intVec;
        map<string, vector< vector<float>>>         _hdf_evtVec_floatVec;
        map<string, vector< vector<double>>>        _hdf_evtVec_doubleVec;
        map<string, int>                            _hdf_evtVec_max_size;
        
        map<string, double>                         _hdf_intVec_N;
        map<string, double>                         _hdf_intVec_mean;
        map<string, double>                         _hdf_intVec_std;
        map<string, double>                         _hdf_floatVec_N;
        map<string, double>                         _hdf_floatVec_mean;
        map<string, double>                         _hdf_floatVec_std;
        map<string, double>                         _hdf_doubleVec_N;    // Entry counting (not WgtSum), good enough for preprocessing
        map<string, double>                         _hdf_doubleVec_mean;
        map<string, double>                         _hdf_doubleVec_std;
        
        
        
        double                      evtWeight;
        double                      SumGenWeights;
        map<string, vector<float>>  sys_vertical_sfs;
        vector<int>                 sys_vertical_size;
        vector<int>                 sys_regions;
        
        //---------------------------------------------------------------------------------------------------
        // CMS Variables
        //---------------------------------------------------------------------------------------------------
        double  DATA_LUMI;       // the data luminosity in fb-1 
        double  DATA_LUMI_TOTAL_UNC;    // the data total luminosity uncertainty
        double  DATA_LUMI_UNCORR_UNC;   // the data uncorrelated luminosity uncertainty
        double  DATA_LUMI_FC_UNC;       // the data full correlated luminosity uncertainty
        double  DATA_LUMI_FC1718_UNC;   // the data full correlated in 2017 and 2018 luminosity uncertainty
        
        double  PROC_XSEC;       // the total cross section for the process in pb
        double  PROC_XSEC_UNC_UP;
        double  PROC_XSEC_UNC_DOWN;
        string  MCmetaFileName;
        double  luminosity;
        double  lumi_total_unc;
        double  lumi_uncorr_unc;
        double  lumi_fc_unc;
        double  lumi_fc1718_unc;
        
        //----DATASET--------------------------------------
        string dataset_group;    // "Data", "Signal" or "Bkg"
        string dataset_year;     // "16", "17" or "18"
        bool   dataset_HIPM;
        // For Data
        string dataset_era;      // if data -> "A", "B", "C", ...,; if mc -> "No" 
        string dataset_sample;   // if data -> "DoubleEle", "SingleEle", "DoubleMu", "SingleMu", "EleMu", "MET", ...; if mc -> "No" 
        
        //----LUMI CERTIFICATE-----------------------------
        string  certificate_file;
        LumiSections lumi_certificate;
        
        //----PDF------------------------------------------
        string  PDF_file;
        string  PDF_TYPE;
        
        //----PILEUP---------------------------------------      
        bool    apply_pileup_wgt;
        double  pileup_wgt;
        string  pileup_file;
        shared_ptr<correction::Correction const> pileup_corr;
        
        //----ELECTRON ID------------------------------------
        bool    apply_electron_wgt;
        double  electron_wgt;
        string  electron_file;
        shared_ptr<correction::Correction const> electron_ID_corr;
        
        //----MUON ID--------------------------------------
        bool    apply_muon_wgt;
        double  muon_wgt;
        string  muon_file;
        shared_ptr<correction::Correction const> muon_ID_corr;
        shared_ptr<correction::Correction const> muon_ISO_corr;
        
        //----JET PU ID------------------------------------
        bool    apply_jet_puid_wgt;
        double  jet_puid_wgt;
        string  jet_puid_file;
        shared_ptr<correction::Correction const> jet_PUID_corr;
        
        //----B TAGGING------------------------------------
        bool    apply_btag_wgt;
        double  btag_wgt;
        string  btag_eff_file;
        string  btag_SF_file;
        BTagEffAnalyzer btag_eff;
        shared_ptr<correction::Correction const> btag_bc_corr;
        shared_ptr<correction::Correction const> btag_udsg_corr;
        
        //----TRIGGER--------------------------------------
        bool    apply_trigger_wgt;
        double  trigger_wgt;
        string  trigger_SF_file;
        sf_trigger triggerSF;
        
        //----PREFIRING------------------------------------
        bool    apply_prefiring_wgt;
        double  prefiring_wgt;
        
        //----JERC------------------------------------------
        string  jet_jerc_file;
        shared_ptr<correction::Correction const> jet_JER_SF_corr;
        shared_ptr<correction::Correction const> jet_JER_PtRes_corr;
        shared_ptr<correction::Correction const> jet_JES_Unc;
        shared_ptr<correction::Correction const> jet_JES_L1;
        shared_ptr<correction::Correction const> jet_JES_L2;
        shared_ptr<correction::Correction const> jet_JES_L3;
        shared_ptr<correction::Correction const> jet_JES_Res;
        shared_ptr<correction::Correction const> jet_JES_L1L2L3Res;
        
        //----JES------------------------------------------
        string  JES_MC_file;
        JES  JES_unc;
        
        //----JER------------------------------------------
        bool    apply_jer_corr;
        //Resolution_Corrector jer_corr;
        //string JER_file;
        //string JER_SF_file;
        
        //----MET------------------------------------------
        bool    apply_met_xy_corr;
        TLorentzVector METCorrectionFromJES;
        TLorentzVector METCorrectionFromJER;
        bool    apply_met_recoil_corr;
        string  Z_recoil_file;
        RecoilCorrector Z_recoil;
        
        //----TOP------------------------------------------
        bool    apply_top_pt_wgt;
        double  top_pt_wgt;
        
        //----VJets----------------------------------------
        bool    apply_vjets_HT_wgt;
        double  vjets_HT_wgt;
        
        //----MUON ROCHESTER-------------------------------
        bool    apply_muon_roc_corr;
        Rochester_Corrector muon_roc_corr;
        string muon_roc_file;
        Float_t Muon_raw_pt[50];
        
        //----VERTICAL SYSTEMATICS-------------------------
        bool get_PDF_sfs;
        bool get_Scale_sfs;
        bool get_ISR_sfs;
        bool get_FSR_sfs;
        bool get_Pileup_sfs;
        bool get_ElectronID_sfs;
        bool get_MuonID_sfs;
        bool get_BTag_sfs;
        bool get_JetPUID_sfs;
        bool get_Trig_sfs;
        bool get_PreFir_sfs;
        bool get_TT1LXS_sfs;
        bool get_TT2LXS_sfs;
        bool get_DYXS_sfs;
        
        //HEPHeroGEN
        //HepMC3::ReaderAscii *_ascii_file;
        HepMC3::ReaderAsciiHepMC2   *_ascii_file;
        HepMC3::GenEvent            _evt;
        bool                        _has_xsec;
        bool                        _has_pdf;
        int                         _N_PS_weights;
        string                      _momentum_unit;
        string                      _length_unit;
        HepMC3::WriterDOT           *_dot_writer;
        
        //---------------------------------------------------------------------------------------------------
        // Generator Variables
        //---------------------------------------------------------------------------------------------------
        double GEN_HT;
        double GEN_MET_pt;
        double GEN_MET_phi;
        double GEN_MHT_pt;
        double GEN_MHT_phi;
        
        //----DATASET--------------------------------------
        //string dataset_group;    
        
        //----VERTICAL SYSTEMATICS-------------------------
        //int RegionID;
        vector<int> parameters_id;
        
        
        //---------------------------------------------------------------------------------------------------
        // HepMC Variables
        //---------------------------------------------------------------------------------------------------
        int     event_number;
        int     N_mpi;
        double  event_scale;
        double  alpha_QCD;
        double  alpha_QED;
        int     signal_process_id;
        int momentum_unit_id;
        int length_unit_id;
        vector<double> weights_value;
        vector<string> weights_name;
        
        // PDF
        int id1;
        int id2;
        int pdf_id1;
        int pdf_id2;
        double x1;
        double x2;
        double scalePDF;
        double pdf1;
        double pdf2;
        
        // CROSS-SECTION
        double cross_section;
        double cross_section_unc;
        
        
    //=================================================================================================================
    // ANALYSIS SETUP
    //=================================================================================================================
    private:  
        
        //---------------------------------------------------------------------------------------------------
        // Analysis Functions
        //---------------------------------------------------------------------------------------------------
        float Recoil_linears( float x, float a1, float b1, float a2, float c );
        void METCorrection();
        void Signal_discriminators();
        void Get_Dijet_Variables();
        void Get_ttbar_Variables();
        void Regions();
        void LeptonSelection();
        void Get_LeadingAndTrailing_Lepton_Variables();
        void Get_ThirdAndFourth_Lepton_Variables();
        void Get_Leptonic_Info( bool get12 = true, bool get34 = true);
        void Get_LepLep_Variables( bool get12 = true, bool get34 = true);
        void JetSelection();
        void Get_Jet_Angular_Variables( int pt_cut = 20 );
        vector<int> findLeadingAndTrailingBJets( int btagFlag );
        float JetLepDR( int iJet );
        int TruthLepID();
        bool Trigger();
        bool SignalBJet( int iJet );
        bool SignalEle( int iEle );
        bool SignalMu( int iMu );
        //float MLClassifier( float LeadingMuPt, float NMuons, float LeadingJetPt, float NJets, float MET, float HT );
        
        
        //---------------------------------------------------------------------------------------------------
        // Analysis Variables
        //---------------------------------------------------------------------------------------------------
        
        //----ML-------------------------------------------
        string preprocessing_keras_file;
        string model_keras_file;
        NN_Keras MLP_keras;
        float MLP_score_keras;
        
        string model_torch_file;
        NN_Torch MLP_torch;
        float MLP_score_torch;
        float MLP4_score_torch;
        
        //----JET------------------------------------------
        int Nbjets;
        int Nbjets30;
        int Nbjets_LepIso04;
        int Nbjets30_LepIso04;
        int Njets;
        int Njets30;
        int Njets40;
        int Njets_LepIso04;
        int Njets30_LepIso04;
        int Njets40_LepIso04;
        int Njets_forward;
        int Njets_tight;
        int Njets_ISR;
        int NPUjets;
        float HT;
        float HT30;
        float HT40;
        float MHT;
        float MHT30;
        float MHT40;
        float MHT_trig; 
        float MDT; 
        float Jet_abseta_max;
        float OmegaMin;
        float ChiMin;
        float FMax;
        float OmegaMin30;
        float ChiMin30;
        float FMax30;
        float OmegaMin40;
        float ChiMin40;
        float FMax40;
        float LeadingJet_pt;
        float SubLeadingJet_pt;
        
        //----MET------------------------------------------
        float MET_RAW_pt;
        float MET_RAW_phi;
        float MET_Unc_pt;
        float MET_Unc_phi;
        float MET_JES_pt;
        float MET_JES_phi;
        float MET_XY_pt;
        float MET_XY_phi;
        float MET_RECOIL_pt;
        float MET_RECOIL_phi;
        float MET_JER_pt;
        float MET_JER_phi;
        float MET_Emu_pt;
        float MET_Emu_phi;
        float MET_sig;
        float Ux;
        float Uy;
        float U1;
        float U2;
        TRandom random_recoil_18;
        
        //----TRIGGERS-------------------------------------
        bool HLT_SingleEle;
        bool HLT_DoubleEle;
        bool HLT_SingleMu;
        bool HLT_DoubleMu;
        bool HLT_EleMu;
        bool HLT_MET;
        bool HLT_LEP;
        
        //----SELECTION------------------------------------
        float JET_ETA_CUT;
        float JET_PT_CUT;
        int   JET_ID_WP;
        int   JET_PUID_WP;
        int   JET_BTAG_WP;
        float JET_LEP_DR_ISO_CUT;
        
        float ELECTRON_GAP_LOWER_CUT;
        float ELECTRON_GAP_UPPER_CUT;
        float ELECTRON_ETA_CUT;
        float ELECTRON_PT_CUT;
        float ELECTRON_LOW_PT_CUT;
        int   ELECTRON_ID_WP;
        
        float MUON_ETA_CUT;
        float MUON_PT_CUT;
        float MUON_LOW_PT_CUT;
        int   MUON_ID_WP;
        int   MUON_ISO_WP;
        
        float LEPTON_DR_ISO_CUT;
        
        float LEADING_LEP_PT_CUT;
        float LEPLEP_PT_CUT;
        float MET_CUT;
        float MET_DY_UPPER_CUT;
        float LEPLEP_DR_CUT;
        float LEPLEP_DM_CUT;
        float MET_LEPLEP_DPHI_CUT;
        float MET_LEPLEP_MT_CUT;
        
        //----GENERAL--------------------------------------
        vector<int> selectedEle;
        vector<int> selectedEleLowPt;
        vector<int> selectedMu;
        vector<int> selectedMuLowPt;
        vector<int> selectedJet;
        vector<bool> Jet_LepOverlap;
        int RecoLepID;  // 11 - reco electron event, 13- reco muon event
        int RegionID;
        const float Z_pdg_mass = 91.1876; //GeV
        const float Muon_pdg_mass = 0.105658; //GeV
        const float Electron_pdg_mass = 0.000510999; //GeV
        int IdxLeadingLep;
        int IdxTrailingLep;
        int IdxThirdLep;
        int IdxFourthLep;
        TLorentzVector lep_1;
        TLorentzVector lep_2;
        TLorentzVector lep_3;
        TLorentzVector lep_4;
        float LeadingLep_pt;
        float LeadingLep_eta;
        float TrailingLep_pt;
        float TrailingLep_eta;
        float VVCR_LeadingLep_pt;
        float LepLep_mass;
        float LepLep_pt;
        float LepLep_deltaR;
        float LepLep_phi;
        float LepLep_eta;
        float LepLep_deltaM;
        float MET_LepLep_deltaPhi;
        float MET_LepLep_Mt;
        float MET_LepLep_deltaPt;
        float Lep3Lep4_deltaM;
        float Lep3Lep4_M;
        float Lep3Lep4_pt;
        float Lep3Lep4_phi;
        float Lep3Lep4_deltaR;
        float LepLep_Lep3_M;
        float LepLep_Lep3_deltaR;
        float LepLep_Lep3_deltaPhi;
        float LepLep_Lep3_deltaPt;
        float MET_Lep3_deltaPhi;
        float MET_Lep3_Mt;
        float Lep3_pt;
        float Lep4_pt;
        bool Lep1_tight;
        bool Lep2_tight;
        bool Lep3_tight;
        bool Lep4_tight;
        float Lep3_Jet_deltaR;
        float Lep3_dxy;
        float Lep3_dz;
        float Min_dilep_deltaR;
        int Nleptons;
        int NleptonsLowPt;
        int Nelectrons;
        int Nmuons;
        float Dijet_pt; 
        float Dijet_M;
        float Dijet_deltaEta;
        float Dijet_H_deltaPhi;
        float Dijet_H_pt;
        float MT2LL;
        int ttbar_reco;
        float ttbar_mass;
        float ttbar_score;
        int ttbar_reco_v2;
        float ttbar_mass_v2;
        float ttbar_score_v2;
        bool HEM_issue_ele;
        bool HEM_issue_jet;
        bool HEM_issue_ele_v2;
        bool HEM_issue_jet_v2;
        bool HEM_issue_met;
        bool HEM_filter;
        
        //float genHT;
        //float genPt;
        
        TLorentzVector LepLep;
        TLorentzVector Dijet;
        TLorentzVector MET;
        
    
    //=================================================================================================================
    // INPUT TREE SETUP
    //=================================================================================================================
    private:
               
        //UInt_t  nFatJet;                    //[50]  7    3
        //UInt_t  nFsrPhoton;                 //[50]  3    0
        //UInt_t  nGenJetAK8;                 //[50]  9    3
        //UInt_t  nSubGenJetAK8;              //[100] 16   4
        //UInt_t  nGenVisTau;                 //[50]  5    2
        //UInt_t  nPhoton;                    //[50]  9    0
        //UInt_t  nGenIsolatedPhoton;         //[50]  2    0
        
        //======================================================

        UInt_t    run;
        UInt_t    luminosityBlock;
        ULong64_t event;

        Float_t btagWeight_CSVV2;
        Float_t btagWeight_DeepCSVB;
   
        UInt_t  nCorrT1METJet;              //[100] 28   6
        Float_t CorrT1METJet_area[100];
        Float_t CorrT1METJet_eta[100];
        Float_t CorrT1METJet_muonSubtrFactor[100];
        Float_t CorrT1METJet_phi[100];
        Float_t CorrT1METJet_rawPt[100];
        /*
        Float_t CaloMET_phi;
        Float_t CaloMET_pt;
        Float_t CaloMET_sumEt;
        Float_t ChsMET_phi;
        Float_t ChsMET_pt;
        Float_t ChsMET_sumEt;
        Float_t DeepMETResolutionTune_phi;
        Float_t DeepMETResolutionTune_pt;
        Float_t DeepMETResponseTune_phi;
        Float_t DeepMETResponseTune_pt;
        */
        
        UInt_t  nElectron;                  //[50]  9    2
        Float_t Electron_deltaEtaSC[50];
        Float_t Electron_dr03EcalRecHitSumEt[50];
        Float_t Electron_dr03HcalDepth1TowerSumEt[50];
        Float_t Electron_dr03TkSumPt[50];
        Float_t Electron_dr03TkSumPtHEEP[50];
        Float_t Electron_dxy[50];
        Float_t Electron_dxyErr[50];
        Float_t Electron_dz[50];
        Float_t Electron_dzErr[50];
        Float_t Electron_eInvMinusPInv[50];
        Float_t Electron_energyErr[50];
        Float_t Electron_eta[50];
        Float_t Electron_hoe[50];
        Float_t Electron_ip3d[50];
        Float_t Electron_jetPtRelv2[50];
        Float_t Electron_jetRelIso[50];
        Float_t Electron_mass[50];
        Float_t Electron_miniPFRelIso_all[50];
        Float_t Electron_miniPFRelIso_chg[50];
        Float_t Electron_mvaFall17V1Iso[50];
        Float_t Electron_mvaFall17V1noIso[50];
        Float_t Electron_mvaFall17V2Iso[50];
        Float_t Electron_mvaFall17V2noIso[50];
        Float_t Electron_pfRelIso03_all[50];
        Float_t Electron_pfRelIso03_chg[50];
        Float_t Electron_phi[50];
        Float_t Electron_pt[50];
        Float_t Electron_r9[50];
        Float_t Electron_scEtOverPt[50];
        Float_t Electron_sieie[50];
        Float_t Electron_sip3d[50];
        Float_t Electron_mvaTTH[50];
        Int_t   Electron_charge[50];
        Int_t   Electron_cutBased[50];
        Int_t   Electron_cutBased_Fall17_V1[50];
        Int_t   Electron_jetIdx[50];
        Int_t   Electron_pdgId[50];
        Int_t   Electron_photonIdx[50];
        Int_t   Electron_tightCharge[50];
        Int_t   Electron_vidNestedWPBitmap[50];
        Int_t   Electron_vidNestedWPBitmapHEEP[50];
        Bool_t  Electron_convVeto[50];
        Bool_t  Electron_cutBased_HEEP[50];
        Bool_t  Electron_isPFcand[50];
        UChar_t Electron_jetNDauCharged[50];
        UChar_t Electron_lostHits[50];
        Bool_t  Electron_mvaFall17V1Iso_WP80[50];
        Bool_t  Electron_mvaFall17V1Iso_WP90[50];
        Bool_t  Electron_mvaFall17V1Iso_WPL[50];
        Bool_t  Electron_mvaFall17V1noIso_WP80[50];
        Bool_t  Electron_mvaFall17V1noIso_WP90[50];
        Bool_t  Electron_mvaFall17V1noIso_WPL[50];
        Bool_t  Electron_mvaFall17V2Iso_WP80[50];
        Bool_t  Electron_mvaFall17V2Iso_WP90[50];
        Bool_t  Electron_mvaFall17V2Iso_WPL[50];
        Bool_t  Electron_mvaFall17V2noIso_WP80[50];
        Bool_t  Electron_mvaFall17V2noIso_WP90[50];
        Bool_t  Electron_mvaFall17V2noIso_WPL[50];
        UChar_t Electron_seedGain[50];
        Int_t   Electron_genPartIdx[50];
        UChar_t Electron_genPartFlav[50];
        
   
        UInt_t  nGenJet;                    //[100] 23   3
        Float_t GenJet_eta[100];
        Float_t GenJet_mass[100];
        Float_t GenJet_phi[100];
        Float_t GenJet_pt[100];
        Int_t   GenJet_partonFlavour[100];
        UChar_t GenJet_hadronFlavour[100];
                
        UInt_t  nGenPart;                   //[500] 143  34
        Float_t GenPart_eta[500];
        Float_t GenPart_mass[500];
        Float_t GenPart_phi[500];
        Float_t GenPart_pt[500];
        Int_t   GenPart_genPartIdxMother[500];
        Int_t   GenPart_pdgId[500];
        Int_t   GenPart_status[500];
        Int_t   GenPart_statusFlags[500];
   
        Float_t Generator_binvar;
        Float_t Generator_scalePDF;
        Float_t Generator_weight;
        Float_t Generator_x1;
        Float_t Generator_x2;
        Float_t Generator_xpdf1;
        Float_t Generator_xpdf2;
        Int_t   Generator_id1;
        Int_t   Generator_id2;
        
        Float_t GenVtx_x;
        Float_t GenVtx_y;
        Float_t GenVtx_z;
     
        Float_t genWeight;
        Float_t LHEWeight_originalXWGTUP;
        UInt_t  nLHEPdfWeight;              //[500] 101  68
        Float_t LHEPdfWeight[500];
        UInt_t  nLHEReweightingWeight;      //[125]  125  125
        Float_t LHEReweightingWeight[125];
        UInt_t  nLHEScaleWeight;            //[50]  9    0
        Float_t LHEScaleWeight[50];
        UInt_t  nPSWeight;                  //[50]  4    0
        Float_t PSWeight[50];
   
        UInt_t  nIsoTrack;                  //[50]  6    1
        Float_t IsoTrack_dxy[50];
        Float_t IsoTrack_dz[50];
        Float_t IsoTrack_eta[50];
        Float_t IsoTrack_pfRelIso03_all[50];
        Float_t IsoTrack_pfRelIso03_chg[50];
        Float_t IsoTrack_phi[50];
        Float_t IsoTrack_pt[50];
        Float_t IsoTrack_miniPFRelIso_all[50];
        Float_t IsoTrack_miniPFRelIso_chg[50];
        Int_t   IsoTrack_fromPV[50];
        Int_t   IsoTrack_pdgId[50];
        Bool_t  IsoTrack_isHighPurityTrack[50];
        Bool_t  IsoTrack_isPFcand[50];
        Bool_t  IsoTrack_isFromLostTrack[50];
   
        UInt_t  nJet;                       //[100] 24   3
        Float_t Jet_area[100];
        Float_t Jet_btagCSVV2[100];
        Float_t Jet_btagDeepB[100];
        Float_t Jet_btagDeepCvB[100];
        Float_t Jet_btagDeepCvL[100];
        Float_t Jet_btagDeepFlavB[100];
        Float_t Jet_btagDeepFlavCvB[100];
        Float_t Jet_btagDeepFlavCvL[100];
        Float_t Jet_btagDeepFlavQG[100];
        Float_t Jet_chEmEF[100];
        Float_t Jet_chFPV0EF[100];
        Float_t Jet_chFPV1EF[100];
        Float_t Jet_chFPV2EF[100];
        Float_t Jet_chFPV3EF[100];
        Float_t Jet_chHEF[100];
        Float_t Jet_eta[100];
        Float_t Jet_mass[100];
        Float_t Jet_muEF[100];
        Float_t Jet_muonSubtrFactor[100];
        Float_t Jet_neEmEF[100];
        Float_t Jet_neHEF[100];
        Float_t Jet_phi[100];
        Float_t Jet_pt[100];
        Float_t Jet_puIdDisc[100];
        Float_t Jet_qgl[100];
        Float_t Jet_rawFactor[100];
        Float_t Jet_bRegCorr[100];
        Float_t Jet_bRegRes[100];
        Float_t Jet_cRegCorr[100];
        Float_t Jet_cRegRes[100];
        Int_t   Jet_electronIdx1[100];
        Int_t   Jet_electronIdx2[100];
        Int_t   Jet_jetId[100];
        Int_t   Jet_muonIdx1[100];
        Int_t   Jet_muonIdx2[100];
        Int_t   Jet_nConstituents[100];
        Int_t   Jet_nElectrons[100];
        Int_t   Jet_nMuons[100];
        Int_t   Jet_puId[100];
        Int_t   Jet_genJetIdx[100];
        Int_t   Jet_hadronFlavour[100];
        Int_t   Jet_partonFlavour[100];
   
        Float_t LHE_HT;
        Float_t LHE_Vpt;
        UChar_t LHE_Njets;
        /*
        Float_t LHE_HTIncoming;
        Float_t LHE_AlphaS;
        UChar_t LHE_Nb;
        UChar_t LHE_Nc;
        UChar_t LHE_Nuds;
        UChar_t LHE_Nglu;
        UChar_t LHE_NpNLO;
        UChar_t LHE_NpLO;
        UInt_t  nLHEPart;                   //[50]  8    3
        Float_t LHEPart_pt[50];
        Float_t LHEPart_eta[50];
        Float_t LHEPart_phi[50];
        Float_t LHEPart_mass[50];
        Float_t LHEPart_incomingpz[50];
        Int_t   LHEPart_pdgId[50];
        Int_t   LHEPart_status[50];
        Int_t   LHEPart_spin[50];
        */
        
        Float_t GenMET_phi;
        Float_t GenMET_pt;
        Float_t MET_MetUnclustEnUpDeltaX;
        Float_t MET_MetUnclustEnUpDeltaY;
        Float_t MET_covXX;
        Float_t MET_covXY;
        Float_t MET_covYY;
        Float_t MET_phi;            
        Float_t MET_pt;             
        Float_t MET_significance;
        Float_t MET_sumEt;
        Float_t MET_sumPtUnclustered;
   
        UInt_t  nMuon;                      //[50]  8    1
        Float_t Muon_dxy[50];
        Float_t Muon_dxyErr[50];
        Float_t Muon_dxybs[50];
        Float_t Muon_dz[50];
        Float_t Muon_dzErr[50];
        Float_t Muon_eta[50];
        Float_t Muon_ip3d[50];
        Float_t Muon_jetPtRelv2[50];
        Float_t Muon_jetRelIso[50];
        Float_t Muon_mass[50];
        Float_t Muon_miniPFRelIso_all[50];
        Float_t Muon_miniPFRelIso_chg[50];
        Float_t Muon_pfRelIso03_all[50];
        Float_t Muon_pfRelIso03_chg[50];
        Float_t Muon_pfRelIso04_all[50];
        Float_t Muon_phi[50];
        Float_t Muon_pt[50];
        Float_t Muon_ptErr[50];
        Float_t Muon_segmentComp[50];
        Float_t Muon_sip3d[50];
        Float_t Muon_softMva[50];
        Float_t Muon_tkRelIso[50];
        Float_t Muon_tunepRelPt[50];
        Float_t Muon_mvaLowPt[50];
        Float_t Muon_mvaTTH[50];
        Int_t   Muon_charge[50];
        Int_t   Muon_jetIdx[50];
        Int_t   Muon_nStations[50];
        Int_t   Muon_nTrackerLayers[50];
        Int_t   Muon_pdgId[50];
        Int_t   Muon_tightCharge[50];
        Int_t   Muon_fsrPhotonIdx[50];
        UChar_t Muon_highPtId[50];
        Bool_t  Muon_highPurity[50];
        Bool_t  Muon_inTimeMuon[50];
        Bool_t  Muon_isGlobal[50];
        Bool_t  Muon_isPFcand[50];
        Bool_t  Muon_isTracker[50];
        UChar_t Muon_jetNDauCharged[50];
        Bool_t  Muon_looseId[50];
        Bool_t  Muon_mediumId[50];
        Bool_t  Muon_mediumPromptId[50];
        UChar_t Muon_miniIsoId[50];
        UChar_t Muon_multiIsoId[50];
        UChar_t Muon_mvaId[50];
        UChar_t Muon_mvaLowPtId[50];
        UChar_t Muon_pfIsoId[50];
        UChar_t Muon_puppiIsoId[50];
        Bool_t  Muon_softId[50];
        Bool_t  Muon_softMvaId[50];
        Bool_t  Muon_tightId[50];
        UChar_t Muon_tkIsoId[50];
        Bool_t  Muon_triggerIdLoose[50];
        Int_t   Muon_genPartIdx[50];
        UChar_t Muon_genPartFlav[50];
        
   
        Float_t Pileup_nTrueInt;
        Float_t Pileup_pudensity;
        Float_t Pileup_gpudensity;
        Int_t   Pileup_nPU;
        Int_t   Pileup_sumEOOT;
        Int_t   Pileup_sumLOOT;
     
        Float_t fixedGridRhoFastjetAll;
        /*
        Float_t fixedGridRhoFastjetCentral;
        Float_t fixedGridRhoFastjetCentralCalo;
        Float_t fixedGridRhoFastjetCentralChargedPileUp;
        Float_t fixedGridRhoFastjetCentralNeutral;
        */
        
        /*
        UInt_t  nGenDressedLepton;          //[50]  4    0
        Float_t GenDressedLepton_eta[50];
        Float_t GenDressedLepton_mass[50];
        Float_t GenDressedLepton_phi[50];
        Float_t GenDressedLepton_pt[50];
        Int_t   GenDressedLepton_pdgId[50];
        Bool_t  GenDressedLepton_hasTauAnc[50];
        */
        
        /*
        UInt_t  nSoftActivityJet;           //[50]  6    0
        Float_t SoftActivityJet_eta[50];
        Float_t SoftActivityJet_phi[50];
        Float_t SoftActivityJet_pt[50];
        Float_t SoftActivityJetHT;
        Float_t SoftActivityJetHT10;
        Float_t SoftActivityJetHT2;
        Float_t SoftActivityJetHT5;
        Int_t   SoftActivityJetNjets10;
        Int_t   SoftActivityJetNjets2;
        Int_t   SoftActivityJetNjets5;
        */
        
        /*
        UInt_t  nSubJet;                    //[100] 10   2
        Float_t SubJet_btagCSVV2[100];
        Float_t SubJet_btagDeepB[100];
        Float_t SubJet_eta[100];
        Float_t SubJet_mass[100];
        Float_t SubJet_n2b1[100];
        Float_t SubJet_n3b1[100];
        Float_t SubJet_phi[100];
        Float_t SubJet_pt[100];
        Float_t SubJet_rawFactor[100];
        Float_t SubJet_tau1[100];
        Float_t SubJet_tau2[100];
        Float_t SubJet_tau3[100];
        Float_t SubJet_tau4[100];
        */
        
        /*
        UInt_t  nTau;                       //[50]  7    1
        Float_t Tau_chargedIso[50];
        Float_t Tau_dxy[50];
        Float_t Tau_dz[50];
        Float_t Tau_eta[50];
        Float_t Tau_leadTkDeltaEta[50];
        Float_t Tau_leadTkDeltaPhi[50];
        Float_t Tau_leadTkPtOverTauPt[50];
        Float_t Tau_mass[50];
        Float_t Tau_neutralIso[50];
        Float_t Tau_phi[50];
        Float_t Tau_photonsOutsideSignalCone[50];
        Float_t Tau_pt[50];
        Float_t Tau_puCorr[50];
        Float_t Tau_rawAntiEle[50];
        Float_t Tau_rawAntiEle2018[50];
        Float_t Tau_rawDeepTau2017v2p1VSe[50];
        Float_t Tau_rawDeepTau2017v2p1VSjet[50];
        Float_t Tau_rawDeepTau2017v2p1VSmu[50];
        Float_t Tau_rawIso[50];
        Float_t Tau_rawIsodR03[50];
        Float_t Tau_rawMVAnewDM2017v2[50];
        Float_t Tau_rawMVAoldDM[50];
        Float_t Tau_rawMVAoldDM2017v1[50];
        Float_t Tau_rawMVAoldDM2017v2[50];
        Float_t Tau_rawMVAoldDMdR032017v2[50];
        Int_t   Tau_charge[50];
        Int_t   Tau_decayMode[50];
        Int_t   Tau_jetIdx[50];
        Int_t   Tau_rawAntiEleCat[50];
        Int_t   Tau_rawAntiEleCat2018[50];
        UChar_t Tau_idAntiEle[50];
        UChar_t Tau_idAntiEle2018[50];
        Bool_t  Tau_idAntiEleDeadECal[50];
        UChar_t Tau_idAntiMu[50];
        Bool_t  Tau_idDecayMode[50];
        Bool_t  Tau_idDecayModeNewDMs[50];
        UChar_t Tau_idDeepTau2017v2p1VSe[50];
        UChar_t Tau_idDeepTau2017v2p1VSjet[50];
        UChar_t Tau_idDeepTau2017v2p1VSmu[50];
        UChar_t Tau_idMVAnewDM2017v2[50];
        UChar_t Tau_idMVAoldDM[50];
        UChar_t Tau_idMVAoldDM2017v1[50];
        UChar_t Tau_idMVAoldDM2017v2[50];
        UChar_t Tau_idMVAoldDMdR032017v2[50];
        */
        
        /*
        UInt_t  nTrigObj;                   //[100] 43   10
        Float_t TrigObj_pt[100];
        Float_t TrigObj_eta[100];
        Float_t TrigObj_phi[100];
        Float_t TrigObj_l1pt[100];
        Float_t TrigObj_l1pt_2[100];
        Float_t TrigObj_l2pt[100];
        Int_t   TrigObj_id[100];
        Int_t   TrigObj_l1iso[100];
        Int_t   TrigObj_l1charge[100];
        Int_t   TrigObj_filterBits[100];
        */
        
        
        //UInt_t  nOtherPV;                   //[50]  3    0
        //Float_t OtherPV_z[50];
        Float_t PV_ndof;
        Float_t PV_x;
        Float_t PV_y;
        Float_t PV_z;
        Float_t PV_chi2;
        Float_t PV_score;
        Int_t   PV_npvs;
        Int_t   PV_npvsGood;
        
        
        
        /*
        Int_t   FatJet_genJetAK8Idx[];
        Int_t   FatJet_hadronFlavour[];
        UChar_t FatJet_nBHadrons[];
        UChar_t FatJet_nCHadrons[];
        Int_t   GenJetAK8_partonFlavour[];
        UChar_t GenJetAK8_hadronFlavour[];
        Float_t GenVtx_t0;
        Int_t   Photon_genPartIdx[];
        UChar_t Photon_genPartFlav[];
        Int_t   SubJet_hadronFlavour[];
        UChar_t SubJet_nBHadrons[];
        UChar_t SubJet_nCHadrons[];
        */
        
        UInt_t  nSV;                        //[100] 13   4
        Float_t SV_dlen[100];
        Float_t SV_dlenSig[100];
        Float_t SV_dxy[100];
        Float_t SV_dxySig[100];
        Float_t SV_pAngle[100];
        Float_t SV_chi2[100];
        Float_t SV_eta[100];
        Float_t SV_mass[100];
        Float_t SV_ndof[100];
        Float_t SV_phi[100];
        Float_t SV_pt[100];
        Float_t SV_x[100];
        Float_t SV_y[100];
        Float_t SV_z[100];
        UChar_t SV_ntracks[100];
   
        Bool_t  Flag_HBHENoiseFilter;
        Bool_t  Flag_HBHENoiseIsoFilter;
        Bool_t  Flag_CSCTightHaloFilter;
        Bool_t  Flag_CSCTightHaloTrkMuUnvetoFilter;
        Bool_t  Flag_CSCTightHalo2015Filter;
        Bool_t  Flag_globalTightHalo2016Filter;
        Bool_t  Flag_globalSuperTightHalo2016Filter;
        Bool_t  Flag_HcalStripHaloFilter;
        Bool_t  Flag_hcalLaserEventFilter;
        Bool_t  Flag_EcalDeadCellTriggerPrimitiveFilter;
        Bool_t  Flag_EcalDeadCellBoundaryEnergyFilter;
        Bool_t  Flag_ecalBadCalibFilter;
        Bool_t  Flag_goodVertices;
        Bool_t  Flag_eeBadScFilter;
        Bool_t  Flag_ecalLaserCorrFilter;
        Bool_t  Flag_trkPOGFilters;
        Bool_t  Flag_chargedHadronTrackResolutionFilter;
        Bool_t  Flag_muonBadTrackFilter;
        Bool_t  Flag_BadChargedCandidateFilter;
        Bool_t  Flag_BadPFMuonFilter;
        Bool_t  Flag_BadPFMuonDzFilter;
        Bool_t  Flag_BadChargedCandidateSummer16Filter;
        Bool_t  Flag_BadPFMuonSummer16Filter;
        Bool_t  Flag_METFilters;
        
        Float_t  L1PreFiringWeight_Dn;
        Float_t  L1PreFiringWeight_Nom;
        Float_t  L1PreFiringWeight_Up;
        
        // Triggers used in 2016
        Bool_t  HLT_Ele27_WPTight_Gsf;
        Bool_t  HLT_Ele115_CaloIdVT_GsfTrkIdT;
        Bool_t  HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ;
        //Bool_t  HLT_DoubleEle37_Ele27_CaloIdL_GsfTrkIdVL;
        Bool_t  HLT_IsoMu24;
        Bool_t  HLT_IsoTkMu24;
        Bool_t  HLT_Mu50;
        Bool_t  HLT_TkMu50;
        Bool_t  HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ;
        Bool_t  HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ;
        Bool_t  HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL;
        Bool_t  HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL;
        //Bool_t  HLT_Mu30_TkMu11;
        Bool_t  HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ; 
        Bool_t  HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ;
        Bool_t  HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL; 
        Bool_t  HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL;
        Bool_t  HLT_PFMET300; 
        Bool_t  HLT_MET200; 
        Bool_t  HLT_PFHT300_PFMET110; 
        Bool_t  HLT_PFMET170_HBHECleaned; 
        Bool_t  HLT_PFMET120_PFMHT120_IDTight;   
        Bool_t  HLT_PFMETNoMu120_PFMHTNoMu120_IDTight;
        Bool_t  HLT_Photon175;
        Bool_t  HLT_DoubleEle33_CaloIdL_GsfTrkIdVL;
        
        // Triggers added in 2017
        Bool_t  HLT_Ele35_WPTight_Gsf;
        Bool_t  HLT_IsoMu27;
        Bool_t  HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL;
        Bool_t  HLT_DoubleEle33_CaloIdL_MW; 
        Bool_t  HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8; 
        Bool_t  HLT_PFMET200_HBHECleaned;
        Bool_t  HLT_PFMET200_HBHE_BeamHaloCleaned; 
        Bool_t  HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned;
        Bool_t  HLT_PFMET120_PFMHT120_IDTight_PFHT60;
        Bool_t  HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60;
        Bool_t  HLT_PFHT500_PFMET100_PFMHT100_IDTight;
        Bool_t  HLT_PFHT700_PFMET85_PFMHT85_IDTight;
        Bool_t  HLT_PFHT800_PFMET75_PFMHT75_IDTight;
        Bool_t  HLT_Photon200;
        
        // Triggers added in 2018
        Bool_t  HLT_Ele32_WPTight_Gsf;
        Bool_t  HLT_DoubleEle25_CaloIdL_MW; 
        Bool_t  HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8;
        Bool_t  HLT_OldMu100;
        Bool_t  HLT_TkMu100;
        
};

#endif
