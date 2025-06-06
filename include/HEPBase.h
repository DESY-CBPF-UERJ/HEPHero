#ifndef HEPBASE_H
#define HEPBASE_H

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
#include <optional>
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
#include <torch/torch.h>
#include <torch/script.h>
#include "rapidcsv.h"
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


using namespace std;

class HEPBase {
    public:
        //=====HEP Tools===========================================================================
        void WriteCutflowInfo();
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
        bool GenRoutines();
        
    
    public:
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
        int                         RegionID;
        
        
        //---------------------------------------------------------------------------------------------------
        // HepMC
        //---------------------------------------------------------------------------------------------------
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
        
        // Generator Variables
        double GEN_HT;
        double GEN_MET_pt;
        double GEN_MET_phi;
        double GEN_MHT_pt;
        double GEN_MHT_phi;
        vector<int> parameters_id;
        
        // HepMC Variables
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


        //---------------------------------------------------------------------------------------------------
        // Experimental Variables
        //---------------------------------------------------------------------------------------------------
        double  DATA_LUMI;       // the data luminosity in fb-1
        double  DATA_LUMI_TOTAL_UNC;    // the data total luminosity uncertainty
        vector<double> DATA_LUMI_VALUES_UNC;
        vector<string> DATA_LUMI_TAGS_UNC;

        double  PROC_XSEC;       // the total cross section for the process in pb
        double  PROC_XSEC_UNC_UP;
        double  PROC_XSEC_UNC_DOWN;
        double  luminosity;
        double  lumi_total_unc;
        vector<double> lumi_values_unc;

        string dataset_group;    // "Data", "Signal" or "Bkg"
        string dataset_year;     // "16", "17" or "18"
        int    dataset_dti;      // Data-taking interval
        string dataset_era;      // if data -> "A", "B", "C", ...,; if mc -> "No"
        string dataset_sample;   // if data -> "DoubleEle", "SingleEle", "MET", ...; if mc -> "No"
        
};

#endif
