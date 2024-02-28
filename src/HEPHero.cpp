#include "HEPHero.h"

//---------------------------------------------------------------------------------------------------------------
// Initiator
//---------------------------------------------------------------------------------------------------------------
HEPHero* HEPHero::GetInstance( char *configFileName ) {
    if( !_instance ) {
        _instance = new HEPHero( configFileName );
    }
    return _instance;
}

HEPHero *HEPHero::_instance = 0;

//---------------------------------------------------------------------------------------------------------------
// Constructor
//---------------------------------------------------------------------------------------------------------------
HEPHero::HEPHero( char *configFileName ) {
    
    //======START TIME=============================================================================
    _begin = time(NULL);
    
    //======READ THE CONTROL VARIABLES FROM THE TEXT FILE==========================================
    string DatasetID;
    ifstream configFile( configFileName, ios::in );
    while( configFile.good() ) {
        string key, value;
        configFile >> key >> ws >> value;
        if( configFile.eof() ) break;
        if( key == "Selection"                  )   _SELECTION = value;
        if( key == "Analysis"                   )   _ANALYSIS = value;
        if( key == "Outpath"                    )   _outputDirectory = value;
        if( key == "InputFile"                  )   _inputFileNames.push_back( value ); 
        if( key == "InputTree"                  )   _inputTreeName = value; 
        if( key == "DatasetName"                )   _datasetName = value;
        if( key == "Files"                      )   _Files = value;
        if( key == "DatasetID"                  )   DatasetID = value;
        if( key == "LumiWeights"                )   _applyEventWeights = ( atoi(value.c_str()) == 1 );
        if( key == "Universe"                   )   _Universe = atoi(value.c_str());
        if( key == "SysID_lateral"              )   _sysID_lateral = atoi(value.c_str());
        if( key == "SysName_lateral"            )   _sysName_lateral = value.c_str();
        if( key == "SysSubSource"               )   _SysSubSource = value.c_str();
        if( key == "SysIDs_vertical"            )   _sysIDs_vertical.push_back( atoi(value.c_str()) );
        if( key == "SysNames_vertical"          )   _sysNames_vertical.push_back( value );
        if( key == "Show_Timer"                 )   _Show_Timer = (bool)(atoi(value.c_str()));
        if( key == "Get_Image_in_EPS"           )   _Get_Image_in_EPS = (bool)(atoi(value.c_str()));
        if( key == "Get_Image_in_PNG"           )   _Get_Image_in_PNG = (bool)(atoi(value.c_str()));
        if( key == "Get_Image_in_PDF"           )   _Get_Image_in_PDF = (bool)(atoi(value.c_str()));
        if( key == "Check"                      )   _check = (bool)(atoi(value.c_str()));
        if( key == "NumMaxEvents"               )   _NumMaxEvents = atoi(value.c_str());
        if( key == "Redirector"                 )   _Redirector = value;
        if( key == "Machines"                   )   _Machines = value;
        
        if( key == "MCmetaFileName"             )   MCmetaFileName = value;
        if( key == "DATA_LUMI"                  )   DATA_LUMI = atof(value.c_str());
        if( key == "DATA_LUMI_TOTAL_UNC"        )   DATA_LUMI_TOTAL_UNC = atof(value.c_str());
        if( key == "DATA_LUMI_UNCORR_UNC"       )   DATA_LUMI_UNCORR_UNC = atof(value.c_str());
        if( key == "DATA_LUMI_FC_UNC"           )   DATA_LUMI_FC_UNC = atof(value.c_str());
        if( key == "DATA_LUMI_FC1718_UNC"       )   DATA_LUMI_FC1718_UNC = atof(value.c_str());
        
        //----CORRECTIONS------------------------------------------------------------------------------------
        if( key == "PILEUP_WGT"                 )   apply_pileup_wgt = ( atoi(value.c_str()) == 1 );
        if( key == "ELECTRON_ID_WGT"            )   apply_electron_wgt = ( atoi(value.c_str()) == 1 );
        if( key == "MUON_ID_WGT"                )   apply_muon_wgt = ( atoi(value.c_str()) == 1 );
        if( key == "JET_PUID_WGT"               )   apply_jet_puid_wgt = ( atoi(value.c_str()) == 1 );
        if( key == "BTAG_WGT"                   )   apply_btag_wgt = ( atoi(value.c_str()) == 1 );
        if( key == "TRIGGER_WGT"                )   apply_trigger_wgt = ( atoi(value.c_str()) == 1 );
        if( key == "PREFIRING_WGT"              )   apply_prefiring_wgt = ( atoi(value.c_str()) == 1 );
        if( key == "JER_CORR"                   )   apply_jer_corr = ( atoi(value.c_str()) == 1 );
        if( key == "MET_XY_CORR"                )   apply_met_xy_corr = ( atoi(value.c_str()) == 1 );
        if( key == "MET_RECOIL_CORR"            )   apply_met_recoil_corr = ( atoi(value.c_str()) == 1 );
        if( key == "TOP_PT_WGT"                 )   apply_top_pt_wgt = ( atoi(value.c_str()) == 1 );
        if( key == "VJETS_HT_WGT"               )   apply_vjets_HT_wgt = ( atoi(value.c_str()) == 1 );
        if( key == "MUON_ROC_CORR"              )   apply_muon_roc_corr = ( atoi(value.c_str()) == 1 );
        
        //----METADATA FILES---------------------------------------------------------------------------------
        if( key == "lumi_certificate"           )   certificate_file = value;
        if( key == "pdf_type"                   )   PDF_file = value;
        if( key == "pileup"                     )   pileup_file = value;
        if( key == "electron"                   )   electron_file = value;
        if( key == "muon"                       )   muon_file = value;
        if( key == "JES_MC"                     )   JES_MC_file = value;
        if( key == "jet_puID"                   )   jet_puid_file = value;
        if( key == "btag_SF"                    )   btag_SF_file = value;
        if( key == "btag_eff"                   )   btag_eff_file = value;
        if( key == "trigger"                    )   trigger_SF_file = value;
        //if( key == "JER_MC"                     )   JER_file = value;
        //if( key == "JER_SF_MC"                  )   JER_SF_file = value;
        if( key == "JERC"                       )   jet_jerc_file = value;
        if( key == "mu_RoccoR"                  )   muon_roc_file = value;
        if( key == "Z_recoil"                   )   Z_recoil_file = value;
        
        //----SELECTION--------------------------------------------------------------------------------------
        if( key == "JET_ETA_CUT"                )   JET_ETA_CUT = atof(value.c_str());
        if( key == "JET_PT_CUT"                 )   JET_PT_CUT = atof(value.c_str());
        if( key == "JET_ID_WP"                  )   JET_ID_WP = atoi(value.c_str());
        if( key == "JET_PUID_WP"                )   JET_PUID_WP = atoi(value.c_str());
        if( key == "JET_BTAG_WP"                )   JET_BTAG_WP = atoi(value.c_str());
        if( key == "JET_LEP_DR_ISO_CUT"         )   JET_LEP_DR_ISO_CUT = atof(value.c_str());
        
        if( key == "ELECTRON_GAP_LOWER_CUT"     )   ELECTRON_GAP_LOWER_CUT = atof(value.c_str());
        if( key == "ELECTRON_GAP_UPPER_CUT"     )   ELECTRON_GAP_UPPER_CUT = atof(value.c_str());
        if( key == "ELECTRON_ETA_CUT"           )   ELECTRON_ETA_CUT = atof(value.c_str());
        if( key == "ELECTRON_PT_CUT"            )   ELECTRON_PT_CUT = atof(value.c_str()); 
        if( key == "ELECTRON_LOW_PT_CUT"        )   ELECTRON_LOW_PT_CUT = atof(value.c_str());
        if( key == "ELECTRON_ID_WP"             )   ELECTRON_ID_WP = atoi(value.c_str());
        
        if( key == "MUON_ETA_CUT"               )   MUON_ETA_CUT = atof(value.c_str());
        if( key == "MUON_PT_CUT"                )   MUON_PT_CUT = atof(value.c_str()); 
        if( key == "MUON_LOW_PT_CUT"            )   MUON_LOW_PT_CUT = atof(value.c_str()); 
        if( key == "MUON_ID_WP"                 )   MUON_ID_WP = atoi(value.c_str());
        if( key == "MUON_ISO_WP"                )   MUON_ISO_WP = atoi(value.c_str());

        if( key == "TAU_ETA_CUT"                )   TAU_ETA_CUT = atof(value.c_str());
        if( key == "TAU_PT_CUT"                 )   TAU_PT_CUT = atof(value.c_str());
        
        if( key == "LEPTON_DR_ISO_CUT"          )   LEPTON_DR_ISO_CUT = atof(value.c_str());
        
        if( key == "LEADING_LEP_PT_CUT"         )   LEADING_LEP_PT_CUT = atof(value.c_str());
        if( key == "LEPLEP_PT_CUT"              )   LEPLEP_PT_CUT = atof(value.c_str());
        if( key == "MET_CUT"                    )   MET_CUT = atof(value.c_str());
        if( key == "MET_DY_UPPER_CUT"           )   MET_DY_UPPER_CUT = atof(value.c_str());
        if( key == "LEPLEP_DR_CUT"              )   LEPLEP_DR_CUT = atof(value.c_str()); 
        if( key == "LEPLEP_DM_CUT"              )   LEPLEP_DM_CUT = atof(value.c_str());
        if( key == "MET_LEPLEP_DPHI_CUT"        )   MET_LEPLEP_DPHI_CUT = atof(value.c_str());
        if( key == "MET_LEPLEP_MT_CUT"          )   MET_LEPLEP_MT_CUT = atof(value.c_str());
        
        //----MACHINE LEARNING--------------------------------------------------------------------------------------
        if( key == "NN_prep_keras"              )   preprocessing_keras_file = value;
        if( key == "NN_model_keras"             )   model_keras_file = value;
        if( key == "NN_model_torch"             )   model_torch_file = value;
        
    }
    

    //======GET DATASET INFORMATION================================================================
    if( _ANALYSIS == "GEN" ){
        // HEPHeroGEN
        if( _datasetName.substr(0,2) == "H7" ){
            dataset_group = "H7";
        }
        _DatasetID = atoi(DatasetID.c_str());
        cout << "group: " << dataset_group << endl;
        
        
        string sfile = _inputFileNames[0];
        string sdelimiter = "/";
        size_t spos = sfile.rfind(sdelimiter);
        string sfile_start = sfile.substr(0, spos);
        spos = sfile_start.rfind(sdelimiter);
        string sparam = sfile_start.erase(0, spos + sdelimiter.length());
        
        string pdelimiter = "_";
        int max_nparam = 100;
        int nparam = 0;
        do{
            size_t ppos = sparam.find(pdelimiter);
            if( ppos < sparam.npos ){
                parameters_id.push_back(atoi(sparam.substr(0, ppos).c_str()));
                sparam = sparam.erase(0, ppos + pdelimiter.length());
            }else{
                parameters_id.push_back(atoi(sparam.c_str()));
                break;
            }
            nparam += 1;
        }while( nparam < max_nparam );
    }else{
        dataset_year = _datasetName.substr(_datasetName.length()-2,2);
        if( _datasetName.substr(0,4) == "Data" ){
            dataset_group = "Data";
            dataset_era = _datasetName.substr(_datasetName.length()-4,1);
            if( _datasetName.substr(5,9) == "DoubleEle" ){
                dataset_sample = "DoubleEle";
            }else if( _datasetName.substr(5,9) == "SingleEle" ){
                dataset_sample = "SingleEle";
            }else if( _datasetName.substr(5,8) == "DoubleMu" ){
                dataset_sample = "DoubleMu";
            }else if( _datasetName.substr(5,8) == "SingleMu" ){
                dataset_sample = "SingleMu";
            }else if( _datasetName.substr(5,5) == "EleMu" ){
                dataset_sample = "EleMu";
            }else if( _datasetName.substr(5,9) == "Electrons" ){
                dataset_sample = "Electrons";
            }else if( _datasetName.substr(5,3) == "MET" ){
                dataset_sample = "MET";
            }else if( _datasetName.substr(5,8) == "TauPlusX" ){
                dataset_sample = "TauPlusX";
            }
        }else{
            dataset_era = "No";
            dataset_sample = "No";
            if( _datasetName.substr(0,6) == "Signal" ){
                dataset_group = "Signal";
            }else{
                dataset_group = "Bkg";
            }
        }
        
        if( DatasetID.substr(6,1) == "1" ){
            dataset_HIPM = true;
        }else{
            dataset_HIPM = false;
        }
        _DatasetID = atoi(DatasetID.c_str());
        
        cout << "group: " << dataset_group << endl;
        cout << "year: " << dataset_year << endl;
        cout << "HIPM: " << dataset_HIPM << endl;
        cout << "era: " << dataset_era << endl; 
        cout << "sample: " << dataset_sample << endl;
    }

    string s = _Files;
    string delimiter = "_";
    size_t pos = s.find(delimiter);
    string file_start = s.substr(0, pos);
    string file_end = s.erase(0, pos + delimiter.length());
    _FilesID = atoi(file_start.c_str())*1000 + atoi(file_end.c_str());
    
    
    //======CREATE OUTPUT DIRECTORY FOR THE DATASET================================================
    string raw_outputDirectory = _outputDirectory;
    _outputDirectory = _outputDirectory + "/" + _SELECTION + "/" + _datasetName + "_files_" + _Files + "/";
    mkdir(_outputDirectory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    string sysDirectory = _outputDirectory + "/Systematics"; 
    mkdir(sysDirectory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if( _ANALYSIS == "GEN" ){
        string dotDirectory = _outputDirectory + "/Views"; 
        mkdir(dotDirectory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    

    //======ADD THE INPUT TREES TO THE TCHAINS=====================================================
    if( _ANALYSIS != "GEN" ){
        gErrorIgnoreLevel = kError;
        _inputTree = new TChain(_inputTreeName.c_str());
        for( vector<string>::iterator itr = _inputFileNames.begin(); itr != _inputFileNames.end(); ++itr ) {
            
            string inputFileName;
            if( _Machines == "CERN" ) inputFileName = "root://" + _Redirector + "//" + (*itr);
            if( _Machines == "DESY" ) inputFileName = "/pnfs/desy.de/cms/tier2/" + (*itr);  
            if( _Machines == "UERJ" ) inputFileName = "/mnt/hadoop/cms/" + (*itr);
            if( _ANALYSIS == "OPENDATA" ) inputFileName = raw_outputDirectory.substr(0,raw_outputDirectory.size()-15) + "/opendata/" + (*itr);
            if( _check || DatasetID.substr(2,2) == "99" ) inputFileName = (*itr);
            _inputTree -> Add( inputFileName.c_str() ); 
        }
    }
    

    //======ADD ASCII FILE=========================================================================
    // HEPHeroGEN
    if( _ANALYSIS == "GEN" ) _ascii_file = new HepMC3::ReaderAsciiHepMC2(_inputFileNames.at(0));
    
        
    //======CREATE OUTPUT FILE AND TREE============================================================
    if( _sysID_lateral == 0 ) {
        _outputFileName = _outputDirectory + "/Tree.root";
        _outputFile = new TFile( _outputFileName.c_str(), "RECREATE" );
    }
    _outputTree = new TTree( "selection", "selection" );
    
    
    //======CREATE HDF FILE========================================================================
    if( _sysID_lateral == 0 ) {
        _hdf_file = new HighFive::File(_outputDirectory + "selection.h5", HighFive::File::Overwrite);
    }
    
    //======CREATE SYSTEMATIC FILE AND TREE========================================================
    _sysFileName = sysDirectory + "/" + to_string(_sysID_lateral) + "_" + to_string(_Universe) + ".root";
    _sysFile = new TFile( _sysFileName.c_str(), "RECREATE" );
    _sysFileName = sysDirectory + "/" + to_string(_sysID_lateral) + "_" + to_string(_Universe) + ".json";
    _sysFileJson.open( _sysFileName.c_str(), ios::out );
    

}


//---------------------------------------------------------------------------------------------------------------
// Init
//---------------------------------------------------------------------------------------------------------------
bool HEPHero::Init() {
    
    //======SET HISTOGRAMS STYLE===================================================================
    setStyle(1.0,true,0.15);

    if( _ANALYSIS != "GEN" ){
        //======SET THE BRANCH ADDRESSES===============================================================
        _inputTree->SetBranchAddress("run", &run );
        _inputTree->SetBranchAddress("luminosityBlock", &luminosityBlock );
        _inputTree->SetBranchAddress("event", &event );
        
        /*
        _inputTree->SetBranchAddress("nFatJet", &nFatJet );
        _inputTree->SetBranchAddress("nFsrPhoton", &nFsrPhoton );
        _inputTree->SetBranchAddress("nGenJetAK8", &nGenJetAK8 );
        
        _inputTree->SetBranchAddress("nSubGenJetAK8", &nSubGenJetAK8 );
        _inputTree->SetBranchAddress("nGenVisTau", &nGenVisTau );
        _inputTree->SetBranchAddress("nLHEPdfWeight", &nLHEPdfWeight );
        _inputTree->SetBranchAddress("nLHEReweightingWeight", &nLHEReweightingWeight );
        _inputTree->SetBranchAddress("nLHEScaleWeight", &nLHEScaleWeight );
        _inputTree->SetBranchAddress("nPSWeight", &nPSWeight );
        _inputTree->SetBranchAddress("nIsoTrack", &nIsoTrack );
        _inputTree->SetBranchAddress("nLHEPart", &nLHEPart );
        _inputTree->SetBranchAddress("nPhoton", &nPhoton );
        _inputTree->SetBranchAddress("nGenDressedLepton", &nGenDressedLepton );
        _inputTree->SetBranchAddress("nGenIsolatedPhoton", &nGenIsolatedPhoton );
        _inputTree->SetBranchAddress("nSoftActivityJet", &nSoftActivityJet );
        _inputTree->SetBranchAddress("nSubJet", &nSubJet );
        _inputTree->SetBranchAddress("nTau", &nTau );
        _inputTree->SetBranchAddress("nTrigObj", &nTrigObj );
        _inputTree->SetBranchAddress("nOtherPV", &nOtherPV );
        _inputTree->SetBranchAddress("nSV", &nSV );
    */
        
        _inputTree->SetBranchAddress("fixedGridRhoFastjetAll", &fixedGridRhoFastjetAll );
        
        //-----------------------------------------------------------------------------------------------------------------------
        _inputTree->SetBranchAddress("nElectron", &nElectron );
        _inputTree->SetBranchAddress("Electron_pt", &Electron_pt );
        _inputTree->SetBranchAddress("Electron_eta", &Electron_eta );
        _inputTree->SetBranchAddress("Electron_phi", &Electron_phi );
        _inputTree->SetBranchAddress("Electron_charge", &Electron_charge );
        _inputTree->SetBranchAddress("Electron_mass", &Electron_mass );
        _inputTree->SetBranchAddress("Electron_pdgId", &Electron_pdgId );
        _inputTree->SetBranchAddress("Electron_miniPFRelIso_all", &Electron_miniPFRelIso_all );
        _inputTree->SetBranchAddress("Electron_miniPFRelIso_chg", &Electron_miniPFRelIso_chg );
        _inputTree->SetBranchAddress("Electron_jetRelIso", &Electron_jetRelIso );
        _inputTree->SetBranchAddress("Electron_jetIdx", &Electron_jetIdx );
        _inputTree->SetBranchAddress("Electron_isPFcand", &Electron_isPFcand );
        _inputTree->SetBranchAddress("Electron_dxy", &Electron_dxy );
        _inputTree->SetBranchAddress("Electron_dxyErr", &Electron_dxyErr );
        _inputTree->SetBranchAddress("Electron_dz", &Electron_dz );
        _inputTree->SetBranchAddress("Electron_dzErr", &Electron_dzErr );
        _inputTree->SetBranchAddress("Electron_deltaEtaSC", &Electron_deltaEtaSC );
        
        _inputTree->SetBranchAddress("Electron_cutBased", &Electron_cutBased );
        _inputTree->SetBranchAddress("Electron_cutBased_HEEP", &Electron_cutBased_HEEP );
        //_inputTree->SetBranchAddress("Electron_cutBased_Fall17_V1", &Electron_cutBased_Fall17_V1 );
        //_inputTree->SetBranchAddress("Electron_mvaFall17V1Iso", &Electron_mvaFall17V1Iso );
        //_inputTree->SetBranchAddress("Electron_mvaFall17V1noIso", &Electron_mvaFall17V1noIso );
        _inputTree->SetBranchAddress("Electron_mvaFall17V2Iso", &Electron_mvaFall17V2Iso );
        _inputTree->SetBranchAddress("Electron_mvaFall17V2noIso", &Electron_mvaFall17V2noIso );
        _inputTree->SetBranchAddress("Electron_pfRelIso03_all", &Electron_pfRelIso03_all );
        _inputTree->SetBranchAddress("Electron_pfRelIso03_chg", &Electron_pfRelIso03_chg );
        _inputTree->SetBranchAddress("Electron_mvaFall17V2Iso_WP80", &Electron_mvaFall17V2Iso_WP80 );
        _inputTree->SetBranchAddress("Electron_mvaFall17V2Iso_WP90", &Electron_mvaFall17V2Iso_WP90 );
        _inputTree->SetBranchAddress("Electron_mvaFall17V2Iso_WPL", &Electron_mvaFall17V2Iso_WPL );
        _inputTree->SetBranchAddress("Electron_mvaFall17V2noIso_WP80", &Electron_mvaFall17V2noIso_WP80 );
        _inputTree->SetBranchAddress("Electron_mvaFall17V2noIso_WP90", &Electron_mvaFall17V2noIso_WP90 );

        
        //-----------------------------------------------------------------------------------------------------------------------
        _inputTree->SetBranchAddress("nMuon", &nMuon );
        _inputTree->SetBranchAddress("Muon_pt", &Muon_pt );
        _inputTree->SetBranchAddress("Muon_eta", &Muon_eta );
        _inputTree->SetBranchAddress("Muon_phi", &Muon_phi );
        _inputTree->SetBranchAddress("Muon_charge", &Muon_charge );
        _inputTree->SetBranchAddress("Muon_mass", &Muon_mass );
        _inputTree->SetBranchAddress("Muon_pdgId", &Muon_pdgId );
        _inputTree->SetBranchAddress("Muon_miniPFRelIso_all", &Muon_miniPFRelIso_all );
        _inputTree->SetBranchAddress("Muon_miniPFRelIso_chg", &Muon_miniPFRelIso_chg );
        _inputTree->SetBranchAddress("Muon_jetRelIso", &Muon_jetRelIso );
        _inputTree->SetBranchAddress("Muon_jetIdx", &Muon_jetIdx );
        _inputTree->SetBranchAddress("Muon_isPFcand", &Muon_isPFcand );
        _inputTree->SetBranchAddress("Muon_isGlobal", &Muon_isGlobal );
        _inputTree->SetBranchAddress("Muon_isTracker", &Muon_isTracker );
        _inputTree->SetBranchAddress("Muon_dxy", &Muon_dxy );
        _inputTree->SetBranchAddress("Muon_dxyErr", &Muon_dxyErr );
        _inputTree->SetBranchAddress("Muon_dz", &Muon_dz );
        _inputTree->SetBranchAddress("Muon_dzErr", &Muon_dzErr );
        _inputTree->SetBranchAddress("Muon_nTrackerLayers", &Muon_nTrackerLayers );
        
        _inputTree->SetBranchAddress("Muon_looseId", &Muon_looseId );
        _inputTree->SetBranchAddress("Muon_mediumId", &Muon_mediumId );
        _inputTree->SetBranchAddress("Muon_tightId", &Muon_tightId );
        _inputTree->SetBranchAddress("Muon_multiIsoId", &Muon_multiIsoId );
        _inputTree->SetBranchAddress("Muon_softId", &Muon_softId );
        _inputTree->SetBranchAddress("Muon_tkIsoId", &Muon_tkIsoId );
        _inputTree->SetBranchAddress("Muon_tkRelIso", &Muon_tkRelIso );
        _inputTree->SetBranchAddress("Muon_pfRelIso04_all", &Muon_pfRelIso04_all );
        _inputTree->SetBranchAddress("Muon_pfRelIso03_all", &Muon_pfRelIso03_all );
        _inputTree->SetBranchAddress("Muon_pfRelIso03_chg", &Muon_pfRelIso03_chg );
        _inputTree->SetBranchAddress("Muon_pfIsoId", &Muon_pfIsoId );
        _inputTree->SetBranchAddress("Muon_highPtId", &Muon_highPtId );


        //-----------------------------------------------------------------------------------------------------------------------
        _inputTree->SetBranchAddress("nTau", &nTau );                       //[50]  7    1
        _inputTree->SetBranchAddress("Tau_chargedIso", &Tau_chargedIso );
        _inputTree->SetBranchAddress("Tau_dxy", &Tau_dxy );
        _inputTree->SetBranchAddress("Tau_dz", &Tau_dz );
        _inputTree->SetBranchAddress("Tau_eta", &Tau_eta );
        _inputTree->SetBranchAddress("Tau_leadTkDeltaEta", &Tau_leadTkDeltaEta );
        _inputTree->SetBranchAddress("Tau_leadTkDeltaPhi", &Tau_leadTkDeltaPhi );
        _inputTree->SetBranchAddress("Tau_leadTkPtOverTauPt", &Tau_leadTkPtOverTauPt );
        _inputTree->SetBranchAddress("Tau_mass", &Tau_mass );
        _inputTree->SetBranchAddress("Tau_neutralIso", &Tau_neutralIso );
        _inputTree->SetBranchAddress("Tau_phi", &Tau_phi );
        _inputTree->SetBranchAddress("Tau_pt", &Tau_pt );
        _inputTree->SetBranchAddress("Tau_puCorr", &Tau_puCorr );
        _inputTree->SetBranchAddress("Tau_charge", &Tau_charge );
        _inputTree->SetBranchAddress("Tau_decayMode", &Tau_decayMode );
        _inputTree->SetBranchAddress("Tau_jetIdx", &Tau_jetIdx );
        _inputTree->SetBranchAddress("Tau_idAntiEle", &Tau_idAntiEle );
        _inputTree->SetBranchAddress("Tau_idAntiEle2018", &Tau_idAntiEle2018 );
        _inputTree->SetBranchAddress("Tau_idAntiEleDeadECal", &Tau_idAntiEleDeadECal );
        _inputTree->SetBranchAddress("Tau_idAntiMu", &Tau_idAntiMu );
        _inputTree->SetBranchAddress("Tau_idDecayMode", &Tau_idDecayMode );
        _inputTree->SetBranchAddress("Tau_idDecayModeNewDMs", &Tau_idDecayModeNewDMs );
        _inputTree->SetBranchAddress("Tau_idDeepTau2017v2p1VSe", &Tau_idDeepTau2017v2p1VSe );
        _inputTree->SetBranchAddress("Tau_idDeepTau2017v2p1VSjet", &Tau_idDeepTau2017v2p1VSjet );
        _inputTree->SetBranchAddress("Tau_idDeepTau2017v2p1VSmu", &Tau_idDeepTau2017v2p1VSmu );
        _inputTree->SetBranchAddress("Tau_idMVAnewDM2017v2", &Tau_idMVAnewDM2017v2 );


        //-----------------------------------------------------------------------------------------------------------------------
        _inputTree->SetBranchAddress("nCorrT1METJet", &nCorrT1METJet );
        _inputTree->SetBranchAddress("CorrT1METJet_area", &CorrT1METJet_area );
        _inputTree->SetBranchAddress("CorrT1METJet_eta", &CorrT1METJet_eta );
        _inputTree->SetBranchAddress("CorrT1METJet_muonSubtrFactor", &CorrT1METJet_muonSubtrFactor );
        _inputTree->SetBranchAddress("CorrT1METJet_phi", &CorrT1METJet_phi );
        _inputTree->SetBranchAddress("CorrT1METJet_rawPt", &CorrT1METJet_rawPt );
        
        _inputTree->SetBranchAddress("nJet", &nJet );
        _inputTree->SetBranchAddress("Jet_pt", &Jet_pt );
        _inputTree->SetBranchAddress("Jet_eta", &Jet_eta );
        _inputTree->SetBranchAddress("Jet_phi", &Jet_phi );
        _inputTree->SetBranchAddress("Jet_area", &Jet_area );
        _inputTree->SetBranchAddress("Jet_mass", &Jet_mass );
        _inputTree->SetBranchAddress("Jet_jetId", &Jet_jetId );
        _inputTree->SetBranchAddress("Jet_chEmEF", &Jet_chEmEF );
        _inputTree->SetBranchAddress("Jet_chHEF", &Jet_chHEF );
        _inputTree->SetBranchAddress("Jet_neEmEF", &Jet_neEmEF );
        _inputTree->SetBranchAddress("Jet_neHEF", &Jet_neHEF );
        _inputTree->SetBranchAddress("Jet_btagDeepB", &Jet_btagDeepB );
        _inputTree->SetBranchAddress("Jet_btagDeepFlavB", &Jet_btagDeepFlavB );
        _inputTree->SetBranchAddress("Jet_puIdDisc", &Jet_puIdDisc );
        _inputTree->SetBranchAddress("Jet_puId", &Jet_puId );
        _inputTree->SetBranchAddress("Jet_qgl", &Jet_qgl );
        _inputTree->SetBranchAddress("Jet_nConstituents", &Jet_nConstituents );
        _inputTree->SetBranchAddress("Jet_nElectrons", &Jet_nElectrons );
        _inputTree->SetBranchAddress("Jet_nMuons", &Jet_nMuons );
        _inputTree->SetBranchAddress("Jet_rawFactor", &Jet_rawFactor );
        _inputTree->SetBranchAddress("Jet_muonSubtrFactor", &Jet_muonSubtrFactor );
        
        _inputTree->SetBranchAddress("nFatJet", &nFatJet );
        _inputTree->SetBranchAddress("FatJet_eta", &FatJet_eta );
        _inputTree->SetBranchAddress("FatJet_phi", &FatJet_phi );
        _inputTree->SetBranchAddress("FatJet_pt", &FatJet_pt );
        _inputTree->SetBranchAddress("FatJet_mass", &FatJet_mass );
        _inputTree->SetBranchAddress("FatJet_deepTagMD_ZHbbvsQCD", &FatJet_deepTagMD_ZHbbvsQCD );
        _inputTree->SetBranchAddress("FatJet_deepTagMD_ZbbvsQCD", &FatJet_deepTagMD_ZbbvsQCD );
        _inputTree->SetBranchAddress("FatJet_deepTagMD_HbbvsQCD", &FatJet_deepTagMD_HbbvsQCD );
        _inputTree->SetBranchAddress("FatJet_deepTag_H", &FatJet_deepTag_H );
        _inputTree->SetBranchAddress("FatJet_deepTag_ZvsQCD", &FatJet_deepTag_ZvsQCD );
        _inputTree->SetBranchAddress("FatJet_deepTagMD_bbvsLight", &FatJet_deepTagMD_bbvsLight );
        _inputTree->SetBranchAddress("FatJet_btagDDBvL", &FatJet_btagDDBvL );

        //-----------------------------------------------------------------------------------------------------------------------
        _inputTree->SetBranchAddress("MET_phi", &MET_phi );
        _inputTree->SetBranchAddress("MET_pt", &MET_pt );
        _inputTree->SetBranchAddress("MET_MetUnclustEnUpDeltaX", &MET_MetUnclustEnUpDeltaX );
        _inputTree->SetBranchAddress("MET_MetUnclustEnUpDeltaY", &MET_MetUnclustEnUpDeltaY );    
        _inputTree->SetBranchAddress("MET_covXX", &MET_covXX );
        _inputTree->SetBranchAddress("MET_covXY", &MET_covXY );
        _inputTree->SetBranchAddress("MET_covYY", &MET_covYY );
        _inputTree->SetBranchAddress("MET_significance", &MET_significance );

        
        //-----------------------------------------------------------------------------------------------------------------------
        _inputTree->SetBranchAddress("PV_ndof", &PV_ndof );
        _inputTree->SetBranchAddress("PV_x", &PV_x );
        _inputTree->SetBranchAddress("PV_y", &PV_y );
        _inputTree->SetBranchAddress("PV_z", &PV_z );
        _inputTree->SetBranchAddress("PV_chi2", &PV_chi2 );
        _inputTree->SetBranchAddress("PV_score", &PV_score );
        _inputTree->SetBranchAddress("PV_npvs", &PV_npvs );
        _inputTree->SetBranchAddress("PV_npvsGood", &PV_npvsGood );
        
        
        //-----------------------------------------------------------------------------------------------------------------------
        _inputTree->SetBranchAddress("nSV", &nSV );
        _inputTree->SetBranchAddress("SV_dlen", &SV_dlen );
        _inputTree->SetBranchAddress("SV_dlenSig", &SV_dlenSig );
        _inputTree->SetBranchAddress("SV_dxy", &SV_dxy );
        _inputTree->SetBranchAddress("SV_dxySig", &SV_dxySig );
        _inputTree->SetBranchAddress("SV_pAngle", &SV_pAngle );
        _inputTree->SetBranchAddress("SV_chi2", &SV_chi2 );
        _inputTree->SetBranchAddress("SV_eta", &SV_eta );
        _inputTree->SetBranchAddress("SV_mass", &SV_mass );
        _inputTree->SetBranchAddress("SV_ndof", &SV_ndof );
        _inputTree->SetBranchAddress("SV_phi", &SV_phi );
        _inputTree->SetBranchAddress("SV_pt", &SV_pt );
        _inputTree->SetBranchAddress("SV_x", &SV_x );
        _inputTree->SetBranchAddress("SV_y", &SV_y );
        _inputTree->SetBranchAddress("SV_z", &SV_z );
        //_inputTree->SetBranchAddress("SV_ntracks", &SV_ntracks ); // not found in data 2016
        
        
        //-----------------------------------------------------------------------------------------------------------------------
        _inputTree->SetBranchAddress("nIsoTrack", &nIsoTrack );
        _inputTree->SetBranchAddress("IsoTrack_dxy", &IsoTrack_dxy );
        _inputTree->SetBranchAddress("IsoTrack_dz", &IsoTrack_dz );
        _inputTree->SetBranchAddress("IsoTrack_eta", &IsoTrack_eta );
        _inputTree->SetBranchAddress("IsoTrack_pfRelIso03_all", &IsoTrack_pfRelIso03_all );
        _inputTree->SetBranchAddress("IsoTrack_pfRelIso03_chg", &IsoTrack_pfRelIso03_chg );
        _inputTree->SetBranchAddress("IsoTrack_phi", &IsoTrack_phi );
        _inputTree->SetBranchAddress("IsoTrack_pt", &IsoTrack_pt );
        _inputTree->SetBranchAddress("IsoTrack_miniPFRelIso_all", &IsoTrack_miniPFRelIso_all );
        _inputTree->SetBranchAddress("IsoTrack_miniPFRelIso_chg", &IsoTrack_miniPFRelIso_chg );
        _inputTree->SetBranchAddress("IsoTrack_fromPV", &IsoTrack_fromPV );
        _inputTree->SetBranchAddress("IsoTrack_pdgId", &IsoTrack_pdgId );
        _inputTree->SetBranchAddress("IsoTrack_isHighPurityTrack", &IsoTrack_isHighPurityTrack );
        _inputTree->SetBranchAddress("IsoTrack_isPFcand", &IsoTrack_isPFcand );
        _inputTree->SetBranchAddress("IsoTrack_isFromLostTrack", &IsoTrack_isFromLostTrack );
        

        //-----------------------------------------------------------------------------------------------------------------------
        _inputTree->SetBranchAddress("Flag_goodVertices", &Flag_goodVertices );
        _inputTree->SetBranchAddress("Flag_globalSuperTightHalo2016Filter", &Flag_globalSuperTightHalo2016Filter );
        _inputTree->SetBranchAddress("Flag_HBHENoiseFilter", &Flag_HBHENoiseFilter );
        _inputTree->SetBranchAddress("Flag_HBHENoiseIsoFilter", &Flag_HBHENoiseIsoFilter );
        _inputTree->SetBranchAddress("Flag_EcalDeadCellTriggerPrimitiveFilter", &Flag_EcalDeadCellTriggerPrimitiveFilter );
        _inputTree->SetBranchAddress("Flag_BadPFMuonFilter", &Flag_BadPFMuonFilter );
        _inputTree->SetBranchAddress("Flag_BadPFMuonDzFilter", &Flag_BadPFMuonDzFilter );
        _inputTree->SetBranchAddress("Flag_ecalBadCalibFilter", &Flag_ecalBadCalibFilter );
        _inputTree->SetBranchAddress("Flag_eeBadScFilter", &Flag_eeBadScFilter );
        
        //-----------------------------------------------------------------------------------------------------------------------
        HLT_Ele27_WPTight_Gsf = false;
        HLT_Ele115_CaloIdVT_GsfTrkIdT = false;
        HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ = false;
        HLT_IsoMu24 = false;
        HLT_IsoTkMu24 = false;
        HLT_Mu50 = false;
        HLT_TkMu50 = false;
        HLT_PFMET300 = false;
        HLT_MET200 = false;
        HLT_PFHT300_PFMET110 = false;
        HLT_PFMET170_HBHECleaned = false;
        HLT_PFMET120_PFMHT120_IDTight = false;
        HLT_PFMETNoMu120_PFMHTNoMu120_IDTight = false;
        HLT_Photon175 = false;
        HLT_DoubleEle33_CaloIdL_GsfTrkIdVL = false;
        HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ = false;
        HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ = false;
        HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ = false;
        HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ = false;
        HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL = false;
        HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL = false;
        HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL = false;
        HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL = false;
        HLT_Ele35_WPTight_Gsf = false;
        HLT_IsoMu27 = false;
        HLT_OldMu100 = false;
        HLT_TkMu100 = false;
        HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL = false;
        HLT_DoubleEle33_CaloIdL_MW = false;
        HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8 = false;
        HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8 = false;
        HLT_PFMET200_HBHECleaned = false;
        HLT_PFMET200_HBHE_BeamHaloCleaned = false;
        HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned = false;
        HLT_PFMET120_PFMHT120_IDTight_PFHT60 = false;
        HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60 = false;
        HLT_PFHT500_PFMET100_PFMHT100_IDTight = false;
        HLT_PFHT700_PFMET85_PFMHT85_IDTight = false;
        HLT_PFHT800_PFMET75_PFMHT75_IDTight = false;
        HLT_Photon200 = false;
        HLT_Ele32_WPTight_Gsf = false;
        HLT_DoubleEle25_CaloIdL_MW = false;

        if( dataset_year == "16" ){
            _inputTree->SetBranchAddress("HLT_Ele27_WPTight_Gsf", &HLT_Ele27_WPTight_Gsf );
            _inputTree->SetBranchAddress("HLT_Ele115_CaloIdVT_GsfTrkIdT", &HLT_Ele115_CaloIdVT_GsfTrkIdT );
            _inputTree->SetBranchAddress("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", &HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ );
            //_inputTree->SetBranchAddress("HLT_DoubleEle37_Ele27_CaloIdL_GsfTrkIdVL", &HLT_DoubleEle37_Ele27_CaloIdL_GsfTrkIdVL );
            _inputTree->SetBranchAddress("HLT_IsoMu24", &HLT_IsoMu24 );
            _inputTree->SetBranchAddress("HLT_IsoTkMu24", &HLT_IsoTkMu24 );
            _inputTree->SetBranchAddress("HLT_Mu50", &HLT_Mu50 );
            _inputTree->SetBranchAddress("HLT_TkMu50", &HLT_TkMu50 );
            //_inputTree->SetBranchAddress("HLT_Mu30_TkMu11", &HLT_Mu30_TkMu11 );
            _inputTree->SetBranchAddress("HLT_PFMET300", &HLT_PFMET300 );
            _inputTree->SetBranchAddress("HLT_MET200", &HLT_MET200 );
            _inputTree->SetBranchAddress("HLT_PFHT300_PFMET110", &HLT_PFHT300_PFMET110 );
            _inputTree->SetBranchAddress("HLT_PFMET170_HBHECleaned", &HLT_PFMET170_HBHECleaned );
            _inputTree->SetBranchAddress("HLT_PFMET120_PFMHT120_IDTight", &HLT_PFMET120_PFMHT120_IDTight );
            _inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight );
            _inputTree->SetBranchAddress("HLT_Photon175", &HLT_Photon175 );
            _inputTree->SetBranchAddress("HLT_DoubleEle33_CaloIdL_GsfTrkIdVL", &HLT_DoubleEle33_CaloIdL_GsfTrkIdVL );
            
            //if( (dataset_group == "Data") && (dataset_era == "H") ){
            _inputTree->SetBranchAddress("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ", &HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ );
            _inputTree->SetBranchAddress("HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ", &HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ );
            _inputTree->SetBranchAddress("HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ", &HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ );
            _inputTree->SetBranchAddress("HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", &HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ );
            //}else{
            _inputTree->SetBranchAddress("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL", &HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL );
            _inputTree->SetBranchAddress("HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL", &HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL );
            _inputTree->SetBranchAddress("HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL", &HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL );
            _inputTree->SetBranchAddress("HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL", &HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL );
            //}
        }else if( dataset_year == "17" ){
            _inputTree->SetBranchAddress("HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ", &HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ ); 
            _inputTree->SetBranchAddress("HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", &HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ );
            _inputTree->SetBranchAddress("HLT_Ele35_WPTight_Gsf", &HLT_Ele35_WPTight_Gsf );
            _inputTree->SetBranchAddress("HLT_IsoMu27", &HLT_IsoMu27 );
            _inputTree->SetBranchAddress("HLT_Mu50", &HLT_Mu50 );
            _inputTree->SetBranchAddress("HLT_OldMu100", &HLT_OldMu100 );
            _inputTree->SetBranchAddress("HLT_TkMu100", &HLT_TkMu100 );
            _inputTree->SetBranchAddress("HLT_Ele115_CaloIdVT_GsfTrkIdT", &HLT_Ele115_CaloIdVT_GsfTrkIdT );
            _inputTree->SetBranchAddress("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL", &HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL );
            _inputTree->SetBranchAddress("HLT_DoubleEle33_CaloIdL_MW", &HLT_DoubleEle33_CaloIdL_MW ); 
            _inputTree->SetBranchAddress("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8", &HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8 ); 
            _inputTree->SetBranchAddress("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8", &HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8 );
            _inputTree->SetBranchAddress("HLT_PFMET200_HBHECleaned", &HLT_PFMET200_HBHECleaned ); 
            _inputTree->SetBranchAddress("HLT_PFMET200_HBHE_BeamHaloCleaned", &HLT_PFMET200_HBHE_BeamHaloCleaned ); 
            _inputTree->SetBranchAddress("HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned", &HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned ); 
            _inputTree->SetBranchAddress("HLT_PFMET120_PFMHT120_IDTight", &HLT_PFMET120_PFMHT120_IDTight ); 
            _inputTree->SetBranchAddress("HLT_PFMET120_PFMHT120_IDTight_PFHT60", &HLT_PFMET120_PFMHT120_IDTight_PFHT60 ); 
            _inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight ); 
            _inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60 ); 
            _inputTree->SetBranchAddress("HLT_PFHT500_PFMET100_PFMHT100_IDTight", &HLT_PFHT500_PFMET100_PFMHT100_IDTight ); 
            _inputTree->SetBranchAddress("HLT_PFHT700_PFMET85_PFMHT85_IDTight", &HLT_PFHT700_PFMET85_PFMHT85_IDTight ); 
            _inputTree->SetBranchAddress("HLT_PFHT800_PFMET75_PFMHT75_IDTight", &HLT_PFHT800_PFMET75_PFMHT75_IDTight );
            _inputTree->SetBranchAddress("HLT_Photon200", &HLT_Photon200 );
        }else if( dataset_year == "18" ){
            _inputTree->SetBranchAddress("HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ", &HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ ); 
            _inputTree->SetBranchAddress("HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL", &HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL ); 
            _inputTree->SetBranchAddress("HLT_IsoMu24", &HLT_IsoMu24 ); 
            _inputTree->SetBranchAddress("HLT_Mu50", &HLT_Mu50 );
            _inputTree->SetBranchAddress("HLT_OldMu100", &HLT_OldMu100 );
            _inputTree->SetBranchAddress("HLT_TkMu100", &HLT_TkMu100 );
            _inputTree->SetBranchAddress("HLT_Ele32_WPTight_Gsf", &HLT_Ele32_WPTight_Gsf );
            _inputTree->SetBranchAddress("HLT_Ele115_CaloIdVT_GsfTrkIdT", &HLT_Ele115_CaloIdVT_GsfTrkIdT );
            _inputTree->SetBranchAddress("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL", &HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL ); 
            _inputTree->SetBranchAddress("HLT_DoubleEle25_CaloIdL_MW", &HLT_DoubleEle25_CaloIdL_MW ); 
            _inputTree->SetBranchAddress("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8", &HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8 ); 
            _inputTree->SetBranchAddress("HLT_PFMET200_HBHECleaned", &HLT_PFMET200_HBHECleaned ); 
            _inputTree->SetBranchAddress("HLT_PFMET200_HBHE_BeamHaloCleaned", &HLT_PFMET200_HBHE_BeamHaloCleaned ); 
            _inputTree->SetBranchAddress("HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned", &HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned ); 
            _inputTree->SetBranchAddress("HLT_PFMET120_PFMHT120_IDTight", &HLT_PFMET120_PFMHT120_IDTight ); 
            _inputTree->SetBranchAddress("HLT_PFMET120_PFMHT120_IDTight_PFHT60", &HLT_PFMET120_PFMHT120_IDTight_PFHT60 ); 
            _inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight ); 
            _inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60 ); 
            _inputTree->SetBranchAddress("HLT_PFHT500_PFMET100_PFMHT100_IDTight", &HLT_PFHT500_PFMET100_PFMHT100_IDTight ); 
            _inputTree->SetBranchAddress("HLT_PFHT700_PFMET85_PFMHT85_IDTight", &HLT_PFHT700_PFMET85_PFMHT85_IDTight ); 
            _inputTree->SetBranchAddress("HLT_PFHT800_PFMET75_PFMHT75_IDTight", &HLT_PFHT800_PFMET75_PFMHT75_IDTight );
        }
        
        //-----------------------------------------------------------------------------------------------------------------------
        if( dataset_group != "Data" ) {
            if( _ANALYSIS != "OPENDATA" ) _inputTree->SetBranchAddress("genWeight", &genWeight );
            
            _inputTree->SetBranchAddress("Electron_genPartIdx", &Electron_genPartIdx );
            _inputTree->SetBranchAddress("Muon_genPartIdx", &Muon_genPartIdx );
            _inputTree->SetBranchAddress("Jet_genJetIdx", &Jet_genJetIdx );
            _inputTree->SetBranchAddress("Jet_hadronFlavour", &Jet_hadronFlavour );
            
            _inputTree->SetBranchAddress("Pileup_nTrueInt", &Pileup_nTrueInt );
            _inputTree->SetBranchAddress("Pileup_nPU", &Pileup_nPU );
            
            _inputTree->SetBranchAddress("nGenJet", &nGenJet );
            _inputTree->SetBranchAddress("GenJet_eta", &GenJet_eta );
            _inputTree->SetBranchAddress("GenJet_phi", &GenJet_phi );
            _inputTree->SetBranchAddress("GenJet_pt", &GenJet_pt );
            _inputTree->SetBranchAddress("GenJet_partonFlavour", &GenJet_partonFlavour );
            
            _inputTree->SetBranchAddress("GenMET_phi", &GenMET_phi );
            _inputTree->SetBranchAddress("GenMET_pt", &GenMET_pt );
            
            _inputTree->SetBranchAddress("nGenPart", &nGenPart );
            _inputTree->SetBranchAddress("GenPart_eta", &GenPart_eta );
            _inputTree->SetBranchAddress("GenPart_mass", &GenPart_mass );
            _inputTree->SetBranchAddress("GenPart_phi", &GenPart_phi );
            _inputTree->SetBranchAddress("GenPart_pt", &GenPart_pt );
            _inputTree->SetBranchAddress("GenPart_genPartIdxMother", &GenPart_genPartIdxMother );
            _inputTree->SetBranchAddress("GenPart_pdgId", &GenPart_pdgId );
            _inputTree->SetBranchAddress("GenPart_status", &GenPart_status );
            _inputTree->SetBranchAddress("GenPart_statusFlags", &GenPart_statusFlags );
            
            _inputTree->SetBranchAddress("LHEWeight_originalXWGTUP", &LHEWeight_originalXWGTUP );
            _inputTree->SetBranchAddress("nLHEPdfWeight", &nLHEPdfWeight );
            _inputTree->SetBranchAddress("LHEPdfWeight", &LHEPdfWeight );
            _inputTree->SetBranchAddress("nLHEScaleWeight", &nLHEScaleWeight );
            _inputTree->SetBranchAddress("LHEScaleWeight", &LHEScaleWeight );
            _inputTree->SetBranchAddress("nPSWeight", &nPSWeight );
            _inputTree->SetBranchAddress("PSWeight", &PSWeight );
            _inputTree->SetBranchAddress("nLHEReweightingWeight", &nLHEReweightingWeight );
            _inputTree->SetBranchAddress("LHEReweightingWeight", &LHEReweightingWeight );
            
            _inputTree->SetBranchAddress("LHE_HT", &LHE_HT );
            _inputTree->SetBranchAddress("LHE_Vpt", &LHE_Vpt );
            _inputTree->SetBranchAddress("LHE_Njets", &LHE_Njets );

            _inputTree->SetBranchAddress("L1PreFiringWeight_Dn", &L1PreFiringWeight_Dn );
            _inputTree->SetBranchAddress("L1PreFiringWeight_Nom", &L1PreFiringWeight_Nom );
            _inputTree->SetBranchAddress("L1PreFiringWeight_Up", &L1PreFiringWeight_Up );
        }

        //-----------------------------------------------------------------------------------------------------------------------
        if( _ANALYSIS == "OPENDATA" ) {
            genWeight = 1.;
            _inputTree->SetBranchAddress("Tau_relIso_all", &Tau_relIso_all );
            _inputTree->SetBranchAddress("Tau_genPartIdx", &Tau_genPartIdx );
            _inputTree->SetBranchAddress("Tau_idIsoRaw", &Tau_idIsoRaw );
            _inputTree->SetBranchAddress("Tau_idIsoVLoose", &Tau_idIsoVLoose );
            _inputTree->SetBranchAddress("Tau_idIsoLoose", &Tau_idIsoLoose );
            _inputTree->SetBranchAddress("Tau_idIsoMedium", &Tau_idIsoMedium );
            _inputTree->SetBranchAddress("Tau_idIsoTight", &Tau_idIsoTight );
            _inputTree->SetBranchAddress("Tau_idAntiEleLoose", &Tau_idAntiEleLoose );
            _inputTree->SetBranchAddress("Tau_idAntiEleMedium", &Tau_idAntiEleMedium );
            _inputTree->SetBranchAddress("Tau_idAntiEleTight", &Tau_idAntiEleTight );
            _inputTree->SetBranchAddress("Tau_idAntiMuLoose", &Tau_idAntiMuLoose );
            _inputTree->SetBranchAddress("Tau_idAntiMuMedium", &Tau_idAntiMuMedium );
            _inputTree->SetBranchAddress("Tau_idAntiMuTight", &Tau_idAntiMuTight );

            _inputTree->SetBranchAddress("Jet_btag", &Jet_btag );

            _inputTree->SetBranchAddress("MET_sumet", &MET_sumet );
            _inputTree->SetBranchAddress("MET_CovXX", &MET_CovXX );
            _inputTree->SetBranchAddress("MET_CovXY", &MET_CovXY );
            _inputTree->SetBranchAddress("MET_CovYY", &MET_CovYY );

            _inputTree->SetBranchAddress("HLT_IsoMu24", &HLT_IsoMu24 );
            _inputTree->SetBranchAddress("HLT_IsoMu24_eta2p1", &HLT_IsoMu24_eta2p1 );
            _inputTree->SetBranchAddress("HLT_IsoMu17_eta2p1_LooseIsoPFTau20", &HLT_IsoMu17_eta2p1_LooseIsoPFTau20 );
        }
    }

    return true;
}


//---------------------------------------------------------------------------------------------------------------
// RunEventLoop
//---------------------------------------------------------------------------------------------------------------
void HEPHero::RunEventLoop( int ControlEntries ) {

    
    //======GET NUMBER OF EVENTS===================================================================
    if( ControlEntries < 0 ){
        if( _ANALYSIS == "GEN" ){
            // HEPHeroGEN
            ifstream in_file( _inputFileNames.at(0), ios::in );
            string line;

            int E_counts = 0;
            int C_counts = 0;
            int F_counts = 0;
            int N_counts = 0;
            while(std::getline( in_file, line)){
                if( line.substr(0,2) == "E " ) E_counts += 1;
                if( (line.substr(0,2) == "N ") && (N_counts == 0) ){ 
                    string delimiter = " ";
                    size_t pos = line.find(delimiter);
                    string line_part2 = line.erase(0, pos + delimiter.length());
                    pos = line_part2.find(delimiter);
                    string N_weights_str = line_part2.substr(0, pos);
                    _N_PS_weights = 18; //atoi(N_weights_str.c_str());
                    N_counts += 1;
                }
                if( line.substr(0,2) == "C " ) C_counts += 1;
                if( line.substr(0,2) == "F " ) F_counts += 1;
            }
            _has_xsec = false;
            _has_pdf = false;
            if( C_counts == E_counts ) _has_xsec = true;
            if( F_counts == E_counts ) _has_pdf = true;
            
            _NumberEntries = E_counts;
        }else{
            _NumberEntries = _inputTree -> GetEntries();
        }
        if( _NumberEntries > _NumMaxEvents && _NumMaxEvents > 0 ) _NumberEntries = _NumMaxEvents;
    }

    
    //======PRINT INFO ABOUT THE SELECTION PROCESS=================================================
    cout << endl;
    cout << endl;
    cout << "===================================================================================" << endl;
    cout << "RUNNING SELECTION " << _SELECTION << " ON " << _datasetName << " FILES " << _Files << " (SYSID=" << to_string(_sysID_lateral) << ", Universe=" << to_string(_Universe) << ")" << endl; 
    cout << "===================================================================================" << endl;
    cout << "Number of files: " << _inputFileNames.size() << endl;
    cout << "Number of entries considered: " << _NumberEntries << endl;
    cout << "-----------------------------------------------------------------------------------" << endl;
    cout << endl;

    if( _sysID_lateral == 0 ) {
        _CutflowFileName = _outputDirectory + "cutflow.txt";
        _CutflowFile.open( _CutflowFileName.c_str(), ios::out );
        _CutflowFile << "===========================================================================================" << endl;
        _CutflowFile << "RUNNING SELECTION " << _SELECTION << " ON " << _datasetName << " FILES " << _Files << endl; 
        _CutflowFile << "===========================================================================================" << endl;
        _CutflowFile << "Ntuples path:" << endl;
        int pos = _inputFileNames.at(0).find("00/"); 
        string inputDirectory = _inputFileNames.at(0).substr(0,pos+2);
        _CutflowFile << inputDirectory << endl;
        _CutflowFile << "Number of files: " << _inputFileNames.size() << endl;
        _CutflowFile << "Number of entries considered: " << setprecision(12) << _NumberEntries << endl;
        _CutflowFile.close();
    }
    

    //======VERTICAL SYSTEMATICS SIZE==============================================================
    if( _ANALYSIS == "OPENDATA" ) VerticalSysSizesOD( );
    else if( _ANALYSIS != "GEN" ) VerticalSysSizes( );
    
    //======PRE-OBJECTS SETUP======================================================================
    PreObjects();

    //======SETUP SELECTION========================================================================
    if( false );
    // SETUP YOUR SELECTION HERE
    else {
      cout << "Unknown selection requested. Exiting. " << endl;
      return;
    }

    //======SETUP STATISTICAL ERROR================================================================
    vector<double> cutflowOld(50, 0.);
    for( int icut = 0; icut < 50; ++icut ){
        _StatisticalError.push_back(0);
    }
    
    //======SETUP TIMER============================================================================
    int itime = 0;
    int hoursEstimated = 0;
    int minutesEstimated = 0;
    int secondsEstimated = 0;
    time_t timeOld;
    timeOld = time(NULL);
    
    //======LOOP OVER THE EVENTS===================================================================
    SumGenWeights = 0;
    
    for( int i = 0; i < _NumberEntries; ++i ) {
        if( _ANALYSIS == "GEN" ){ 
           if( _ascii_file->failed() ) break;
        }
        
        //======TIMER====================================================================
        int timer_steps;
        if( _ANALYSIS == "GEN" ){ 
            timer_steps = 1000;
        }else if( _ANALYSIS == "OPENDATA" ){
            timer_steps = 500000;
        }else{
            timer_steps = 10000;
        }
        if( (i+1)/timer_steps != itime ){
            int timeEstimated = (_NumberEntries-i)*difftime(time(NULL),timeOld)/(timer_steps*1.);
            hoursEstimated = timeEstimated/3600;
            minutesEstimated = (timeEstimated%3600)/60;
            secondsEstimated = (timeEstimated%3600)%60;
            ++itime;
            timeOld = time(NULL);
        }
        if( _Show_Timer ) {
            cout << "NOW RUNNING EVENT " << setw(8) << i << " | TIME TO FINISH: " << setw(2) << hoursEstimated << " hours " << setw(2) << minutesEstimated << " minutes " << setw(2) << secondsEstimated << " seconds." << "\r"; 
            fflush(stdout);
        }

    
        //======SETUP EVENT==============================================================
        _EventPosition = i;
        if( _ANALYSIS == "GEN" ){ 
            _ascii_file->read_event(_evt);
        }else{
            _inputTree->GetEntry(i);
        }
        
        
        //======RUN OBJECTS SETUP========================================================
        if( !RunObjects() ) continue;
        
        
        //======START EVENT WEIGHT=====================================================
        evtWeight = 1.;
        if( (_ANALYSIS != "GEN") && (dataset_group != "Data") ) evtWeight = genWeight;
        
        
        //======RUN REGION SETUP=========================================================
        bool Selected = true;
        // SET THE REGION OF YOUR SELECTION HERE
        
        
        //======COMPUTE STATISTICAL ERROR================================================
        int icut = 0;
        for ( map<string,double>::iterator it = _cutFlow.begin(); it != _cutFlow.end(); ++it ){
            if( cutflowOld.at(icut) != it->second ){
                _StatisticalError.at(icut) += evtWeight*evtWeight;
                cutflowOld.at(icut) = it->second;
            }
            ++icut;
        }
        
        
        //======GO TO NEXT STEPS ONLY IF THE EVENT IS SELECTED===========================
        if( !Selected ) continue;
        
        
        //======RUN SELECTION ON THE MAIN UNIVERSE=======================================
        if( _sysID_lateral == 0 ){
            
            // CALL YOUR SELECTION HERE
            
        }
              
              
        //======RUN SYSTEMATIC PRODUCTION=================================================
        if( _ANALYSIS == "OPENDATA" ) VerticalSysOD();
        else if( _ANALYSIS != "GEN" ) VerticalSys();
        
        // PRODUCE THE SYSTEMATIC OF YOUR SELECTION HERE

    }
    
    
    //======COMPUTE STATISTICAL ERROR==============================================================
    int icut = 0;
    for ( map<string,double>::iterator it = _cutFlow.begin(); it != _cutFlow.end(); ++it ){
        _StatisticalError.at(icut) = sqrt(_StatisticalError.at(icut));
        ++icut;
    }
    
    return;
} 


//---------------------------------------------------------------------------------------------------------------
// FinishRun
//---------------------------------------------------------------------------------------------------------------
void HEPHero::FinishRun() {

    gErrorIgnoreLevel = kWarning;
    
    
    //======HISTOGRAM PLOT FORMATS=================================================================
    if( _Get_Image_in_EPS ) _plotFormats.push_back(".eps");
    if( _Get_Image_in_PNG ) _plotFormats.push_back(".png");
    if( _Get_Image_in_PDF ) _plotFormats.push_back(".pdf");

    
    if( _sysID_lateral == 0 ) {
        //======PRODUCE 1D AND 2D HISTOGRAMS===========================================================
        TCanvas c("","");
        for( map<string,TH2D>::iterator itr_h = _histograms2D.begin(); itr_h != _histograms2D.end(); ++itr_h ) {
            (*itr_h).second.Draw( (_histograms2DDrawOptions.at((*itr_h).first)).c_str()  );
            for( vector<string>::iterator itr_f = _plotFormats.begin(); itr_f != _plotFormats.end(); ++itr_f ) {
                string thisPlotName = _outputDirectory + (*itr_h).first + (*itr_f);
                c.Print( thisPlotName.c_str() );
            }
        }
    
        for( map<string,TH1D>::iterator itr_h = _histograms1D.begin(); itr_h != _histograms1D.end(); ++itr_h ) {
            (*itr_h).second.Draw( (_histograms1DDrawOptions.at((*itr_h).first)).c_str() );
            for( vector<string>::iterator itr_f = _plotFormats.begin(); itr_f != _plotFormats.end(); ++itr_f ) {
                string thisPlotName = _outputDirectory + (*itr_h).first + (*itr_f);
                c.Print( thisPlotName.c_str() );
            }
            //(*itr_h).second.Sumw2();
        }
    }
        
    //======FINISH SELECTION=======================================================================
    // FINISH YOUR SELECTION HERE
    
    
    //======PRINT INFO ABOUT THE SELECTION PROCESS=================================================
    if( _ANALYSIS == "GEN" ){
        WriteGenCutflowInfo();
    }else{
        WriteCutflowInfo();
    }
    
    
    //======CLOSE ASCII FILE=======================================================================
    // HEPHeroGEN
    if( _ANALYSIS == "GEN" ) _ascii_file->close();
    
    
    //======STORE HISTOGRAMS IN A ROOT FILE========================================================
    if( _sysID_lateral == 0 ) {
        string HistogramsFileName = _outputDirectory + "Histograms.root";
        TFile *fHistOutput = new TFile( HistogramsFileName.c_str(), "RECREATE" );
        for( map<string,TH1D>::iterator itr_h = _histograms1D.begin(); itr_h != _histograms1D.end(); ++itr_h ) {
            (*itr_h).second.Write();
        }
        for( map<string,TH2D>::iterator itr_h = _histograms2D.begin(); itr_h != _histograms2D.end(); ++itr_h ) {
            (*itr_h).second.Write();
        }
        fHistOutput->Close();
    }
    
    
    //======WRITE OUTPUT TREE IN THE OUTPUT ROOT FILE==============================================
    if( _sysID_lateral == 0 ) {
        _outputFile->cd();
        _outputTree->Write();
        _outputFile->Close();
    }
    
    
    //======WRITE OUTPUT INFORMATION IN THE HDF FILE===============================================
    if( _sysID_lateral == 0 ) HDF_write();

    
    //======WRITE OUTPUT SYS IN THE OUTPUT ROOT FILE===============================================
    _sysFile->cd();
    _sysFileJson << "{" << endl;
    
    bool is_first = true;
    
    for( map<string,TH1D>::iterator itr_h = _systematics1D.begin(); itr_h != _systematics1D.end(); ++itr_h ) {
        (*itr_h).second.Write();
        
        if( is_first ){
            is_first = false;
        }else{
            _sysFileJson << ", " << endl;
        }
        _sysFileJson << "\"" << (*itr_h).first << "\": {\"Nbins\": " << (*itr_h).second.GetXaxis()->GetNbins() << ", \"Start\": " << (*itr_h).second.GetXaxis()->GetBinLowEdge(1) << ", \"End\": " << (*itr_h).second.GetXaxis()->GetBinLowEdge( (*itr_h).second.GetXaxis()->GetNbins() + 1) << ", \"Hist\": [";
        
        for( int iBin = 1; iBin <= (*itr_h).second.GetXaxis()->GetNbins(); ++iBin ) {
            if( iBin < (*itr_h).second.GetXaxis()->GetNbins() ){
                _sysFileJson << (*itr_h).second.GetBinContent(iBin) << ", ";
            }else{
                _sysFileJson << (*itr_h).second.GetBinContent(iBin) << "], \"Unc\": [";
            }
        }
        
        for( int iBin = 1; iBin <= (*itr_h).second.GetXaxis()->GetNbins(); ++iBin ) {
            if( iBin < (*itr_h).second.GetXaxis()->GetNbins() ){
                _sysFileJson << (*itr_h).second.GetBinError(iBin) << ", ";
            }else{
                _sysFileJson << (*itr_h).second.GetBinError(iBin) << "]}";
            }
        }
    }
    
    for( map<string,TH2D>::iterator itr_h = _systematics2D.begin(); itr_h != _systematics2D.end(); ++itr_h ) {
        (*itr_h).second.Write();
        
        if( is_first ){
            is_first = false;
        }else{
            _sysFileJson << ", " << endl;
        }
        _sysFileJson << "\"" << (*itr_h).first << "\": {\"NbinsX\": " << (*itr_h).second.GetXaxis()->GetNbins() << ", \"StartX\": " << (*itr_h).second.GetXaxis()->GetBinLowEdge(1) << ", \"EndX\": " << (*itr_h).second.GetXaxis()->GetBinLowEdge( (*itr_h).second.GetXaxis()->GetNbins() + 1) << ", \"NbinsY\": " << (*itr_h).second.GetYaxis()->GetNbins() << ", \"StartY\": " << (*itr_h).second.GetYaxis()->GetBinLowEdge(1) << ", \"EndY\": " << (*itr_h).second.GetYaxis()->GetBinLowEdge( (*itr_h).second.GetYaxis()->GetNbins() + 1) << ", \"Hist\": [";
        
        for( int iBin = 1; iBin <= (*itr_h).second.GetXaxis()->GetNbins(); ++iBin ) {
            _sysFileJson << "[";
            for( int jBin = 1; jBin <= (*itr_h).second.GetYaxis()->GetNbins(); ++jBin ) {
                if( iBin < (*itr_h).second.GetXaxis()->GetNbins() ){
                    if( jBin < (*itr_h).second.GetYaxis()->GetNbins() ){
                        _sysFileJson << (*itr_h).second.GetBinContent(iBin,jBin) << ", ";
                    }else{
                        _sysFileJson << (*itr_h).second.GetBinContent(iBin,jBin) << "], ";
                    }
                }else{
                    if( jBin < (*itr_h).second.GetYaxis()->GetNbins() ){
                        _sysFileJson << (*itr_h).second.GetBinContent(iBin,jBin) << ", ";
                    }else{
                        _sysFileJson << (*itr_h).second.GetBinContent(iBin,jBin) << "]], \"Unc\": [";
                    }
                }
            }
        }
        
        for( int iBin = 1; iBin <= (*itr_h).second.GetXaxis()->GetNbins(); ++iBin ) {
            _sysFileJson << "[";
            for( int jBin = 1; jBin <= (*itr_h).second.GetYaxis()->GetNbins(); ++jBin ) {
                if( iBin < (*itr_h).second.GetXaxis()->GetNbins() ){
                    if( jBin < (*itr_h).second.GetYaxis()->GetNbins() ){
                        _sysFileJson << (*itr_h).second.GetBinError(iBin,jBin) << ", ";
                    }else{
                        _sysFileJson << (*itr_h).second.GetBinError(iBin,jBin) << "], ";
                    }
                }else{
                    if( jBin < (*itr_h).second.GetYaxis()->GetNbins() ){
                        _sysFileJson << (*itr_h).second.GetBinError(iBin,jBin) << ", ";
                    }else{
                        _sysFileJson << (*itr_h).second.GetBinError(iBin,jBin) << "]]}";
                    }
                }
            }
        }
        
    }
    
    _sysFileJson << "}" << endl;
    _sysFile->Close();
    
    
    //======END TIME===============================================================================
    _end = time(NULL);
    int time_elapsed = difftime(_end,_begin);
    int hours = time_elapsed/3600;
    int minutes = (time_elapsed%3600)/60;
    int seconds = (time_elapsed%3600)%60;
    
    if( _sysID_lateral == 0 ) {
        _CutflowFile.open( _CutflowFileName.c_str(), ios::app );
        _CutflowFile << "-----------------------------------------------------------------------------------" << endl;
        _CutflowFile << "Time to process the selection: " << hours << " hours " << minutes << " minutes " << seconds << " seconds." << endl;
        _CutflowFile << "-----------------------------------------------------------------------------------" << endl;
        _CutflowFile.close();
    }
    
    //======DELETE POINTERS========================================================================
    delete _inputTree;
    //delete _outputTree;
    //if( _sysID_lateral == 0 ) delete _outputFile;
    delete _sysFile;
    
    
   

}


