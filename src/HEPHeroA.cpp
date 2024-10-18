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
        if( key == "DATA_LUMI_TAGS_UNC"         )   DATA_LUMI_TAGS_UNC.push_back( value );
        if( key == "DATA_LUMI_VALUES_UNC"       )   DATA_LUMI_VALUES_UNC.push_back( atof(value.c_str()) );

        FillControlVariables( key, value);
        
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
            dataset_era = _datasetName.substr(_datasetName.length()-6,1);
            string data_sample = _datasetName.substr(5,_datasetName.length());
            size_t spos = data_sample.find("_");
            dataset_sample = data_sample.substr(0, spos).c_str();
        }else{
            dataset_era = "none";
            dataset_sample = "none";
            if( _datasetName.substr(0,6) == "Signal" ){
                dataset_group = "Signal";
            }else{
                dataset_group = "Bkg";
            }
        }

        dataset_dti = atoi(DatasetID.substr(6,1).c_str());
        _DatasetID = atoi(DatasetID.c_str());

        cout << " " << endl;
        cout << "group: " << dataset_group << endl;
        cout << "year: " << dataset_year << endl;
        cout << "dti: " << dataset_dti << endl;
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

    //const std::string model_s = "C:/Downloads/resnet18-v1-7.tar/resnet18-v1-7/resnet18-v1-7.onnx";
    //std::basic_string<ORTCHAR_T> model = std::basic_string<ORTCHAR_T>(model_s.begin(), model_s.end());
    //std::cout << model.c_str();
    

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
    //if( _ANALYSIS == "OPENDATA" ) VerticalSysSizesOD( );
    if( _ANALYSIS != "GEN" ) VerticalSysSizes( );
    
    //======PRE-ROUTINES SETUP=====================================================================
    if( _ANALYSIS != "GEN") PreRoutines();

    //======SETUP SELECTION========================================================================
    SetupAna();

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
        
        
        //======RUN ROUTINES SETUP======================================================
        if( _ANALYSIS == "GEN" ){
            if( !GenRoutines() ) continue;
        }else{
            if( !RunRoutines() ) continue;
        }
        
        //======START EVENT WEIGHT=====================================================
        evtWeight = 1.;
        if( (_ANALYSIS != "GEN") && (dataset_group != "Data") ) evtWeight = genWeight;
        
        
        //======RUN REGION SETUP=========================================================
        bool Selected = AnaRegion();
        
        
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
            AnaSelection();
        }
              
              
        //======RUN SYSTEMATIC PRODUCTION=================================================
        //if( _ANALYSIS == "OPENDATA" ) VerticalSysOD();
        if( _ANALYSIS != "GEN" ) VerticalSys();
        
        AnaSystematic();

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
    FinishAna();

    
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









