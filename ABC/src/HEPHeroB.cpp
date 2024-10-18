#include "HEPHero.h"


//---------------------------------------------------------------------------------------------------------------
// FILL CONTROL VARIABLES WITH INPUT FILE LINES
//---------------------------------------------------------------------------------------------------------------
void HEPHero::FillControlVariables( string key, string value){

    //----CORRECTIONS------------------------------------------------------------------------------------


    //----METADATA FILES---------------------------------------------------------------------------------


    //----SELECTION--------------------------------------------------------------------------------------


}


//---------------------------------------------------------------------------------------------------------------
// Init
//---------------------------------------------------------------------------------------------------------------
bool HEPHero::Init() {
    
    //======SET HISTOGRAMS STYLE===================================================================
    setStyle(1.0,true,0.15);

    if( _ANALYSIS != "GEN" ){
        //======SET THE BRANCH ADDRESSES===============================================================
        //_inputTree->SetBranchAddress("run", &run );
        //_inputTree->SetBranchAddress("luminosityBlock", &luminosityBlock );
        //_inputTree->SetBranchAddress("event", &event );
        

        
        //-----------------------------------------------------------------------------------------------------------------------
        if( dataset_group != "Data" ) {
            //_inputTree->SetBranchAddress("genWeight", &genWeight );
            

        }

    }

    return true;
}


//---------------------------------------------------------------------------------------------------------------
// Weight corrections
//---------------------------------------------------------------------------------------------------------------
void HEPHero::Weight_corrections(){

    //pileup_wgt = 1.;

    if(dataset_group != "Data"){

        /*
        if( apply_pileup_wgt ){
            pileup_wgt = GetPileupWeight(Pileup_nTrueInt, "nominal");
            evtWeight *= pileup_wgt;
        }
        */
    }
}


//---------------------------------------------------------------------------------------------------------------
// Get size of vertical systematic weights
// Keep the same order used in runSelection.py
//---------------------------------------------------------------------------------------------------------------
void HEPHero::VerticalSysSizes( ){
    if( (_sysID_lateral == 0) && (dataset_group != "Data") ) {
        sys_vertical_size.clear();
        _inputTree->GetEntry(0);

        //get_Pileup_sfs = false;

        for( int ivert = 0; ivert < _sysNames_vertical.size(); ++ivert ){
            string sysName = _sysNames_vertical.at(ivert);

            /*
            if( sysName == "Pileup" ){
                sys_vertical_size.push_back(2);
                get_Pileup_sfs = true;
            }
            */
        }
    }
}


//---------------------------------------------------------------------------------------------------------------
// Vertical systematics
// Keep the same order used in runSelection.py
//---------------------------------------------------------------------------------------------------------------
void HEPHero::VerticalSys(){
    if( (_sysID_lateral == 0) && (dataset_group != "Data") ) {
        sys_vertical_sfs.clear();

        //-----------------------------------------------------------------------------------
        /*
        if( get_Pileup_sfs ){
            vector<float> Pileup_sfs;
            double pileup_wgt_down = GetPileupWeight(Pileup_nTrueInt, "down");
            double pileup_wgt_up = GetPileupWeight(Pileup_nTrueInt, "up");
            Pileup_sfs.push_back(pileup_wgt_down/pileup_wgt);
            Pileup_sfs.push_back(pileup_wgt_up/pileup_wgt);
            sys_vertical_sfs.insert(pair<string, vector<float>>("Pileup", Pileup_sfs));
        }
        */
    }
}



//---------------------------------------------------------------------------------------------------------------
// MCsamples processing
//---------------------------------------------------------------------------------------------------------------
bool HEPHero::MC_processing(){

    bool pass_cut = true;
    string dsName = _datasetName.substr(0,_datasetName.length()-5);



    return pass_cut;
}


//---------------------------------------------------------------------------------------------------------------
// ANAFILES' ROUTINES
//---------------------------------------------------------------------------------------------------------------
void HEPHero::SetupAna(){
    if( false );
    else if( _SELECTION == "Test" ) SetupTest();
    // SETUP YOUR SELECTION HERE
    else {
      cout << "Unknown selection requested. Exiting. " << endl;
      return;
    }
}

bool HEPHero::AnaRegion(){
    bool Selected = true;
    if( _SELECTION == "Test" && !TestRegion() ) Selected = false;
    // SET THE REGION OF YOUR SELECTION HERE

    return Selected;
}

void HEPHero::AnaSelection(){
    if( _SELECTION == "Test" ) TestSelection();
    // CALL YOUR SELECTION HERE
}

void HEPHero::AnaSystematic(){
    if( _SELECTION == "Test" ) TestSystematic();
    // PRODUCE THE SYSTEMATIC OF YOUR SELECTION HERE
}

void HEPHero::FinishAna(){
    if( _SELECTION == "Test" ) FinishTest();
    // FINISH YOUR SELECTION HERE
}
   




