#include "HEPHero.h"

//-------------------------------------------------------------------------------------------------
// Description:
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// Define output variables
//-------------------------------------------------------------------------------------------------
namespace Cutflow{

    //int variable1Name;   [example]
}


//-------------------------------------------------------------------------------------------------
// Define output derivatives
//-------------------------------------------------------------------------------------------------
void HEPHero::SetupCutflow() {

    //======SETUP CUTFLOW==========================================================================
    _cutFlow.insert(pair<string,double>("00_TwoLepOS", 0) );
    _cutFlow.insert(pair<string,double>("01_NBJets", 0) );
    _cutFlow.insert(pair<string,double>("02_METFilters", 0) );
    _cutFlow.insert(pair<string,double>("03_METCut", 0) );
    _cutFlow.insert(pair<string,double>("04_LeadingLep_Pt", 0) );
    _cutFlow.insert(pair<string,double>("05_LepLep_DM", 0) );
    _cutFlow.insert(pair<string,double>("06_LepLep_Pt", 0) );
    _cutFlow.insert(pair<string,double>("07_LepLep_DR", 0) );
    _cutFlow.insert(pair<string,double>("08_MET_LepLep_DPhi", 0) );
    _cutFlow.insert(pair<string,double>("09_MET_LepLep_Mt", 0) );
    _cutFlow.insert(pair<string,double>("10_TTbar_Reco", 0) );
    _cutFlow.insert(pair<string,double>("11_Trigger", 0) );


    //======SETUP HISTOGRAMS=======================================================================
    //makeHist( "histogram1DName", 40, 0., 40., "xlabel", "ylabel" );   [example]
    //makeHist( "histogram2DName", 40, 0., 40., 100, 0., 50., "xlabel",  "ylabel", "zlabel", "COLZ" );   [example]

    //======SETUP SYSTEMATIC HISTOGRAMS============================================================
    //makeSysHist( "histogramSysName", 40, 0., 40., "xlabel", "ylabel" );   [example]

    //======SETUP OUTPUT BRANCHES==================================================================
    //_outputTree->Branch("variable1NameInTheTree", &Cutflow::variable1Name );  [example]

    return;
}


//-------------------------------------------------------------------------------------------------
// Define the selection region
//-------------------------------------------------------------------------------------------------
bool HEPHero::CutflowRegion() {
    
    LeptonSelection();
    if( !((RecoLepID == 11) || (RecoLepID == 13)) ) return false;               // Has two reconstructed leptons with opposite signal
    //Weight_corrections();
    _cutFlow.at("00_TwoLepOS") += evtWeight;
    
    JetSelection();
    if( !(Nbjets >= 1) ) return false;                                          // Has b-jets
    _cutFlow.at("01_NBJets") += evtWeight;
    
    if( !METFilters() ) return false;                                           // Selected by MET filters
    _cutFlow.at("02_METFilters") += evtWeight;
    
    METCorrection();
    if( !(MET_pt > MET_CUT) ) return false;                                     // MET > CUT
    _cutFlow.at("03_METCut") += evtWeight; 
    
    Get_Leptonic_Info(true, true);
    if( !(LeadingLep_pt > LEADING_LEP_PT_CUT) ) return false;                   // Leading lepton pt > CUT
    _cutFlow.at("04_LeadingLep_Pt") += evtWeight; 
    
    Get_LepLep_Variables(true, true);
    if( !(LepLep_deltaM < LEPLEP_DM_CUT) ) return false;                        // Difference between Z boson mass and the inv. mass of two leptons < CUT
    _cutFlow.at("05_LepLep_DM") += evtWeight;  
    
    if( !(LepLep_pt > LEPLEP_PT_CUT) ) return false;                            // Two leptons system pt > CUT
    _cutFlow.at("06_LepLep_Pt") += evtWeight;  
    
    if( !(LepLep_deltaR < LEPLEP_DR_CUT) ) return false;                        // Upper and lower cuts in LepLep Delta R 
    _cutFlow.at("07_LepLep_DR") += evtWeight;  
    
    if( !(MET_LepLep_deltaPhi > MET_LEPLEP_DPHI_CUT) ) return false;            // Dealta Phi between MET and two leptons system > CUT
    _cutFlow.at("08_MET_LepLep_DPhi") += evtWeight;  
    
    if( !(MET_LepLep_Mt > MET_LEPLEP_MT_CUT) ) return false;                    // Transverse mass between MET and two leptons system > CUT
    _cutFlow.at("09_MET_LepLep_Mt") += evtWeight;
    
    Get_ttbar_Variables();
    if( !(ttbar_reco == 0) ) return false;                                      // Events with successfully reconstructed ttbar particles: 0=fail, 1=success
    _cutFlow.at("10_TTbar_Reco") += evtWeight;
    
    if( !Trigger() ) return false;                                              // Selected by triggers
    _cutFlow.at("11_Trigger") += evtWeight;
    
    
    return true;
}


//-------------------------------------------------------------------------------------------------
// Write your analysis code here
//-------------------------------------------------------------------------------------------------
void HEPHero::CutflowSelection() {



    // Use only the inclusive sample for Drell-Yan!









    //======ASSIGN VALUES TO THE OUTPUT VARIABLES==================================================
    //Cutflow::variable1Name = 100;      [Example]

    //======FILL THE HISTOGRAMS====================================================================
    //_histograms1D.at("histogram1DName").Fill( var, evtWeight );               [Example]
    //_histograms2D.at("histogram2DName").Fill( var1, var2, evtWeight );        [Example]

    //======FILL THE OUTPUT TREE===================================================================
    //_outputTree->Fill();

    return;
}


//-------------------------------------------------------------------------------------------------
// Produce systematic histograms
//-------------------------------------------------------------------------------------------------
void HEPHero::CutflowSystematic() {

    //FillSystematic( "histogramSysName", var, evtWeight );  [Example]
}


//-------------------------------------------------------------------------------------------------
// Make efficiency plots
//-------------------------------------------------------------------------------------------------
void HEPHero::FinishCutflow() {

    //MakeEfficiencyPlot( _histograms1D.at("Matched_pt"), _histograms1D.at("all_pt"), "Match_pt" );   [example]

    return;
}
