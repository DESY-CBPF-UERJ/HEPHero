#include "HEPHero.h"

//-------------------------------------------------------------------------------------------------
// Description:
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// Define output variables
//-------------------------------------------------------------------------------------------------
namespace TestOD{

    //int variable1Name;   [example]
}


//-------------------------------------------------------------------------------------------------
// Define output derivatives
//-------------------------------------------------------------------------------------------------
void HEPHero::SetupTestOD() {

    //======SETUP CUTFLOW==========================================================================
    _cutFlow.insert(pair<string,double>("00_MinimalSelection", 0) );
    _cutFlow.insert(pair<string,double>("01_Trigger", 0) );
    _cutFlow.insert(pair<string,double>("02_FilterGoodEvents", 0) );
    _cutFlow.insert(pair<string,double>("03_HasMuonTauPair", 0) );
    _cutFlow.insert(pair<string,double>("04_QCDsuppression", 0) );
    _cutFlow.insert(pair<string,double>("05_MuonL_pt", 0) );
    _cutFlow.insert(pair<string,double>("06_MuonL_MET_Mt", 0) );
    _cutFlow.insert(pair<string,double>("07_TauH_MuonL_M", 0) );
    _cutFlow.insert(pair<string,double>("08_Selected", 0) );
    _cutFlow.insert(pair<string,double>("09_Corrected", 0) );


    //======SETUP HISTOGRAMS=======================================================================
    //makeHist( "histogram1DName", 40, 0., 40., "xlabel", "ylabel" );   [example]
    //makeHist( "histogram2DName", 40, 0., 40., 100, 0., 50., "xlabel",  "ylabel", "zlabel", "COLZ" );   [example]

    //======SETUP SYSTEMATIC HISTOGRAMS============================================================
    //sys_regions = { 0, 1, 2 }; [example] // Choose regions as defined in RegionID. Empty vector means that all events will be used.
    //makeSysHist( "histogram1DSysName", 40, 0., 40., "xlabel", "ylabel" );   [example]
    //makeSysHist( "histogram2DSysName", 40, 0., 40., 100, 0., 50., "xlabel",  "ylabel", "zlabel", "COLZ" );   [example]

    //======SETUP OUTPUT BRANCHES==================================================================
    //_outputTree->Branch("variable1NameInTheTree", &TestOD::variable1Name );  [example]

    //======SETUP INFORMATION IN OUTPUT HDF5 FILE==================================================
    HDF_insert( "RegionID", &RegionID );

    HDF_insert( "MET_pt", &MET_pt );
    HDF_insert( "MET_phi", &MET_phi );
    HDF_insert( "PV_npvs", &PV_npvs );
    HDF_insert( "Nmuons", &Nmuons );
    HDF_insert( "Ntaus", &Ntaus);
    HDF_insert( "Nbjets", &Nbjets );
    HDF_insert( "Njets", &Njets );
    //HDF_insert( "Njets30", &Njets30 );
    //HDF_insert( "Njets_forward", &Njets_forward );
    //HDF_insert( "Njets30_forward", &Njets30_forward );
    HDF_insert( "HT", &HT );
    HDF_insert( "HT30", &HT30 );
    HDF_insert( "MHT", &MHT );
    HDF_insert( "MHT30", &MHT30 );
    HDF_insert( "LeadingJet_pt", &LeadingJet_pt );
    //HDF_insert( "SubLeadingJet_pt", &SubLeadingJet_pt );
    HDF_insert( "OmegaMin30", &OmegaMin30);
    HDF_insert( "ChiMin30", &ChiMin30 );
    HDF_insert( "FMax30", &FMax30 );

    HDF_insert( "TauH_pt", &TauH_pt );
    HDF_insert( "MuonL_pt", &MuonL_pt );
    HDF_insert( "MuonL_MET_pt", &MuonL_MET_pt );
    HDF_insert( "MuonL_MET_dphi", &MuonL_MET_dphi );
    HDF_insert( "MuonL_MET_Mt", &MuonL_MET_Mt );
    HDF_insert( "TauH_TauL_pt", &TauH_TauL_pt );
    HDF_insert( "TauH_TauL_dphi", &TauH_TauL_dphi );
    HDF_insert( "TauH_TauL_Mt", &TauH_TauL_Mt );
    HDF_insert( "TauH_MuonL_pt", &TauH_MuonL_pt );
    HDF_insert( "TauH_MuonL_M", &TauH_MuonL_M );

    HDF_insert( "LepLep_deltaM", &LepLep_deltaM );
    HDF_insert( "Has_2OC_muons", &Has_2OC_muons );

    HDF_insert( "MLP_score_torch", &MLP_score_torch );



    return;
}


//-------------------------------------------------------------------------------------------------
// Define the selection region
//-------------------------------------------------------------------------------------------------
bool HEPHero::TestODRegion() {

    //-------------------------------------------------------------------------
    // Minimal Selection
    //-------------------------------------------------------------------------
    if( !((nMuon > 0) && (nTau > 0)) ) return false;
    _cutFlow.at("00_MinimalSelection") += evtWeight;

    //-------------------------------------------------------------------------
    // Trigger
    //-------------------------------------------------------------------------
    if( !((HLT_IsoMu17_eta2p1_LooseIsoPFTau20 == true) ))// || (HLT_IsoMu24_eta2p1 == true) || (HLT_IsoMu24 == true)) ) return false;
    _cutFlow.at("01_Trigger") += evtWeight;

    LeptonSelectionOD();

    //-------------------------------------------------------------------------
    // Filter Good Events
    //-------------------------------------------------------------------------
    if( !(((selectedMu.size() == 1) || (selectedMu.size() == 2)) && (selectedTau.size() >= 1)) ) return false;
    _cutFlow.at("02_FilterGoodEvents") += evtWeight;

    bool Has_MuonTau_pair = MuonTauPairSelectionOD();

    //-------------------------------------------------------------------------
    // Has a good MuonTau pair
    //-------------------------------------------------------------------------
    if( !Has_MuonTau_pair ) return false;
    _cutFlow.at("03_HasMuonTauPair") += evtWeight;

    JetSelectionOD();
    Get_Jet_Angular_Variables( 30 );

    //-------------------------------------------------------------------------
    // QCD suppression
    //-------------------------------------------------------------------------
    //if( !(OmegaMin30 > 0.3) ) return false;
    _cutFlow.at("04_QCDsuppression") += evtWeight;

    METCorrectionOD();
    Jet_TauTau_VariablesOD();

    //-------------------------------------------------------------------------
    // MuonL_pt
    //-------------------------------------------------------------------------
    //if( !(MuonL_pt < 170) ) return false;
    _cutFlow.at("05_MuonL_pt") += evtWeight;

    //-------------------------------------------------------------------------
    // MuonL_MET_Mt
    //-------------------------------------------------------------------------
    //if( !(MuonL_MET_Mt < 100) ) return false;
    _cutFlow.at("06_MuonL_MET_Mt") += evtWeight;

    //-------------------------------------------------------------------------
    // TauH_MuonL_M
    //-------------------------------------------------------------------------
    //if( !((TauH_MuonL_M > 80) && (TauH_MuonL_M < 260)) ) return false;
    _cutFlow.at("07_TauH_MuonL_M") += evtWeight;


    //RegionsOD();
    //if( !((RegionID >= 0) && (RegionID <= 3)) ) return false;     // 0=SR, 1=DY-CR, 2=ttbar-CR, 3=Wjets-CR
    _cutFlow.at("08_Selected") += evtWeight;

    Weight_correctionsOD();
    _cutFlow.at("09_Corrected") += evtWeight;

    //cout << _EventPosition << " " << endl;

    /*
    MLP_score_torch = 0;
    if( (RegionID == 0) && (MET_pt > 5) && (MET_pt < 150) && (MuonL_MET_dphi > 0) && (MuonL_MET_dphi < 2.5) && (MuonL_MET_pt > 30) && (MuonL_MET_pt < 250) && (MuonL_pt > 45) && (MuonL_pt < 120) && (TauH_MuonL_M > 80) && (TauH_MuonL_M < 240) && (TauH_pt > 45) && (TauH_pt < 120) && (TauH_TauL_dphi > 0) && (TauH_TauL_dphi < 3.13) && (TauH_TauL_Mt > 10) && (TauH_TauL_Mt < 300) && (TauH_TauL_pt > 10) && (TauH_TauL_pt < 200) ){
        cout << "MET_pt " << MET_pt << endl;
        cout << "MuonL_MET_dphi " << MuonL_MET_dphi << endl;
        cout << "MuonL_MET_pt " << MuonL_MET_pt << endl;
        cout << "MuonL_pt " << MuonL_pt << endl;
        cout << "TauH_MuonL_M " << TauH_MuonL_M << endl;
        cout << "TauH_pt " << TauH_pt << endl;
        cout << "TauH_TauL_dphi " << TauH_TauL_dphi << endl;
        cout << "TauH_TauL_Mt " << TauH_TauL_Mt << endl;
        cout << "TauH_TauL_pt " << TauH_TauL_pt << endl;
        Signal_discriminatorsOD();
    }
    */

    return true;
}


//-------------------------------------------------------------------------------------------------
// Write your analysis code here
//-------------------------------------------------------------------------------------------------
void HEPHero::TestODSelection() {













    //======ASSIGN VALUES TO THE OUTPUT VARIABLES==================================================
    //TestOD::variable1Name = 100;      [Example]

    //======FILL THE HISTOGRAMS====================================================================
    //_histograms1D.at("histogram1DName").Fill( var, evtWeight );               [Example]
    //_histograms2D.at("histogram2DName").Fill( var1, var2, evtWeight );        [Example]

    //======FILL THE OUTPUT TREE===================================================================
    //_outputTree->Fill();

    //======FILL THE OUTPUT HDF5 INFO===============================================================
    HDF_fill();

    return;
}


//-------------------------------------------------------------------------------------------------
// Produce systematic histograms
//-------------------------------------------------------------------------------------------------
void HEPHero::TestODSystematic() {

    //FillSystematic( "histogram1DSysName", var, evtWeight );  [Example]
    //FillSystematic( "histogram2DSysName", var1, var2, evtWeight );  [Example]
}


//-------------------------------------------------------------------------------------------------
// Make efficiency plots
//-------------------------------------------------------------------------------------------------
void HEPHero::FinishTestOD() {

    //MakeEfficiencyPlot( _histograms1D.at("Matched_pt"), _histograms1D.at("all_pt"), "Match_pt" );   [example]

    return;
}
