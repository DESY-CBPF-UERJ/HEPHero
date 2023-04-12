#include "HEPHero.h"

//-------------------------------------------------------------------------------------------------
// Description:
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// Define output variables
//-------------------------------------------------------------------------------------------------
namespace Test{

    //int Leading_pfIsoId;
    //int Trailing_pfIsoId;
}


//-------------------------------------------------------------------------------------------------
// Define output derivatives
//-------------------------------------------------------------------------------------------------
void HEPHero::SetupTest() {

    //======SETUP CUTFLOW==========================================================================
    _cutFlow.insert(pair<string,double>("00_TwoLepOS", 0) );
    _cutFlow.insert(pair<string,double>("01_MET", 0) );
    _cutFlow.insert(pair<string,double>("02_LeadingLep_Pt", 0) );
    _cutFlow.insert(pair<string,double>("03_LepLep_DM", 0) );
    _cutFlow.insert(pair<string,double>("04_LepLep_Pt", 0) );
    _cutFlow.insert(pair<string,double>("05_LepLep_DR", 0) );
    _cutFlow.insert(pair<string,double>("06_MET_LepLep_DPhi", 0) );
    _cutFlow.insert(pair<string,double>("07_MET_LepLep_Mt", 0) );
    _cutFlow.insert(pair<string,double>("08_Selected", 0) );
    _cutFlow.insert(pair<string,double>("09_Corrected", 0) );

    //======SETUP HISTOGRAMS=======================================================================
    //makeHist( "histogram1DName", 40, 0., 40., "xlabel", "ylabel" );   [example]
    //makeHist( "histogram2DName", 40, 0., 40., 100, 0., 50., "xlabel",  "ylabel", "zlabel", "COLZ" );   [example]

    //======SETUP SYSTEMATIC HISTOGRAMS============================================================
    sys_regions = { 0, 1, 2, 3, 4 }; // Use the same IDs produced by the Regions() function. Empty vector means that all events will be used.
    makeSysHist( "N_PV", 50, 0., 50., "Number of PVs", "Number of events"); 
    makeSysHist( "MET_pt", 2500, 0., 2500., "MET [GeV]", "Number of events");
    makeSysHist( "N_Jets", 50, 0., 50., "Number of jets", "Number of events");
    makeSysHist( "Nbjets", 50, 0., 50., "Number of b-jets", "Number of events"); 
    makeSysHist( "LeadingLep_pt", 2500, 0., 2500., "Leading Lep Pt [GeV]", "Number of events");
    makeSysHist( "TrailingLep_pt", 2500, 0., 2500., "Trailing Lep Pt [GeV]", "Number of events");
    makeSysHist( "LepLep_pt", 2500, 0., 2500., "LepLep Pt [GeV]", "Number of events");
    makeSysHist( "LepLep_deltaR", 320, 0., 3.2, "LepLep deltaR", "Number of events");
    makeSysHist( "MLP_score_1000_100", 1000, 0., 1., "MLP_score_1000_100", "Number of events");
    makeSysHist( "MLP_score_400_200", 1000, 0., 1., "MLP_score_400_200", "Number of events");
    
    makeSysHist( "MLP_score", 100, 0., 1., 100, 0., 1., "Signal boosted score", "Signal not boosted score", "Number of events");

    //======SETUP OUTPUT BRANCHES==================================================================
    HDF_insert( "RecoLepID", &RecoLepID );
    HDF_insert( "RegionID", &RegionID );
    HDF_insert( "DatasetID", &_DatasetID );
    HDF_insert( "LepLep_pt", &LepLep_pt );
    HDF_insert( "LepLep_deltaR", &LepLep_deltaR );
    HDF_insert( "LeadingLep_pt", &LeadingLep_pt );
    HDF_insert( "TrailingLep_pt", &TrailingLep_pt );
    HDF_insert( "LeadingLep_eta", &LeadingLep_eta );
    HDF_insert( "TrailingLep_eta", &TrailingLep_eta );
    HDF_insert( "MET_pt", &MET_pt );
    HDF_insert( "LepLep_deltaM", &LepLep_deltaM );
    HDF_insert( "MET_LepLep_deltaPhi", &MET_LepLep_deltaPhi );
    HDF_insert( "MET_LepLep_Mt", &MET_LepLep_Mt );
    HDF_insert( "Nbjets", &Nbjets );
    HDF_insert( "Njets", &Njets );
    HDF_insert( "Njets30", &Njets30 );
    HDF_insert( "Njets40", &Njets40 );
    HDF_insert( "HT30", &HT30 );
    HDF_insert( "MHT30", &MHT30 );
    HDF_insert( "OmegaMin", &OmegaMin );
    HDF_insert( "ChiMin", &ChiMin );
    HDF_insert( "FMax", &FMax );
    HDF_insert( "MT2LL", &MT2LL );
    HDF_insert( "Njets_forward", &Njets_forward );
    HDF_insert( "Jet_abseta_max", &Jet_abseta_max );
    HDF_insert( "Dijet_deltaEta", &Dijet_deltaEta );
    HDF_insert( "Dijet_M", &Dijet_M );
    HDF_insert( "Dijet_pt", &Dijet_pt );
    HDF_insert( "ttbar_reco", &ttbar_reco );
    HDF_insert( "PV_npvs", &PV_npvs );
    HDF_insert( "PV_npvsGood", &PV_npvsGood );
    HDF_insert( "pileup_wgt", &pileup_wgt );
    HDF_insert( "electron_wgt", &electron_wgt );
    HDF_insert( "muon_wgt", &muon_wgt );
    HDF_insert( "btag_wgt", &btag_wgt );
    HDF_insert( "trigger_wgt", &trigger_wgt );
    HDF_insert( "prefiring_wgt", &prefiring_wgt );
    HDF_insert( "top_pt_wgt", &top_pt_wgt );
    HDF_insert( "jet_puid_wgt", &jet_puid_wgt );
    HDF_insert( "MET_RAW_pt", &MET_RAW_pt );
    HDF_insert( "MET_RAW_phi", &MET_RAW_phi );
    HDF_insert( "MET_Unc_pt", &MET_Unc_pt );
    HDF_insert( "MET_Unc_phi", &MET_Unc_phi );
    HDF_insert( "MET_JES_pt", &MET_JES_pt );
    HDF_insert( "MET_JES_phi", &MET_JES_phi );
    HDF_insert( "MET_XY_pt", &MET_XY_pt );
    HDF_insert( "MET_XY_phi", &MET_XY_phi );
    HDF_insert( "MET_JER_pt", &MET_JER_pt );
    HDF_insert( "MET_JER_phi", &MET_JER_phi );
    HDF_insert( "MET_Emu_pt", &MET_Emu_pt );
    HDF_insert( "MET_Emu_phi", &MET_Emu_phi );
    HDF_insert( "MLP_score_1000_100", &MLP_score_1000_100 );
    HDF_insert( "MLP_score_400_200", &MLP_score_400_200 );
    //HDF_insert( "param_variation_weights", &Test::param_variation_weights );
    HDF_insert( "VVCR_LeadingLep_pt", &VVCR_LeadingLep_pt );
    HDF_insert( "MET_significance", &MET_significance );
        
    return;
}


//-------------------------------------------------------------------------------------------------
// Define the selection region
//-------------------------------------------------------------------------------------------------
bool HEPHero::TestRegion() {

    LeptonSelection();
    
    if( !(RecoLepID > 0) ) return false;                                        // Has two reconstructed leptons with opposite signal
    _cutFlow.at("00_TwoLepOS") += evtWeight;
    
    JetSelection();
    METCorrection();
    
    if( !(MET_pt > MET_CUT) ) return false;                                     // MET > CUT
    _cutFlow.at("01_MET") += evtWeight;  
    
    Get_Leptonic_Info(true, true);
    
    if( !(LeadingLep_pt > LEADING_LEP_PT_CUT) ) return false;                   // Leading lepton pt > CUT
    _cutFlow.at("02_LeadingLep_Pt") += evtWeight; 
    
    Get_LepLep_Variables(true, true);
    
    if( !(LepLep_deltaM < LEPLEP_DM_CUT) ) return false;                        // Difference between Z boson mass and the inv. mass of two leptons < CUT
    _cutFlow.at("03_LepLep_DM") += evtWeight; 
    
    if( !(LepLep_pt > LEPLEP_PT_CUT) ) return false;                            // Two leptons system pt > CUT
    _cutFlow.at("04_LepLep_Pt") += evtWeight;
    
    if( !(LepLep_deltaR < LEPLEP_DR_CUT) ) return false;                        // Upper cut in LepLep Delta R 
    _cutFlow.at("05_LepLep_DR") += evtWeight; 
    
    if( !(MET_LepLep_deltaPhi > MET_LEPLEP_DPHI_CUT) ) return false;            // Dealta Phi between MET and two leptons system > CUT
    _cutFlow.at("06_MET_LepLep_DPhi") += evtWeight;  
    
    if( !(MET_LepLep_Mt > MET_LEPLEP_MT_CUT) ) return false;                    // Transverse mass between MET and two leptons system > CUT
    _cutFlow.at("07_MET_LepLep_Mt") += evtWeight;
    
    Get_ttbar_Variables();
    
    Regions();
    if( !((RegionID >= 0) && (RegionID <= 4)) ) return false;                   // 0=SR, 1=DY-CR, 2=ttbar-CR, 3=WZ-CR, 4=ZZ-CR
     
    bool GoodEvent = lumi_certificate.GoodLumiSection( _datasetName, run, luminosityBlock );
    if( !GoodEvent ) return false;                                              // Select only certified data events
    
    if( !METFilters() ) return false;                                           // Selected by MET filters
    
    if( !Trigger() ) return false;                                              // Selected by triggers
    _cutFlow.at("08_Selected") += evtWeight;
    
    Weight_corrections();
    _cutFlow.at("09_Corrected") += evtWeight;
    
    Signal_discriminators();
    Get_Jet_Angular_Variables( );
    Get_Dijet_Variables();
    
    return true;
}


//-------------------------------------------------------------------------------------------------
// Write your analysis code here
//-------------------------------------------------------------------------------------------------
void HEPHero::TestSelection() {

    /*
    ML::param_variation_weights.clear();
    if( dataset_group == "Signal" ) {
        for( unsigned int iw = 0; iw < nLHEReweightingWeight; ++iw ) {
            ML::param_variation_weights.push_back( LHEReweightingWeight[iw] );
        }
    }
    */

    //======ASSIGN VALUES TO THE OUTPUT VARIABLES==================================================
    //Test::variable1Name = 100;      [Example]

    //======FILL THE HISTOGRAMS====================================================================
    //_histograms1D.at("histogram1DName").Fill( var, evtWeight );               [Example]
    //_histograms2D.at("histogram2DName").Fill( var1, var2, evtWeight );        [Example]

    //======FILL THE OUTPUT TREE===================================================================
    HDF_fill();

    return;
}


//-------------------------------------------------------------------------------------------------
// Produce systematic histograms
//-------------------------------------------------------------------------------------------------
void HEPHero::TestSystematic() {
    
    FillSystematic( "N_PV", PV_npvs, evtWeight ); 
    FillSystematic( "MET_pt", MET_pt, evtWeight );
    FillSystematic( "N_Jets", Njets30, evtWeight );    
    FillSystematic( "Nbjets", Nbjets, evtWeight ); 
    FillSystematic( "LeadingLep_pt", LeadingLep_pt, evtWeight );
    FillSystematic( "TrailingLep_pt", TrailingLep_pt, evtWeight );
    FillSystematic( "LepLep_pt", LepLep_pt, evtWeight );
    FillSystematic( "LepLep_deltaR", LepLep_deltaR, evtWeight );
    FillSystematic( "MLP_score_1000_100", MLP_score_1000_100, evtWeight );
    FillSystematic( "MLP_score_400_200", MLP_score_400_200, evtWeight );
    FillSystematic( "MLP_score", MLP_score_1000_100, MLP_score_400_200, evtWeight );
    
}


//-------------------------------------------------------------------------------------------------
// Make efficiency plots
//-------------------------------------------------------------------------------------------------
void HEPHero::FinishTest() {

    //MakeEfficiencyPlot( _histograms1D.at("Matched_pt"), _histograms1D.at("all_pt"), "Match_pt" );   [example]

    return;
}
