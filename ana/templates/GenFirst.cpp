#include "HEPHero.h"

//-------------------------------------------------------------------------------------------------
// Description:
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// Define output variables
//-------------------------------------------------------------------------------------------------
namespace GenFirst{

    float wgt_DefHighAll;
    float wgt_DefHighHard; 
    float wgt_DefHighSecondary;
    float wgt_DefLowAll; 
    float wgt_DefLowHard;
    float wgt_DefLowSecondary;
    
    float boson_pt;
    float boson_m;
    float boson_gen_m;
    float HT;
    float RN_FD_wgt;
    float RN_FU_wgt;
}


//-------------------------------------------------------------------------------------------------
// Define output derivatives
//-------------------------------------------------------------------------------------------------
void HEPHero::SetupGenFirst() {

    //======SETUP CUTFLOW==========================================================================
    _cutFlow.insert(pair<string,double>("good_events", 0) );

    //======SETUP HISTOGRAMS=======================================================================
    //makeHist( "histogram1DName", 40, 0., 40., "xlabel", "ylabel" );   [example]
    //makeHist( "histogram2DName", 40, 0., 40., 100, 0., 50., "xlabel",  "ylabel", "zlabel", "COLZ" );   [example]

    //======SETUP SYSTEMATIC HISTOGRAMS============================================================
    RegionID = 0;
    sys_regions = { 0, 1, 2 };  // Define regions, same as in RegionID. Empty vector means that all events will be used.
    makeSysHist( "Z_mass", 200, 0., 200., "Z mass [GeV]", "Events" ); 

    //======SETUP OUTPUT BRANCHES==================================================================
    //_outputTree->Branch("variable1NameInTheTree", &GenFirst::variable1Name );  [example]

    //======SETUP INFORMATION IN OUTPUT HDF5 FILE==================================================
    HDF_insert("HT", &GEN_HT );
    HDF_insert("MHT_pt", &GEN_MHT_pt );
    HDF_insert("MHT_phi", &GEN_MHT_phi );
    HDF_insert("MET_pt", &GEN_MET_pt );
    HDF_insert("MET_phi", &GEN_MET_phi );
    
    HDF_insert("event_number", &event_number );
    HDF_insert("signal_process_id", &signal_process_id );
    HDF_insert("alphaQCD", &alpha_QCD );
    HDF_insert("alphaQED", &alpha_QED );
    HDF_insert("event_scale", &event_scale );
    HDF_insert("N_mpi", &N_mpi );

    HDF_insert("id1", &id1 );
    HDF_insert("id2", &id2 );
    HDF_insert("x1", &x1 );
    HDF_insert("x2", &x2 );
    HDF_insert("scalePDF", &scalePDF );
    HDF_insert("pdf1", &pdf1 );
    HDF_insert("pdf2", &pdf2 );
    
    HDF_insert("weights_value", &weights_value );
    HDF_insert("parameters_id", &parameters_id );
    //HDF_insert("wgt_DefHighAll", &PSweights::wgt_DefHighAll );
    //HDF_insert("wgt_DefHighHard", &PSweights::wgt_DefHighHard ); 
    //HDF_insert("wgt_DefHighSecondary", &PSweights::wgt_DefHighSecondary );
    //HDF_insert("wgt_DefLowAll", &PSweights::wgt_DefLowAll ); 
    //HDF_insert("wgt_DefLowHard", &PSweights::wgt_DefLowHard );
    //HDF_insert("wgt_DefLowSecondary", &PSweights::wgt_DefLowSecondary );
    
    HDF_insert("boson_uncorr_pt", &GenFirst::boson_pt );
    HDF_insert("boson_uncorr_m", &GenFirst::boson_m );
    HDF_insert("boson_gen_m", &GenFirst::boson_gen_m );
    
    HDF_insert("Test_HT", &GenFirst::HT );
    
    HDF_insert("RN_FD_wgt", &GenFirst::RN_FD_wgt );
    HDF_insert("RN_FU_wgt", &GenFirst::RN_FU_wgt );

    return;
}


//-------------------------------------------------------------------------------------------------
// Define the selection region
//-------------------------------------------------------------------------------------------------
bool HEPHero::GenFirstRegion() {

    //-------------------------------------------------------------------------
    // Cut description
    //-------------------------------------------------------------------------
    //if( !(CutCondition) ) return false;           [Example]
    //_cutFlow.at("CutName") += evtWeight;          [Example]

    return true;
}


//-------------------------------------------------------------------------------------------------
// Write your analysis code here
//-------------------------------------------------------------------------------------------------
void HEPHero::GenFirstSelection() {


    event_number = _evt.event_number();
    
    event_scale = _evt.attribute<HepMC3::DoubleAttribute>("event_scale")->value();   
    if( _evt.attribute<HepMC3::DoubleAttribute>("AlphaQCD") ){
        alpha_QCD = _evt.attribute<HepMC3::DoubleAttribute>("AlphaQCD")->value();
    }
    if( _evt.attribute<HepMC3::DoubleAttribute>("AlphaEM") ){
        alpha_QED = _evt.attribute<HepMC3::DoubleAttribute>("AlphaEM")->value();
    }
    if( _evt.attribute<HepMC3::DoubleAttribute>("signal_process_id") ){
        signal_process_id = _evt.attribute<HepMC3::IntAttribute>("signal_process_id")->value();
    }
    if( _evt.attribute<HepMC3::DoubleAttribute>("mpi") ){
        N_mpi = _evt.attribute<HepMC3::IntAttribute>("mpi")->value();
    }
    
    
    std::shared_ptr<HepMC3::GenPdfInfo> pdf = _evt.attribute<HepMC3::GenPdfInfo>("GenPdfInfo");
    scalePDF = pdf->scale;
    id1 = pdf->parton_id[0];
    id2 = pdf->parton_id[1];
    x1 = pdf->x[0];
    x2 = pdf->x[1];
    pdf1 = pdf->xf[0];
    pdf2 = pdf->xf[1];
    /*
    vertices:
    • weights – vector of floating point numbers which correspond to the weights assigned to this vertex.

    particles:
    • flows – vector of integer numbers which correspond to the QCD color flow information. No encoding scheme of the colour flows is imposed by the library, but it is expected to comply with the rules in Ref. [2].
    • theta – an attribute holding the floating point value of the θ angle for polarisation.
    */
    
    calculate_gen_variables();


    
    GenFirst::RN_FD_wgt = weights_value[1];
    GenFirst::RN_FU_wgt = weights_value[2];
    
    
    GenFirst::boson_pt = -1;
    GenFirst::boson_m = -1;
    bool is_good = false;
    for (auto p: _evt.particles()) {
        if( (*p).end_vertex() && ((abs((*p).pdg_id()) == 22) || (abs((*p).pdg_id()) == 23)) ){
            for (auto daughter : (*p).end_vertex()->particles_out() ) { 
                if( (abs((*daughter).pdg_id()) == 11) || (abs((*daughter).pdg_id()) == 13) || (abs((*daughter).pdg_id()) == 15) ){
                    _cutFlow.at("good_events") += 1; 
                    is_good = true;
                    GenFirst::boson_pt = (*p).momentum().perp();
                    GenFirst::boson_m = (*p).momentum().m();
                    GenFirst::boson_gen_m = (*p).generated_mass();
                    break;
                }
            }
            if( is_good ) break;
        }
        
        //cout << "flow: " << (*p).flow() << endl;
        //cout << "polarization: " << (*p).polarization() << endl;
    }
    
    /*
    vector<ConstGenParticlePtr> getDescendants( ConstGenParticlePtr parent ) {
        Filter f = ( StandardSelector :: STATUS == 1 &&
                    StandardSelector :: PT > 0.1 &&
                    StandardSelector :: ETA > -2.5 &&
                    StandardSelector :: ETA < 2.5);
        return applyFilter(f, Relatives::DESCENDANTS( parent ));
    }
    */
    
    GenFirst::HT = 0;
    for (auto p: _evt.particles()) {
        if( !(*p).end_vertex() || ((*p).status() == 1) ){
            bool jet_component = false;
            for (auto ancestor : HepMC3::Relatives::ANCESTORS(p) ) {
                if( (*ancestor).pdg_id() == 81 ){
                    jet_component = true;
                    break;
                }
            }
            if( jet_component ){ 
                GenFirst::HT += (*p).momentum().perp();
            }else{
                //cout << "particle ID = " << (*p).pdg_id() << endl;
            }
        }
    }


    plot_events({1, 5});


    //======ASSIGN VALUES TO THE OUTPUT VARIABLES==================================================
    //GenFirst::variable1Name = 100;      [Example]

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
void HEPHero::GenFirstSystematic() {

    FillSystematic( "Z_mass", GenFirst::boson_m, evtWeight );
}


//-------------------------------------------------------------------------------------------------
// Make efficiency plots
//-------------------------------------------------------------------------------------------------
void HEPHero::FinishGenFirst() {

    //MakeEfficiencyPlot( _histograms1D.at("Matched_pt"), _histograms1D.at("all_pt"), "Match_pt" );   [example]

    return;
}
