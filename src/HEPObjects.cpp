#include "HEPHero.h"

//---------------------------------------------------------------------------------------------------------------
// Pre-Objects Setup
//--------------------------------------------------------------------------------------------------------------- 
void HEPHero::PreObjects() {
    
    if( _ANALYSIS != "GEN" ){
        //=================================================================================================================
        // CMS SETUP
        //=================================================================================================================
        lumi_certificate.ReadFile(certificate_file);
        PDFtype();
        GetMCMetadata();
        
        //----OUTPUT INFO------------------------------------------------------------------------------
        _outputTree->Branch( "evtWeight", &evtWeight );
        HDF_insert( "evtWeight", &evtWeight );
        
        size_t pos_HT = _datasetName.find("_HT-");
        size_t pos_Pt = _datasetName.find("_Pt-");
        size_t pos_PtZ = _datasetName.find("_PtZ-");
        size_t pos_NJets = _datasetName.find("_NJets-");
        if( (pos_HT != string::npos) || (pos_Pt != string::npos) || (pos_PtZ != string::npos) || (pos_NJets != string::npos) ){
            _outputTree->Branch( "LHE_HT", &LHE_HT );
            _outputTree->Branch( "LHE_Vpt", &LHE_Vpt );
            //_outputTree->Branch( "LHE_Njets", &LHE_Njets );
            HDF_insert( "LHE_HT", &LHE_HT );
            HDF_insert( "LHE_Vpt", &LHE_Vpt );
            //HDF_insert( "LHE_Njets", &LHE_Njets );
        }
        
        
        /*
        TFile* vjets_file = new TFile("Metadata/QCD_NLO_k_factors_VJets/UL/scale_factors.root","READ");
        
        TH2D* vjets_h2d = (TH2D*) vjets_file->Get("dy_sf_ul");
        
        float mjj = 500.;
        float Vpt = 230.;
        
        int mjj_i = -1;
        int Vpt_j = -1;
        
        float imjj[9] = {200, 400, 600, 900, 1200, 1500, 2000, 2700, 3500};
        for( unsigned int i = 0; i < 8; ++i ) {
            if( (mjj >= imjj[i]) && (mjj < imjj[i+1]) ){ 
                mjj_i = i+1;
                break;
            }
        }
        
        float jVpt[10] = {200, 240, 260, 280, 300, 340, 400, 500, 740, 1000};
        for( unsigned int j = 0; j < 9; ++j ) {
            if( (Vpt >= jVpt[j]) && (Vpt < jVpt[j+1]) ){ 
                Vpt_j = j+1;
                break;
            }
        }
        
        if( (mjj_i > 0) && (Vpt_j > 0) ){
            float value_s = vjets_h2d->GetBinContent(mjj_i,Vpt_j) ;
            float err_s = vjets_h2d->GetBinError(mjj_i,Vpt_j);
            cout << "value_s " << value_s << " " << endl;
            cout << "err_s " << err_s << " " << endl; 
        }
        */
        
        

        //----JES--------------------------------------------------------------------------------------
        //if( dataset_group != "Data" ) JES_unc.ReadFile( JES_MC_file );
        
        //----JER--------------------------------------------------------------------------------------
        //if( apply_jer_corr ){
        //    if( dataset_group != "Data" ) jer_corr.ReadFiles(JER_file, JER_SF_file);
        //}
        
        //----ELECTRON ID------------------------------------------------------------------------------
        if( apply_electron_wgt ){
            auto electron_set = correction::CorrectionSet::from_file(electron_file.c_str());
            electron_ID_corr = electron_set->at("UL-Electron-ID-SF");
        }
        
        //----MUON ID----------------------------------------------------------------------------------
        if( apply_muon_wgt ){
            auto muon_set = correction::CorrectionSet::from_file(muon_file.c_str());
            
            string MuID_WP;
            if( MUON_ID_WP == 0 ){ 
                MuID_WP = "NUM_LooseID_DEN_TrackerMuons";
            }else if( MUON_ID_WP == 1 ){
                MuID_WP = "NUM_MediumID_DEN_TrackerMuons";
            }else if( MUON_ID_WP == 2 ){
                MuID_WP = "NUM_TightID_DEN_TrackerMuons";
            }
            muon_ID_corr = muon_set->at(MuID_WP);
            
            string MuISO_WP;
            if( MUON_ISO_WP == 0 ){ 
                MuISO_WP = "NUM_LooseRelIso_DEN_LooseID"; // dumb value, not used
            }else if( MUON_ISO_WP == 1 ){ 
                MuISO_WP = "NUM_LooseRelIso_DEN_LooseID";
            }else if( MUON_ISO_WP == 2 ){
                MuISO_WP = "NUM_LooseRelIso_DEN_MediumID";
            }else if( MUON_ISO_WP == 3 ){
                MuISO_WP = "NUM_TightRelIso_DEN_MediumID";
            }
            muon_ISO_corr = muon_set->at(MuISO_WP);
        }    

        //----JET PU ID--------------------------------------------------------------------------------
        if( apply_jet_puid_wgt ){
            auto jet_puid_set = correction::CorrectionSet::from_file(jet_puid_file.c_str());        
            jet_PUID_corr = jet_puid_set->at("PUJetID_eff");
        }
        
        //----JERC-------------------------------------------------------------------------------------
        auto jet_jerc_set = correction::CorrectionSet::from_file(jet_jerc_file.c_str());
        
        string jer_SF_corr_name;
        string jer_PtRes_corr_name;
        string jes_Unc_name;
        
        if( dataset_year == "16" ){
            if( dataset_HIPM ){
                jer_SF_corr_name = "Summer20UL16APV_JRV3_MC_ScaleFactor_AK4PFchs";
                jer_PtRes_corr_name = "Summer20UL16APV_JRV3_MC_PtResolution_AK4PFchs";
                jes_Unc_name = "Summer19UL16APV_V7_MC_Total_AK4PFchs";
                if( _sysName_lateral == "JES" ) jes_Unc_name = "Summer19UL16APV_V7_MC_"+_SysSubSource+"_AK4PFchs";
            }else{
                jer_SF_corr_name = "Summer20UL16_JRV3_MC_ScaleFactor_AK4PFchs";
                jer_PtRes_corr_name = "Summer20UL16_JRV3_MC_PtResolution_AK4PFchs";
                jes_Unc_name = "Summer19UL16_V7_MC_Total_AK4PFchs";
                if( _sysName_lateral == "JES" ) jes_Unc_name = "Summer19UL16_V7_MC_"+_SysSubSource+"_AK4PFchs";
            }
        }else if( dataset_year == "17" ){
            jer_SF_corr_name = "Summer19UL17_JRV2_MC_ScaleFactor_AK4PFchs";
                jer_PtRes_corr_name = "Summer19UL17_JRV2_MC_PtResolution_AK4PFchs";
                jes_Unc_name = "Summer19UL17_V5_MC_Total_AK4PFchs";
                if( _sysName_lateral == "JES" ) jes_Unc_name = "Summer19UL17_V5_MC_"+_SysSubSource+"_AK4PFchs";
        }else if( dataset_year == "18" ){
            jer_SF_corr_name = "Summer19UL18_JRV2_MC_ScaleFactor_AK4PFchs";
                jer_PtRes_corr_name = "Summer19UL18_JRV2_MC_PtResolution_AK4PFchs";
                jes_Unc_name = "Summer19UL18_V5_MC_Total_AK4PFchs";
                if( _sysName_lateral == "JES" ) jes_Unc_name = "Summer19UL18_V5_MC_"+_SysSubSource+"_AK4PFchs";
        }
        
        if( apply_jer_corr ){
            jet_JER_SF_corr = jet_jerc_set->at(jer_SF_corr_name);
            jet_JER_PtRes_corr = jet_jerc_set->at(jer_PtRes_corr_name);
        }
        jet_JES_Unc = jet_jerc_set->at(jes_Unc_name);
        
        /*
        shared_ptr<correction::Correction const> jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_Total_AK4PFchs");
        double total_jes = jet_JES_cout->evaluate({1.3, 30.});
        cout << "Total " << total_jes << endl;
        
        double part_jes = 0;
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_AbsoluteMPFBias_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "AbsoluteMPFBias " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_AbsoluteScale_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "AbsoluteScale " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_AbsoluteStat_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "AbsoluteStat " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_FlavorQCD_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "FlavorQCD " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_Fragmentation_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "Fragmentation " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_PileUpDataMC_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "PileUpDataMC " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_PileUpPtBB_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "PileUpPtBB " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_PileUpPtEC1_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "PileUpPtEC1 " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_PileUpPtEC2_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "PileUpPtEC2 " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_PileUpPtHF_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "PileUpPtHF " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_PileUpPtRef_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "PileUpPtRef " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_RelativeFSR_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "RelativeFSR " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_RelativeJEREC1_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "RelativeJEREC1 " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_RelativeJEREC2_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "RelativeJEREC2 " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_RelativeJERHF_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "RelativeJERHF " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_RelativePtBB_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "RelativePtBB " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_RelativePtEC1_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "RelativePtEC1 " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_RelativePtEC2_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "RelativePtEC2 " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_RelativePtHF_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "RelativePtHF " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_RelativeBal_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "RelativeBal " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_RelativeSample_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "RelativeSample " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_RelativeStatEC_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "RelativeStatEC " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_RelativeStatFSR_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "RelativeStatFSR " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_RelativeStatHF_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "RelativeStatHF " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_SinglePionECAL_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "SinglePionECAL " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_SinglePionHCAL_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "SinglePionHCAL " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        jet_JES_cout = jet_jerc_set->at("Summer19UL16APV_V7_MC_TimePtEta_AK4PFchs");
        part_jes += pow(jet_JES_cout->evaluate({1.3, 30.}),2);
        cout << "TimePtEta " << jet_JES_cout->evaluate({1.3, 30.}) << endl;
        
        part_jes = sqrt(part_jes);
        cout << "Check Total " << part_jes << endl;
        */
        
        
        //----B TAGGING--------------------------------------------------------------------------------
        if( apply_btag_wgt ){
            
            string dsName;
            if( dataset_HIPM ){
                dsName = _datasetName.substr(0,_datasetName.length()-7);
            }else{
                dsName = _datasetName.substr(0,_datasetName.length()-3);
            }
            string dsName10 = dsName.substr(0,10);
            if( dsName10 == "DYJetsToLL" ) dsName = dsName10;
            
            btag_eff.readFile(btag_eff_file);
            if( dataset_group != "Data" ) btag_eff.calib(dsName, "TTTo2L2Nu");
            
            // Choose btag algo
            // https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL18
            std::string btagAlgorithmMujets;
            std::string btagAlgorithmIncl;
            std::string btagAlgorithmComb;

            if (JET_BTAG_WP >= 0 and JET_BTAG_WP <= 2) {
                btagAlgorithmMujets = "deepJet_mujets";
                btagAlgorithmIncl = "deepJet_incl";
                btagAlgorithmComb = "deepJet_comb";
            }
            else if (JET_BTAG_WP >= 3 and JET_BTAG_WP <= 5) {
                btagAlgorithmMujets = "deepCSV_mujets";
                btagAlgorithmIncl = "deepCSV_incl";
                btagAlgorithmComb = "deepCSV_comb";
            }

            auto btag_set = correction::CorrectionSet::from_file(btag_SF_file.c_str());        
            btag_bc_corr = btag_set->at(btagAlgorithmMujets.c_str());
            btag_udsg_corr = btag_set->at(btagAlgorithmIncl.c_str());
        }
        
        //----TRIGGER----------------------------------------------------------------------------------
        if( apply_trigger_wgt ){
            triggerSF.readFile(trigger_SF_file);
        }
        
        //----PILEUP-----------------------------------------------------------------------------------
        if( apply_pileup_wgt ){
            auto pileup_set = correction::CorrectionSet::from_file(pileup_file.c_str());
            string SetName = "Collisions" + dataset_year +"_UltraLegacy_goldenJSON";
            pileup_corr = pileup_set->at(SetName.c_str());
        }
        
        //----MUON ROCHESTER---------------------------------------------------------------------------
        if( apply_muon_roc_corr ){
            muon_roc_corr.Initialize(muon_roc_file);
        }
        
        //----MET RECOIL-------------------------------------------------------------------------------
        //if( apply_met_recoil_corr ){
        //    Z_recoil.ReadFile(Z_recoil_file.c_str());
        //}
        
        
        //=================================================================================================================
        // ANALYSIS SETUP
        //=================================================================================================================
        
        //----MACHINE LEARNING-------------------------------------------------------------------------
        MLP_keras.readFiles( model_keras_file, preprocessing_keras_file );
        MLP_torch.readFile( model_torch_file ); 
        
    }
    
}


//---------------------------------------------------------------------------------------------------------------
// Objects Setup [run before genWeight count]
//--------------------------------------------------------------------------------------------------------------- 
bool HEPHero::RunObjects() {
    
    if( _ANALYSIS == "GEN" ){
        //======HEPHeroGEN===============================================================
        _evt.set_units(HepMC3::Units::GEV, HepMC3::Units::MM);
        
        weights_value = _evt.weights();
        if( _EventPosition == 0 ) weights_name = _ascii_file->run_info()->weight_names();
        
        // EVENT INFO
        if( _EventPosition == 0 ){
            if( _has_pdf ){
                std::shared_ptr<HepMC3::GenPdfInfo> pdf = _evt.attribute<HepMC3::GenPdfInfo>("GenPdfInfo");
                pdf_id1 = pdf->pdf_id[0];
                pdf_id2 = pdf->pdf_id[1];
            }else{
                pdf_id1 = -1;
                pdf_id2 = -1;
            }
            
            _N_PS_weights = _evt.weights().size();
            
            momentum_unit_id = _evt.momentum_unit();
            length_unit_id = _evt.length_unit();
            _momentum_unit = "MeV";
            _length_unit = "mm";
            if( momentum_unit_id == 1 ) _momentum_unit = "GeV";
            if( length_unit_id == 1 ) _length_unit = "cm";
            
        }
        
        if( _EventPosition+1 == _NumberEntries ){
            if( _has_xsec ){
                std::shared_ptr<HepMC3::GenCrossSection> cs = _evt.attribute<HepMC3::GenCrossSection>("GenCrossSection");
                cross_section = cs->xsec("Default");
                cross_section_unc = cs->xsec_err("Default");
            }else{
                cross_section = -1;
                cross_section_unc = -1;
            }
        }
    }else{
        //======SUM THE GENERATOR WEIGHTS================================================
        SumGenWeights += genWeight;
        
        if( !MC_processing() ) return false;
    }
    
    return true;
}


    

    
