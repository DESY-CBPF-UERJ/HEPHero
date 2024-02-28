#include "HEPHero.h"


//---------------------------------------------------------------------------------------------------------------
// Regions OpenData
//---------------------------------------------------------------------------------------------------------------
void HEPHero::RegionsOD(){
    RegionID = -1;

    if( (Nbjets == 0) &&
        (Nmuons == 1) &&
        (MuonL_MET_Mt < 65)
      ){                                        // [Signal Region]
        RegionID = 0;
    }
    else if( (Nbjets == 0) &&
        (Nmuons == 1) &&
        (MuonL_MET_Mt >= 65)
      ){                                        // [Wjets - Control Region]
        RegionID = 1;
    }
    else if( (Nbjets >= 1) &&
        (Nmuons == 1) &&
        (MuonL_MET_Mt < 65)
      ){                                        // [ttbar - Control Region]
        RegionID = 2;
    }
    else if( (Nbjets == 0) &&
        (Nmuons == 2) &&
        (MuonL_MET_Mt < 65) &&
        (LepLep_deltaM < 15) &&
        (Has_2OC_muons == 1)
      ){                                        // [DY - Control Region]
        RegionID = 3;
    }
}


//---------------------------------------------------------------------------------------------------------------
// Lepton selection OpenData
//---------------------------------------------------------------------------------------------------------------
void HEPHero::LeptonSelectionOD(){

    selectedMu.clear();
    for( unsigned int imu = 0; imu < nMuon; ++imu ) {
        if( abs(Muon_eta[imu]) >= MUON_ETA_CUT ) continue;
        if( Muon_pt[imu] <= MUON_PT_CUT ) continue;
        if( !Muon_tightId ) continue;
        selectedMu.push_back(imu);
    }

    selectedTau.clear();
    for( unsigned int itau = 0; itau < nTau; ++itau ) {
        if( Tau_charge == 0 ) continue;
        if( abs(Tau_eta[itau]) >= TAU_ETA_CUT ) continue;
        if( Tau_pt[itau] <= TAU_PT_CUT ) continue;
        if( !Tau_idDecayMode ) continue;
        if( !Tau_idIsoTight ) continue;
        if( !Tau_idAntiEleTight ) continue;
        if( !Tau_idAntiMuTight ) continue;
        selectedTau.push_back(itau);
    }

    Nmuons = selectedMu.size();
    Ntaus = selectedTau.size();

    Has_2OC_muons = false;
    if( Nmuons == 2 ){
        if( Muon_charge[selectedMu[0]]*Muon_charge[selectedMu[1]] < 0 ) Has_2OC_muons = true;
        lep_1.SetPtEtaPhiM(Muon_pt[selectedMu[0]], Muon_eta[selectedMu[0]], Muon_phi[selectedMu[0]], Muon_mass[selectedMu[0]]);
        lep_2.SetPtEtaPhiM(Muon_pt[selectedMu[1]], Muon_eta[selectedMu[1]], Muon_phi[selectedMu[1]], Muon_mass[selectedMu[1]]);
        LepLep = lep_1 + lep_2;
        LepLep_mass = LepLep.M();
        LepLep_deltaM = abs( LepLep_mass - Z_pdg_mass );
        LepLep_pt = LepLep.Pt();
        LepLep_phi = LepLep.Phi();
    }else if( Nmuons == 1 ){
        LepLep_mass = Muon_mass[selectedMu[0]];
        LepLep_deltaM = abs( LepLep_mass - Z_pdg_mass );
        LepLep_pt = Muon_pt[selectedMu[0]];
        LepLep_phi = Muon_phi[selectedMu[0]];
    }
}


//---------------------------------------------------------------------------------------------------------------
// Muon-Tau pair selection OpenData
//---------------------------------------------------------------------------------------------------------------
bool HEPHero::MuonTauPairSelectionOD(){

    bool has_good_pair = false;

    IdxBestMu = -1;
    float maxpt = -1;
    for( unsigned int iselmu = 0; iselmu < selectedMu.size(); ++iselmu ) {
        int imu = selectedMu[iselmu];

        if( !((Muon_pt[imu] > maxpt) || (Muon_pt[imu] <= maxpt && IdxBestMu == -1)) ) continue;

        IdxBestTau = -1;
        float minIso = 999999999.;
        for( unsigned int iseltau = 0; iseltau < selectedTau.size(); ++iseltau ) {
            int itau = selectedTau[iseltau];

            if( Muon_charge[imu]*Tau_charge[itau] > 0 ) continue;

            if( Tau_relIso_all[itau] < minIso ){
                maxpt = Muon_pt[imu];
                IdxBestMu = imu;
                minIso = Tau_relIso_all[itau];
                IdxBestTau = itau;
                has_good_pair = true;
            }
        }
    }

    return has_good_pair;
}


//---------------------------------------------------------------------------------------------------------------
// JES uncertainty
//---------------------------------------------------------------------------------------------------------------
void HEPHero::JESvariationOD(){

    //if( (_sysName_lateral == "JES") && (dataset_group != "Data") ){
    if( ((_sysName_lateral == "JES") || ((_sysName_lateral == "Recoil") && (_Universe > 3)))  && (dataset_group != "Data") ){
        METCorrectionFromJES.SetXYZT(0.,0.,0.,0.);

        for( unsigned int ijet = 0; ijet < nJet; ++ijet ) {
            TLorentzVector JetLV_before;
            JetLV_before.SetPtEtaPhiM(Jet_pt[ijet],Jet_eta[ijet],Jet_phi[ijet],Jet_mass[ijet]);

            float jet_pt = Jet_pt[ijet];
            float jet_eta = Jet_eta[ijet];
            float jet_area = Jet_area[ijet];

            if( jet_pt <= 10 ) continue;
            if( jet_eta >= 5.2 ) continue;

            //float pt_unc = JES_unc.getUnc( jet_eta, jet_pt );
            float pt_unc = jet_JES_Unc->evaluate({jet_eta, jet_pt});


            if( _Universe % 2 == 0 ){
                Jet_pt[ijet] = jet_pt*(1. - pt_unc);
            }else{
                Jet_pt[ijet] = jet_pt*(1. + pt_unc);
            }

            TLorentzVector JetLV_after;
            JetLV_after.SetPtEtaPhiM(Jet_pt[ijet],Jet_eta[ijet],Jet_phi[ijet],Jet_mass[ijet]);

            METCorrectionFromJES -= JetLV_after - JetLV_before;
        }

    }
}


//---------------------------------------------------------------------------------------------------------------
// Jet lep overlap
//---------------------------------------------------------------------------------------------------------------
void HEPHero::Jet_lep_overlapOD( float deltaR_cut ){

    Jet_LepOverlap.clear();
    for( unsigned int ijet = 0; ijet < nJet; ++ijet ) {
        Jet_LepOverlap.push_back(false);
    }

    if( IdxBestTau >= 0 ){
        for( unsigned int ijet = 0; ijet < nJet; ++ijet ) {
            double deta = fabs(Tau_eta[IdxBestTau] - Jet_eta[ijet]);
            double dphi = fabs(Tau_phi[IdxBestTau] - Jet_phi[ijet]);
            if( dphi > M_PI ) dphi = 2*M_PI - dphi;
            double dr = sqrt( deta*deta + dphi*dphi );
            if( dr < deltaR_cut ){
                Jet_LepOverlap[ijet] = true;
            }
        }
    }

    for( unsigned int iselMu = 0; iselMu < selectedMu.size(); ++iselMu ) {
        int imu = selectedMu.at(iselMu);
        float drMin = 99999.;
        int JetID = -1;
        for( unsigned int ijet = 0; ijet < nJet; ++ijet ) {
            double deta = fabs(Muon_eta[imu] - Jet_eta[ijet]);
            double dphi = fabs(Muon_phi[imu] - Jet_phi[ijet]);
            if( dphi > M_PI ) dphi = 2*M_PI - dphi;
            double dr = sqrt( deta*deta + dphi*dphi );
            if( dr < drMin ){
                drMin = dr;
                JetID = ijet;
            }
        }
        if( (drMin < deltaR_cut) && (JetID >= 0) ) Jet_LepOverlap[JetID] = true;
    }

}


//---------------------------------------------------------------------------------------------------------------
// Jet selection OpenData
//---------------------------------------------------------------------------------------------------------------
void HEPHero::JetSelectionOD(){

    selectedJet.clear();

    JESvariationOD();

    Jet_lep_overlap( JET_LEP_DR_ISO_CUT );

    Nbjets = 0;
    Njets = 0;
    Njets30 = 0;
    Njets_forward = 0;
    Njets30_forward = 0;
    Njets_ISR = 0;
    HT = 0;
    HT30 = 0;
    float HPx = 0;
    float HPy = 0;
    float HPx30 = 0;
    float HPy30 = 0;
    for( unsigned int ijet = 0; ijet < nJet; ++ijet ) {

        if( Jet_pt[ijet] <= JET_PT_CUT ) continue;
        if( Jet_LepOverlap[ijet] ) continue;
        if( (Jet_pt[ijet] < 50) && (Jet_puId[ijet] == 0) ) continue;
        if( abs(Jet_eta[ijet]) >= 4.7 ) continue;
        if( abs(Jet_eta[ijet]) > 1.4 ){
            Njets_forward += 1;
            if( Jet_pt[ijet] > 30 ) Njets30_forward += 1;
        }
        if( abs(Jet_eta[ijet]) >= JET_ETA_CUT ) continue;
        selectedJet.push_back(ijet);
        TLorentzVector Jet;
        Jet.SetPtEtaPhiE(Jet_pt[ijet], Jet_eta[ijet], Jet_phi[ijet], 0);

        Njets += 1;
        HT += Jet_pt[ijet];
        HPx += Jet.Px();
        HPy += Jet.Py();
        if( Jet_pt[ijet] > 30 ){
            Njets30 += 1;
            HT30 += Jet_pt[ijet];
            HPx30 += Jet.Px();
            HPy30 += Jet.Py();
        }
        if( Jet_btag[ijet] > 0.8 ){
            Nbjets += 1;
        }
        if( Jet_pt[ijet] > 26 ) Njets_ISR += 1;
    }
    MHT = sqrt(HPx*HPx + HPy*HPy);
    MHT30 = sqrt(HPx30*HPx30 + HPy30*HPy30);

    LeadingJet_pt = 0;
    SubLeadingJet_pt = 0;
    if( Njets > 0 ) LeadingJet_pt = Jet_pt[selectedJet.at(0)];
    if( Njets > 1 ) SubLeadingJet_pt = Jet_pt[selectedJet.at(1)];

}


//---------------------------------------------------------------------------------------------------------------
// Get TauTau variables
//---------------------------------------------------------------------------------------------------------------
void HEPHero::Jet_TauTau_VariablesOD(){

    int idxMu = selectedMu[0];
    int idxTau = IdxBestTau;

    TauH_pt = Tau_pt[idxTau];
    MuonL_pt = Muon_pt[idxMu];

    TLorentzVector MuonL_L;
    MuonL_L.SetPtEtaPhiM(Muon_pt[idxMu], Muon_eta[idxMu], Muon_phi[idxMu], Muon_mass[idxMu]);
    TLorentzVector TauH_L;
    TauH_L.SetPtEtaPhiM(Tau_pt[idxTau], Tau_eta[idxTau], Tau_phi[idxTau], Tau_mass[idxTau]);
    TLorentzVector MET_L;
    MET_L.SetPxPyPzE(MET_pt*cos(MET_phi), MET_pt*sin(MET_phi), 0., 0.);

    TLorentzVector MuonL_MET_L;
    MuonL_MET_L = MuonL_L + MET_L;

    MuonL_MET_pt = MuonL_MET_L.Pt();
    MuonL_MET_dphi = abs( Muon_phi[idxMu] - MET_phi );
    if( MuonL_MET_dphi > M_PI ) MuonL_MET_dphi = 2*M_PI - MuonL_MET_dphi;
    MuonL_MET_Mt = sqrt( 2*MuonL_pt*MET_pt*( 1 - cos( MuonL_MET_dphi ) ) );

    TLorentzVector TauH_TauL_L;
    TauH_TauL_L = MuonL_MET_L + TauH_L;

    TauH_TauL_pt = TauH_TauL_L.Pt();
    TauH_TauL_dphi = abs( MuonL_MET_L.Phi() - TauH_L.Phi() );
    if( TauH_TauL_dphi > M_PI ) TauH_TauL_dphi = 2*M_PI - TauH_TauL_dphi;
    TauH_TauL_Mt = sqrt( 2*MuonL_MET_L.Pt()*TauH_L.Pt()*( 1 - cos( TauH_TauL_dphi ) ) );

    TLorentzVector TauH_MuonL_L;
    TauH_MuonL_L = MuonL_L + TauH_L;

    TauH_MuonL_pt = TauH_MuonL_L.Pt();
    TauH_MuonL_M = TauH_MuonL_L.M();
    TauH_MuonL_dr = TauH_L.DeltaR( MuonL_L );


    TLorentzVector Jet_L;
    Jet_L.SetPtEtaPhiM(Jet_pt[selectedJet.at(0)], Jet_eta[selectedJet.at(0)], Jet_phi[selectedJet.at(0)], Jet_mass[selectedJet.at(0)]);

    LeadingJet_MuonL_dr = Jet_L.DeltaR( MuonL_L );
    LeadingJet_TauL_dphi = abs( Jet_L.Phi() - TauH_L.Phi() );
    if( LeadingJet_TauL_dphi > M_PI ) LeadingJet_TauL_dphi = 2*M_PI - LeadingJet_TauL_dphi;
    LeadingJet_TauH_dr = Jet_L.DeltaR( TauH_L );
    LeadingJet_TauHMuonL_dr = Jet_L.DeltaR( TauH_MuonL_L );


}


//---------------------------------------------------------------------------------------------------------------
// MLP Model for signal discrimination OpenData
//---------------------------------------------------------------------------------------------------------------
void HEPHero::Signal_discriminatorsOD(){

    //MLP_score_keras = MLP_keras.predict({MuonL_pt, MET_pt, TauH_MuonL_M, MuonL_MET_pt, MuonL_MET_dphi, MuonL_MET_Mt, TauH_TauL_Mt, LeadingJet_TauHMuonL_dr});

    MLP_score_torch = MLP_torch.predict({MuonL_pt, MET_pt, TauH_MuonL_M, MuonL_MET_pt, MuonL_MET_dphi, MuonL_MET_Mt, TauH_TauL_Mt, LeadingJet_TauHMuonL_dr});

    MLP4_score_torch = pow(1.e4,MLP_score_torch)/1.e4;

}


//---------------------------------------------------------------------------------------------------------------
// MET Correction OpenData
//---------------------------------------------------------------------------------------------------------------
void HEPHero::METCorrectionOD(){

    string dsName;
    if( dataset_HIPM ){
        dsName = _datasetName.substr(0,_datasetName.length()-7);
    }else{
        dsName = _datasetName.substr(0,_datasetName.length()-3);
    }
    string dsNameDY = dsName.substr(0,10);
    string dsNameZZ = dsName.substr(0,6);

    MET_RAW_pt = MET_pt;
    MET_RAW_phi = MET_phi;

    //=====JES Variation===========================================================================
    //if( (_sysName_lateral == "JES") && (dataset_group != "Data") && !(apply_met_recoil_corr && (dsNameDY == "DYJetsToLL")) ){
    if( (_sysName_lateral == "JES") && (dataset_group != "Data") ){
        TLorentzVector METLV;
        METLV.SetPtEtaPhiM(MET_pt, 0., MET_phi, 0.);
        METLV += METCorrectionFromJES;

        MET_pt = METLV.Pt();
        MET_phi = METLV.Phi();
    }

    MET_JES_pt = MET_pt;
    MET_JES_phi = MET_phi;

    /*
    //=====Recoil Correction=======================================================================
    if( apply_met_recoil_corr && (dsNameDY == "DYJetsToLL") ){

        Ux = -(MET_pt*cos(MET_phi) + LepLep_pt*cos(LepLep_phi));
        Uy = -(MET_pt*sin(MET_phi) + LepLep_pt*sin(LepLep_phi));

        U1 =  Ux*cos(LepLep_phi) + Uy*sin(LepLep_phi);
        U2 = -Ux*sin(LepLep_phi) + Uy*cos(LepLep_phi);

        vector<float> mc_u1_mean;
        vector<float> mc_u1_mean_unc_up;
        vector<float> mc_u1_mean_unc_down;
        vector<float> diff_u1;
        vector<float> diff_u1_unc_up;
        vector<float> diff_u1_unc_down;
        float sigma_ratio;
        float sigma_ratio_unc_up;
        float sigma_ratio_unc_down;
        vector<float> ZRecoPt;


        if( dataset_year == "12" ){
            sigma_ratio = 1.052;
            sigma_ratio_unc_up = 0.001;
            sigma_ratio_unc_down = 0.001;

            if( Njets_ISR == 0 ){
                mc_u1_mean = {-3.202, -11.509, -21.217, -30.635, -39.378, -48.61, -57.343, -66.751};
                mc_u1_mean_unc_up = {0.009, 0.004, 0.007, 0.015, 0.015, 0.025, 0.027, 0.072};
                mc_u1_mean_unc_down = {0.007, 0.001, 0.002, 0.011, 0.016, 0.018, 0.026, 0.077};

                diff_u1 = {0.119, 0.441, 0.763, 1.085, 1.406, 1.728, 2.05, 2.372};
                diff_u1_unc_up = {0.002, 0.001, 0.004, 0.006, 0.009, 0.012, 0.015, 0.018};
                diff_u1_unc_down = {0.004, 0.004, 0.009, 0.015, 0.02, 0.026, 0.031, 0.037};

                ZRecoPt = {0, 10, 20, 30, 40, 50, 60, 70, 80};
            }else if( Njets_ISR == 1 ){
                mc_u1_mean = {-3.931, -14.576, -26.081, -35.782, -44.595, -53.538, -62.76, -72.701, -81.228, -91.498, -101.16, -110.261, -120.277};
                mc_u1_mean_unc_up = {0.016, 0.022, 0.046, 0.061, 0.002, 0.009, 0.003, 0.007, 0.001, 0.02, 0.016, 0.012, 0.051};
                mc_u1_mean_unc_down = {0.015, 0.02, 0.044, 0.05, 0.001, 0.016, 0.007, 0.005, 0.009, 0.017, 0.016, 0.021, 0.056};

                diff_u1 = {-0.615, -0.379, -0.143, 0.094, 0.33, 0.566, 0.802, 1.039, 1.275, 1.511, 1.747, 1.984, 2.22};
                diff_u1_unc_up = {0.029, 0.026, 0.023, 0.02, 0.017, 0.015, 0.012, 0.01, 0.008, 0.008, 0.008, 0.009, 0.009};
                diff_u1_unc_down = {0.013, 0.013, 0.013, 0.013, 0.014, 0.014, 0.015, 0.016, 0.017, 0.018, 0.02, 0.022, 0.024};

                ZRecoPt = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130};
            }else if( Njets_ISR >= 2 ){
                mc_u1_mean = {-91.241};
                mc_u1_mean_unc_up = {0.051};
                mc_u1_mean_unc_down = {0.063};

                diff_u1 = {0.973};
                diff_u1_unc_up = {0.054};
                diff_u1_unc_down = {0.048};

                ZRecoPt = {90, 100};
            }
        }

        int idx = -1;
        for (unsigned int i = 0; i < mc_u1_mean.size() ; i++){
            if ( LepLep_pt >= ZRecoPt[i] && LepLep_pt < ZRecoPt[i+1]  ){
                idx = i;
                break;
            }
        }

        if( _sysName_lateral == "Recoil" ){
            if( _Universe <= 1 ){   // ratio up
                sigma_ratio = sigma_ratio + sigma_ratio_unc_up;
            }else{                  // ratio down
                sigma_ratio = sigma_ratio - sigma_ratio_unc_down;
            }
        }

        float U2_new = U2*sigma_ratio;
        float Ux_new = - U2_new*sin(LepLep_phi);
        float Uy_new = + U2_new*cos(LepLep_phi);

        if( idx >= 0 ){

            float diff_u1_value = diff_u1[idx];
            float mc_u1_mean_value = mc_u1_mean[idx];

            if( _sysName_lateral == "Recoil" ){
                if( _Universe == 0 ){           // ratio up, diff increase
                    if( diff_u1_value >= 0 ){
                        diff_u1_value = diff_u1_value + diff_u1_unc_up[idx];
                        mc_u1_mean_value = mc_u1_mean_value - mc_u1_mean_unc_down[idx];
                    }else{
                        diff_u1_value = diff_u1_value - diff_u1_unc_down[idx];
                        mc_u1_mean_value = mc_u1_mean_value + mc_u1_mean_unc_up[idx];
                    }
                }else if( _Universe == 1 ){     // ratio up, diff decrease
                    if( diff_u1_value >= 0 ){
                        diff_u1_value = diff_u1_value - diff_u1_unc_down[idx];
                        mc_u1_mean_value = mc_u1_mean_value + mc_u1_mean_unc_up[idx];
                    }else{
                        diff_u1_value = diff_u1_value + diff_u1_unc_up[idx];
                        mc_u1_mean_value = mc_u1_mean_value - mc_u1_mean_unc_down[idx];
                    }
                }else if( _Universe == 2 ){     // ratio down, diff increase
                    if( diff_u1_value >= 0 ){
                        diff_u1_value = diff_u1_value + diff_u1_unc_up[idx];
                        mc_u1_mean_value = mc_u1_mean_value - mc_u1_mean_unc_down[idx];
                    }else{
                        diff_u1_value = diff_u1_value - diff_u1_unc_down[idx];
                        mc_u1_mean_value = mc_u1_mean_value + mc_u1_mean_unc_up[idx];
                    }
                }else if( _Universe == 3 ){     // ratio down, diff decrease
                    if( diff_u1_value >= 0 ){
                        diff_u1_value = diff_u1_value - diff_u1_unc_down[idx];
                        mc_u1_mean_value = mc_u1_mean_value + mc_u1_mean_unc_up[idx];
                    }else{
                        diff_u1_value = diff_u1_value + diff_u1_unc_up[idx];
                        mc_u1_mean_value = mc_u1_mean_value - mc_u1_mean_unc_down[idx];
                    }
                }
            }

            float U1_new = (mc_u1_mean_value+diff_u1_value) + (U1-mc_u1_mean_value)*sigma_ratio;
            Ux_new += U1_new*cos(LepLep_phi);
            Uy_new += U1_new*sin(LepLep_phi);

        }else{
            Ux_new += U1*cos(LepLep_phi);
            Uy_new += U1*sin(LepLep_phi);
        }

        float CorrectedMET_x = -(Ux_new + LepLep_pt*cos(LepLep_phi));
        float CorrectedMET_y = -(Uy_new + LepLep_pt*sin(LepLep_phi));

        double CorrectedMET = sqrt(CorrectedMET_x*CorrectedMET_x+CorrectedMET_y*CorrectedMET_y);
        double CorrectedMETPhi;
        if(CorrectedMET_x==0 && CorrectedMET_y>0) CorrectedMETPhi = TMath::Pi();
        else if(CorrectedMET_x==0 && CorrectedMET_y<0 ) CorrectedMETPhi = -TMath::Pi();
        else if(CorrectedMET_x >0) CorrectedMETPhi = TMath::ATan(CorrectedMET_y/CorrectedMET_x);
        else if(CorrectedMET_x <0 && CorrectedMET_y>0) CorrectedMETPhi = TMath::ATan(CorrectedMET_y/CorrectedMET_x) + TMath::Pi();
        else if(CorrectedMET_x <0 && CorrectedMET_y<0) CorrectedMETPhi = TMath::ATan(CorrectedMET_y/CorrectedMET_x) - TMath::Pi();
        else CorrectedMETPhi =0;

        MET_pt = CorrectedMET;
        MET_phi = CorrectedMETPhi;

    }

    MET_RECOIL_pt = MET_pt;
    MET_RECOIL_phi = MET_phi;
    */

    //=====MET Emulation===========================================================================
    // MET Emulation for region with 2 muons
    bool control = false;
    double METxcorr;
    double METycorr;

    if( Nmuons == 2 ){
        for( unsigned int iselmu = 0; iselmu < Nmuons; ++iselmu ) {
            int imu = selectedMu[iselmu];
            if( imu != IdxBestMu ){
                METxcorr = Muon_pt[imu]*cos(Muon_phi[imu]);
                METycorr = Muon_pt[imu]*sin(Muon_phi[imu]);
                control = true;
            }
        }
    }

    if( control ){
        double EmulatedMET_x = MET_pt*cos(MET_phi) + METxcorr;
        double EmulatedMET_y = MET_pt*sin(MET_phi) + METycorr;

        double EmulatedMET_pt = sqrt(EmulatedMET_x*EmulatedMET_x+EmulatedMET_y*EmulatedMET_y);

        double EmulatedMET_phi;
        if(EmulatedMET_x==0 && EmulatedMET_y>0) EmulatedMET_phi = TMath::Pi();
        else if(EmulatedMET_x==0 && EmulatedMET_y<0 )EmulatedMET_phi = -TMath::Pi();
        else if(EmulatedMET_x >0) EmulatedMET_phi = TMath::ATan(EmulatedMET_y/EmulatedMET_x);
        else if(EmulatedMET_x <0 && EmulatedMET_y>0) EmulatedMET_phi = TMath::ATan(EmulatedMET_y/EmulatedMET_x) + TMath::Pi();
        else if(EmulatedMET_x <0 && EmulatedMET_y<0) EmulatedMET_phi = TMath::ATan(EmulatedMET_y/EmulatedMET_x) - TMath::Pi();
        else EmulatedMET_phi =0;

        MET_pt = EmulatedMET_pt;
        MET_phi = EmulatedMET_phi;
    }

    MET_Emu_pt = MET_pt;
    MET_Emu_phi = MET_phi;


}


//---------------------------------------------------------------------------------------------------------------
// Pileup Correction OpenData
//---------------------------------------------------------------------------------------------------------------
float HEPHero::GetPileupWeightOD( string sysType ){

    double pileup_weight = 1.;

    string dsName = _datasetName.substr(0,_datasetName.length()-3);
    string dsNameDY = dsName.substr(0,10);
    if( dataset_group != "Data" ){

        vector<float> NPV_Bkg = { 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
        14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26.,
        27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39.,
        40., 41., 42., 43., 44., 45., 46., 70.};

        vector<float> facMCToDY = {0.60596421, 0.77569228, 0.88825352, 1.00762809, 1.12165009,
        1.22203554, 1.30064631, 1.34465674, 1.35895262, 1.34339587,
        1.31254424, 1.27193365, 1.22723794, 1.185854  , 1.14467599,
        1.10458676, 1.06382983, 1.02145148, 0.97510731, 0.92408506,
        0.86776156, 0.80618604, 0.74098505, 0.67491278, 0.60519701,
        0.53878726, 0.47457879, 0.41269139, 0.35725551, 0.3059752 ,
        0.25900109, 0.21655361, 0.18108055, 0.1525257 , 0.12465582,
        0.09987348, 0.08263982, 0.06703216, 0.05517279, 0.04314179,
        0.0365397 , 0.0259733 , 0.02333173, 0.01685579, 0.01369018,
        0.00638677};

        vector<float> facMC;
        if( sysType == "nominal" ){
            facMC = {1.26095568, 1.1448191 , 1.1112832 , 1.0753875 , 1.0393764 ,
            1.00625596, 0.9760511 , 0.95553768, 0.94315003, 0.93705221,
            0.93774467, 0.94363179, 0.95527794, 0.96712292, 0.98284986,
            0.99812465, 1.01220839, 1.02898245, 1.04158749, 1.05250627,
            1.06314025, 1.07276411, 1.08164585, 1.08801208, 1.09503861,
            1.0966751 , 1.10768205, 1.11638351, 1.12379529, 1.131215  ,
            1.14497507, 1.16289297, 1.18270749, 1.19902304, 1.24003539,
            1.28704313, 1.31651459, 1.38558137, 1.41052386, 1.52716485,
            1.53607581, 1.80524014, 1.75340122, 2.07447395, 2.41533703,
            3.63074595};
        }else if( sysType == "up" ){
            facMC = {1.52191136, 1.28963821, 1.2225664 , 1.150775  , 1.07875281,
            1.01251192, 0.9521022 , 0.91107536, 0.88630006, 0.87410442,
            0.87548933, 0.88726357, 0.91055587, 0.93424583, 0.96569971,
            0.99624929, 1.02441678, 1.05796489, 1.08317499, 1.10501254,
            1.1262805 , 1.14552822, 1.1632917 , 1.17602416, 1.19007721,
            1.1933502 , 1.21536411, 1.23276703, 1.24759058, 1.26243   ,
            1.28995014, 1.32578593, 1.36541499, 1.39804608, 1.48007078,
            1.57408626, 1.63302919, 1.77116275, 1.82104771, 2.0543297 ,
            2.07215161, 2.61048028, 2.50680245, 3.1489479 , 3.83067407,
            6.26149191};
        }else if( sysType == "down" ){
            facMC = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
        }

        int idx = -1;
        for (unsigned int i = 0; i < facMC.size() ; i++){
            if ( PV_npvs >= NPV_Bkg[i] && PV_npvs < NPV_Bkg[i+1]  ){
                idx = i;
                break;
            }
        }
        if(idx >= 0){
            if(dsNameDY != "DYJetsToLL") pileup_weight *= facMCToDY[idx];
            pileup_weight *= facMC[idx];
        }
    }

    return pileup_weight;
}

/*
//---------------------------------------------------------------------------------------------------------------
// Trigger Correction OpenData
//---------------------------------------------------------------------------------------------------------------
float HEPHero::GetTriggerWeightOD( string sysID ){

    double trigger_weight = 1.;
    if( dataset_group != "Data" ){

        float LeadingTau_pt = Tau_pt[selectedTau[0]];
        vector<float> TauPt_MC;
        vector<float> facMC;
        if( sysID == "nominal" ){
            TauPt_MC = { 30.,  35.,  40.,  45.,  50.,  55.,  60.,  65.,  70.,  75.,  80.,
                        85.,  90.,  95., 100., 105., 110., 115., 120., 160., 200., 240.,
                        500.};
            facMC = {1.11073889, 1.11503933, 1.11478038, 1.10477414, 1.10039739,
                    1.09222083, 1.0880355 , 1.08310863, 1.07507577, 1.07052099,
                    1.06339093, 1.06608262, 1.05500631, 1.04647864, 1.04974252,
                    1.05710489, 1.0569654 , 1.07337116, 1.05810618, 1.08416867,
                    1.07225847, 1.0741809};
        }

        int idx = -1;
        for (unsigned int i = 0; i < facMC.size() ; i++){
            if ( LeadingTau_pt >= TauPt_MC[i] && LeadingTau_pt < TauPt_MC[i+1]  ){
                idx = i;
                break;
            }
        }
        if(idx >= 0) trigger_weight = facMC[idx];
    }

    return trigger_weight;
}
*/


//---------------------------------------------------------------------------------------------------------------
// Weight corrections OpenData
//---------------------------------------------------------------------------------------------------------------
void HEPHero::Weight_correctionsOD(){

    pileup_wgt = 1.;
    //trigger_wgt = 1.;

    if(dataset_group != "Data"){

        if( apply_pileup_wgt ){
            pileup_wgt = GetPileupWeightOD("nominal");
            evtWeight *= pileup_wgt;
        }

        //if( apply_trigger_wgt ){
        //    trigger_wgt = GetTriggerWeightOD("nominal");
        //    evtWeight *= trigger_wgt;
        //}

    }
}


//---------------------------------------------------------------------------------------------------------------
// Get size of vertical systematic weights OpenData
// Keep the same order used in runSelection.py
//---------------------------------------------------------------------------------------------------------------
void HEPHero::VerticalSysSizesOD( ){
    if( (_sysID_lateral == 0) && (dataset_group != "Data") ) {
        sys_vertical_size.clear();
        _inputTree->GetEntry(0);

        get_Pileup_sfs = false;

        for( int ivert = 0; ivert < _sysNames_vertical.size(); ++ivert ){
            string sysName = _sysNames_vertical.at(ivert);
            if( sysName == "Pileup" ){
                sys_vertical_size.push_back(2);
                get_Pileup_sfs = true;
            }
        }
    }
}


//---------------------------------------------------------------------------------------------------------------
// Vertical systematics OpenData
// Keep the same order used in runSelection.py
//---------------------------------------------------------------------------------------------------------------
void HEPHero::VerticalSysOD(){
    if( (_sysID_lateral == 0) && (dataset_group != "Data") ) {
        sys_vertical_sfs.clear();

        //-----------------------------------------------------------------------------------
        if( get_Pileup_sfs ){
            vector<float> Pileup_sfs;
            double pileup_wgt_down = GetPileupWeightOD("down");
            double pileup_wgt_up = GetPileupWeightOD("up");
            Pileup_sfs.push_back(pileup_wgt_down/pileup_wgt);
            Pileup_sfs.push_back(pileup_wgt_up/pileup_wgt);
            sys_vertical_sfs.insert(pair<string, vector<float>>("Pileup", Pileup_sfs));
        }
    }
}

