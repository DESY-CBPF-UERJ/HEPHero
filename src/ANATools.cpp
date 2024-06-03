#include "HEPHero.h"
#include "lester_mt2_bisect.h"
#include "ttbarReco_sonnenschein_opendata.h"
#include "ttbarReco_sonnenschein_desy.h"
#include "XYMETCorrection_withUL17andUL18andUL16.h"

//---------------------------------------------------------------------------------------------------------------
// Regions
//---------------------------------------------------------------------------------------------------------------
void HEPHero::Regions(){
    RegionID = -1;
        
    if( (RecoLepID < 100) && 
        (ttbar_reco == 0) &&
        (Nbjets >= 1) //&&
        //((dataset_group != "Data") || (MLP4_score_torch < 0.22))
      ){                                        // [Signal Region]
        RegionID = 0;
    }
    else if( (RecoLepID < 100) && 
        (Nbjets == 0) &&
        (MET_pt < 140)
      ){                                        // [DY - Control Region]
        RegionID = 1;
    }
    else if( (RecoLepID > 1000) && (RecoLepID < 1999) &&
        (Nbjets >= 1) 
      ){                                        // [ttbar - Control Region]
        RegionID = 2;
    }
    else if( (RecoLepID > 30000) && (RecoLepID < 39999) &&
        (ttbar_reco == 0) &&
        (Nbjets >= 1) &&
        (LepLep_deltaM < 10) &&
        (MET_Lep3_Mt > 50) //&& (MET_Lep3_Mt < 100)
      ){                                        // [WZ - Control Region]
        RegionID = 3;
    }
    else if( (RecoLepID > 40000) && (RecoLepID < 49999) &&
        (ttbar_reco == 0) &&
        (Nbjets >= 1) &&
        (Lep3Lep4_M > 10)
      ){                                        // [ZZ - Control Region] 
        RegionID = 4;
    }
    else if( (RecoLepID < 100) &&
        (Nbjets == 0) &&
        (MET_pt < 300)
      ){                                        // [DY Recoil - Control Region]
        RegionID = 5;
    }
    else if( (RecoLepID > 30000) && (RecoLepID < 39999) &&
        (ttbar_reco == 0) &&
        (Nbjets == 0) &&
        (LepLep_deltaM < 10) &&
        (MET_Lep3_Mt > 50) //&& (MET_Lep3_Mt < 100)
      ){                                        // [WZ 0 b-jets - Control Region]
        RegionID = 6;
    }
    else if( (RecoLepID > 40000) && (RecoLepID < 49999) &&
        (ttbar_reco == 0) &&
        (Nbjets == 0) &&
        (Lep3Lep4_M > 10)
      ){                                        // [ZZ 0 b-jets - Control Region]
        RegionID = 7;
    }
}


//---------------------------------------------------------------------------------------------------------------
// MLP Model for signal discrimination
//---------------------------------------------------------------------------------------------------------------
void HEPHero::Signal_discriminators(){
    
    float floatC = 1.;
    
    // https://github.com/Dobiasd/frugally-deep
    //MLP_score_keras = MLP_keras.predict({LeadingLep_pt, LepLep_pt, LepLep_deltaR, LepLep_deltaM, MET_pt, MET_LepLep_Mt, MET_LepLep_deltaPhi});

    MLP_score_torch = MLP_torch.predict({LeadingLep_pt, LepLep_pt, LepLep_deltaR, LepLep_deltaM, MET_pt, MET_LepLep_Mt, MET_LepLep_deltaPhi, TrailingLep_pt, MT2LL});//, Nbjets*floatC}); backup

    //MLP_score_torch = MLP_torch.predict({LeadingLep_pt, TrailingLep_pt, LepLep_pt, LepLep_deltaR, LepLep_deltaM, MET_pt, MET_LepLep_Mt, MET_LepLep_deltaPhi, MT2LL});//, Nbjets*floatC});

    //MLP_score_torch = MLP_torch.predict({LeadingLep_pt, TrailingLep_pt, LepLep_pt, LepLep_deltaR, LepLep_deltaM, MET_pt, MET_LepLep_Mt, MET_LepLep_deltaPhi, MT2LL, Dijet_deltaEta, Dijet_pt, Dijet_M});//, Nbjets*floatC});

    MLP4_score_torch = pow(1.e4,MLP_score_torch)/1.e4;

}


//---------------------------------------------------------------------------------------------------------------
// Jet lep overlap
//---------------------------------------------------------------------------------------------------------------
void HEPHero::Jet_lep_overlap( float deltaR_cut ){
    
    Jet_LepOverlap.clear();
    for( unsigned int ijet = 0; ijet < nJet; ++ijet ) {
        Jet_LepOverlap.push_back(false);
    }
    
    //for( unsigned int iele = 0; iele < nElectron; ++iele ) {
    for( unsigned int iselEle = 0; iselEle < selectedEle.size(); ++iselEle ) {
        int iele = selectedEle.at(iselEle);
        float drMin = 99999.;
        int JetID = -1;
        for( unsigned int ijet = 0; ijet < nJet; ++ijet ) {
            double deta = fabs(Electron_eta[iele] - Jet_eta[ijet]);
            double dphi = fabs(Electron_phi[iele] - Jet_phi[ijet]);
            if( dphi > M_PI ) dphi = 2*M_PI - dphi;
            double dr = sqrt( deta*deta + dphi*dphi );
            if( dr < drMin ){
                drMin = dr;
                JetID = ijet;
            }
        }
        if( (drMin < deltaR_cut) && (JetID >= 0) ){ 
            //cout << "deltaR " << drMin << " " << endl;
            //cout << "JetID " << JetID << " " << endl;
            Jet_LepOverlap[JetID] = true;
        }
    }
    
    //for( unsigned int imu = 0; imu < nMuon; ++imu ) {
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
        if( (drMin < deltaR_cut) && (JetID >= 0) ){ 
            //cout << "deltaR " << drMin << " " << endl;
            //cout << "JetID " << JetID << " " << endl;
            Jet_LepOverlap[JetID] = true;
        }
    }
    
}


//---------------------------------------------------------------------------------------------------------------
// Jet selection
//---------------------------------------------------------------------------------------------------------------
void HEPHero::JetSelection(){

    selectedJet.clear();

    for( unsigned int ijet = 0; ijet < nJet; ++ijet ) {
        Jet_JES_pt[ijet] = Jet_pt[ijet];
    }

    JESvariation();
    JERvariation();

    Jet_lep_overlap( JET_LEP_DR_ISO_CUT );

    Nbjets = 0;
    Nbjets30 = 0;
    Nbjets_LepIso04 = 0;
    Nbjets30_LepIso04 = 0;
    Njets = 0;
    Njets30 = 0;
    Njets40 = 0;
    Njets_LepIso04 = 0;
    Njets30_LepIso04 = 0;
    Njets40_LepIso04 = 0;
    Njets_forward = 0;
    Njets_tight = 0;
    Njets_ISR = 0;
    NPUjets = 0;
    Jet_abseta_max = 0;
    HT = 0;
    HT30 = 0;
    HT40 = 0;
    float HPx = 0;
    float HPy = 0;
    float HPx30 = 0;
    float HPy30 = 0;
    float HPx40 = 0;
    float HPy40 = 0;
    float HPx_trig = 0;
    float HPy_trig = 0;
    for( unsigned int ijet = 0; ijet < nJet; ++ijet ) {

        if( Jet_pt[ijet] <= JET_PT_CUT ) continue;
        if( Jet_jetId[ijet] >= 2 ){
            TLorentzVector Jet_trig;
            Jet_trig.SetPtEtaPhiE(Jet_pt[ijet], Jet_eta[ijet], Jet_phi[ijet], 0);
            HPx_trig += Jet_trig.Px();
            HPy_trig += Jet_trig.Py();
        }
        if( (abs(Jet_eta[ijet]) < JET_ETA_CUT) and (Jet_jetId[ijet] >= 2) ) Njets_tight += 1;
        if( Jet_jetId[ijet] < JET_ID_WP ) continue;

        //if( Jet_lep_overlap( ijet, JET_LEP_DR_ISO_CUT ) ) continue;
        if( Jet_LepOverlap[ijet] ) continue;
        if( (Jet_pt[ijet] < 50) && (Jet_puId[ijet] < JET_PUID_WP) ) continue;

        if( abs(Jet_eta[ijet]) >= 5.0 ) continue;
        if( abs(Jet_eta[ijet]) > 1.4 ) Njets_forward += 1;
        if( abs(Jet_eta[ijet]) > Jet_abseta_max ) Jet_abseta_max = abs(Jet_eta[ijet]);
        if( abs(Jet_eta[ijet]) >= JET_ETA_CUT ) continue;
        selectedJet.push_back(ijet);
        TLorentzVector Jet;
        Jet.SetPtEtaPhiE(Jet_pt[ijet], Jet_eta[ijet], Jet_phi[ijet], 0);

        Njets += 1;
        //if( !Jet_lep_overlap(ijet, 0.4) ) Njets_LepIso04 += 1;
        if( PileupJet( ijet ) ) NPUjets += 1;
        HT += Jet_pt[ijet];
        HPx += Jet.Px();
        HPy += Jet.Py();
        if( Jet_pt[ijet] > 30 ){
            Njets30 += 1;
            //if( !Jet_lep_overlap(ijet, 0.4) ) Njets30_LepIso04 += 1;
            HT30 += Jet_pt[ijet];
            HPx30 += Jet.Px();
            HPy30 += Jet.Py();
        }
        if( Jet_pt[ijet] > 40 ){
            Njets40 += 1;
            //if( !Jet_lep_overlap(ijet, 0.4) ) Njets40_LepIso04 += 1;
            HT40 += Jet_pt[ijet];
            HPx40 += Jet.Px();
            HPy40 += Jet.Py();
        }
        if( JetBTAG( ijet, JET_BTAG_WP ) ){
            Nbjets += 1;
            if( Jet_pt[ijet] > 30 ) Nbjets30 += 1;
            //if( !Jet_lep_overlap(ijet, 0.4) ){
            //    Nbjets_LepIso04 += 1;
            //    if( Jet_pt[ijet] > 30 ) Nbjets30_LepIso04 += 1;
            //}
        }
        if( Jet_pt[ijet] > 26 ) Njets_ISR += 1;
    }
    MHT = sqrt(HPx*HPx + HPy*HPy);
    MHT30 = sqrt(HPx30*HPx30 + HPy30*HPy30);
    MHT40 = sqrt(HPx40*HPx40 + HPy40*HPy40);
    MHT_trig = sqrt(HPx_trig*HPx_trig + HPy_trig*HPy_trig);

    MDT = abs(MHT_trig - MET_pt);

    LeadingJet_pt = 0;
    SubLeadingJet_pt = 0;
    ThirdLeadingJet_pt = 0;
    FourthLeadingJet_pt = 0;
    if( Njets >= 1 ) LeadingJet_pt = Jet_pt[selectedJet.at(0)];
    if( Njets >= 2 ) SubLeadingJet_pt = Jet_pt[selectedJet.at(1)];
    if( Njets >= 3 ) ThirdLeadingJet_pt = Jet_pt[selectedJet.at(2)];
    if( Njets >= 4 ) FourthLeadingJet_pt = Jet_pt[selectedJet.at(3)];
    
}


//---------------------------------------------------------------------------------------------------------------
// Jet angular variables
//---------------------------------------------------------------------------------------------------------------
void HEPHero::Get_Jet_Angular_Variables( int pt_cut ){
    
    if( (pt_cut != 20) && (pt_cut != 30) && (pt_cut != 40) ){
        cout << "Sorry, for angular variables the only cuts acceptable are 20, 30, or 40. Let's consider your cut equal to 20 GeV!" << endl;
        pt_cut = 20;
    }

    double HPx = 0;
    double HPy = 0;
    for( unsigned int iselJet = 0; iselJet < selectedJet.size(); ++iselJet ) {
        int iJet = selectedJet.at(iselJet);
        if( Jet_pt[iJet] < pt_cut ) continue;
        TLorentzVector Jet;
        Jet.SetPtEtaPhiE(Jet_pt[iJet], Jet_eta[iJet], Jet_phi[iJet], 0);
        HPx += Jet.Px();
        HPy += Jet.Py();
    }
    double MHT_i = sqrt(HPx*HPx + HPy*HPy);
  
    float omegaMin = 999999;
    float chiMin = 999999;
    float fMax = 0;
    for( unsigned int iselJet = 0; iselJet < selectedJet.size(); ++iselJet ) {
        int iJet = selectedJet.at(iselJet);
        if( Jet_pt[iJet] < pt_cut ) continue;
        
        double dPhi_i = abs( Jet_phi[iJet] - atan2(-HPy,-HPx) );
        if( dPhi_i > M_PI ) dPhi_i  = 2*M_PI-dPhi_i ;
        
        double f_i = Jet_pt[iJet]/MHT_i;
        if( f_i > fMax ) fMax = f_i;
        
        double M1 = dPhi_i;
        if( dPhi_i > M_PI/2 ) M1  = M_PI/2 ;
        
        double M2 = f_i;
        if( f_i > -cos(dPhi_i) ) M2 = -cos(dPhi_i);
        
        double MAX = f_i + cos(dPhi_i);
        if( MAX < 0 ) MAX = 0;
        
        double M3 = f_i;
        if( M3 > MAX ) M3 = MAX;
        
        double omega_i = atan2(sin(M1),f_i);
        if( omega_i < omegaMin ) omegaMin = omega_i;
        
        double chi_i = atan2(sqrt(1+M2*M2+2*M2*cos(dPhi_i)),M3);
        if( chi_i < chiMin ) chiMin = chi_i;
        
    }
    if( omegaMin == 999999 ) omegaMin = -1;
    if( chiMin == 999999 ) chiMin = -1;
    
    if( pt_cut == 20 ){
        OmegaMin = omegaMin;
        ChiMin = chiMin;
        FMax = fMax;
    }else if( pt_cut == 30 ){
        OmegaMin30 = omegaMin;
        ChiMin30 = chiMin;
        FMax30 = fMax;
    }else if( pt_cut == 40 ){
        OmegaMin40 = omegaMin;
        ChiMin40 = chiMin;
        FMax40 = fMax;
    }
    
}


//---------------------------------------------------------------------------------------------------------------
// Fit function for MET recoil correction 
//---------------------------------------------------------------------------------------------------------------
float HEPHero::Recoil_linears( float x, float a1, float b1, float a2, float c ){
    float out;
    if( x <= c){ 
        out = a1*x + b1;
    }else{
        out = a2*(x-c) + a1*c + b1;
    }
    return out;
}


//---------------------------------------------------------------------------------------------------------------
// MET Correction
//--------------------------------------------------------------------------------------------------------------- 
void HEPHero::METCorrection(){
    
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
    
    //=====MET XY Correction=======================================================================
    if( apply_met_xy_corr ){
        string year;
        if( dataset_year == "16" ){ 
            if( dataset_HIPM ){
                year = "2016APV";
            }else{
                year = "2016nonAPV";
            }
        }else if( dataset_year == "17" ){
            year = "2017";
        }else if( dataset_year == "18" ){
            year = "2018";
        }
        bool isMC = (dataset_group != "Data");
        bool isUL = true;
        bool ispuppi = false;
        
        std::pair <double,double> CorrMET;
        CorrMET = METXYCorr_Met_MetPhi(MET_pt, MET_phi, run, year, isMC, PV_npvs, isUL, ispuppi);
        MET_pt = CorrMET.first;
        MET_phi =  CorrMET.second;
        
        MET_XY_pt = MET_pt;
        MET_XY_phi = MET_phi;
    }
    
    //=====Unclust Energy Variation================================================================
    if( (_sysName_lateral == "UncMET") && (dataset_group != "Data") && !(apply_met_recoil_corr && (dsNameDY == "DYJetsToLL")) ){
    //if( (_sysName_lateral == "UncMET") && (dataset_group != "Data") ){
            
        double CorrectedMET_x;
        double CorrectedMET_y;
        if( _Universe == 0 ){    
            CorrectedMET_x = MET_pt*cos(MET_phi) - MET_MetUnclustEnUpDeltaX;
            CorrectedMET_y = MET_pt*sin(MET_phi) - MET_MetUnclustEnUpDeltaY;    
        }else if( _Universe == 1 ){
            CorrectedMET_x = MET_pt*cos(MET_phi) + MET_MetUnclustEnUpDeltaX;
            CorrectedMET_y = MET_pt*sin(MET_phi) + MET_MetUnclustEnUpDeltaY;
        }
        
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
    
    MET_Unc_pt = MET_pt;
    MET_Unc_phi = MET_phi;
    
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
    
    //=====Recoil Correction=======================================================================
    //if( apply_met_recoil_corr && ((dsNameDY == "DYJetsToLL") || (dsNameZZ == "ZZTo2L") || (dsNameZZ == "ZZTo4L")) ){
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
        
        if( (dataset_year == "16") && dataset_HIPM ){
            sigma_ratio = 1.039; 
            sigma_ratio_unc_up = 0.006; 
            sigma_ratio_unc_down = 0.006;
      
            if( Njets_ISR == 0 ){
                mc_u1_mean = {-40.187, -49.903, -60.063, -70.695, -81.461, -92.655, -103.001, -113.934, -124.553, -134.528, -144.684, -155.054}; 
                mc_u1_mean_unc_up = {0.06, 0.067, 0.074, 0.067, 0.092, 0.079, 0.036, 0.032, 0.008, 0.033, 0.025, 0.021}; 
                mc_u1_mean_unc_down = {0.022, 0.033, 0.042, 0.04, 0.047, 0.019, 0.026, 0.009, 0.008, 0.018, 0.036, 0.091};
                
                diff_u1 = {1.432, 1.407, 1.528, 1.656, 1.693, 1.587, 1.327, 0.946, 0.52, 0.168, 0.053, 0.381}; 
                diff_u1_unc_up = {0.041, 0.078, 0.054, 0.132, 0.216, 0.164, 0.158, 0.117, 0.39, 0.284, 0.19, 0.061}; 
                diff_u1_unc_down = {0.038, 0.061, 0.083, 0.101, 0.112, 0.115, 0.109, 0.097, 0.08, 0.06, 0.045, 0.771};
                
                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160};
            }else if( Njets_ISR == 1 ){
                mc_u1_mean = {-45.007, -54.486, -64.219, -74.04, -83.916, -94.089, -104.123, -114.264, -124.563, -134.709, -145.09, -155.338, -165.367, -175.702, -185.887, -195.533, -206.004, -216.435, -226.449, -236.638, -247.27, -257.119, -267.692, -277.359, -287.922}; 
                mc_u1_mean_unc_up = {0.001, 0.005, 0.004, 0.022, 0.012, 0.016, 0.013, 0.014, 0.006, 0.01, 0.006, 0.016, 0.019, 0.029, 0.015, 0.011, 0.04, 0.027, 0.012, 0.041, 0.048, 0.09, 0.067, 0.033, 0.033}; 
                mc_u1_mean_unc_down = {0.002, 0.002, 0.002, 0.005, 0.009, 0.01, 0.005, 0.009, 0.023, 0.001, 0.039, 0.044, 0.001, 0.051, 0.012, 0.079, 0.107, 0.03, 0.077, 0.05, 0.092, 0.149, 0.097, 0.111, 0.103};
                
                diff_u1 = {0.302, 0.488, 0.635, 0.748, 0.829, 0.884, 0.917, 0.931, 0.931, 0.922, 0.907, 0.89, 0.876, 0.869, 0.874, 0.893, 0.932, 0.995, 1.086, 1.209, 1.368, 1.567, 1.811, 2.104, 2.45}; 
                diff_u1_unc_up = {0.004, 0.017, 0.022, 0.027, 0.026, 0.062, 0.045, 0.059, 0.069, 0.088, 0.077, 0.042, 0.128, 0.09, 0.096, 0.02, 0.019, 0.02, 0.023, 0.027, 0.032, 0.04, 0.051, 0.063, 0.078}; 
                diff_u1_unc_down = {0.003, 0.014, 0.025, 0.035, 0.042, 0.048, 0.052, 0.055, 0.057, 0.057, 0.056, 0.054, 0.052, 0.048, 0.044, 0.07, 0.115, 0.18, 0.224, 0.206, 0.214, 0.205, 0.257, 0.288, 0.337};
                
                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290};
            }else if( Njets_ISR >= 2 ){
                mc_u1_mean = {-44.964, -55.17, -65.229, -75.05, -84.943, -95.047, -104.974, -114.823, -124.775, -134.917, -144.973, -154.917, -164.986, -175.353, -185.31, -195.742, -205.495, -215.501, -225.637, -235.972, -245.58, -256.391, -266.245, -276.621, -286.843, -296.505}; 
                mc_u1_mean_unc_up = {0.01, 0.009, 0.002, 0.003, 0.006, 0.026, 0.046, 0.012, 0.051, 0.017, 0.002, 0.03, 0.028, 0.048, 0.012, 0.021, 0.035, 0.083, 0.035, 0.037, 0.041, 0.041, 0.056, 0.06, 0.048, 0.053}; 
                mc_u1_mean_unc_down = {0.006, 0.01, 0.005, 0.007, 0.008, 0.009, 0.011, 0.007, 0.002, 0.002, 0.005, 0.001, 0.0, 0.013, 0.015, 0.026, 0.027, 0.037, 0.012, 0.038, 0.019, 0.103, 0.101, 0.18, 0.159, 0.082};
        
                diff_u1 = {0.144, 0.225, 0.302, 0.376, 0.446, 0.512, 0.574, 0.633, 0.688, 0.739, 0.786, 0.83, 0.87, 0.906, 0.939, 0.967, 0.992, 1.014, 1.031, 1.045, 1.055, 1.061, 1.064, 1.063, 1.058, 1.049}; 
                diff_u1_unc_up = {0.053, 0.05, 0.045, 0.071, 0.072, 0.033, 0.035, 0.036, 0.036, 0.036, 0.035, 0.033, 0.032, 0.044, 0.061, 0.077, 0.089, 0.094, 0.091, 0.08, 0.064, 0.058, 0.066, 0.073, 0.082, 0.09}; 
                diff_u1_unc_down = {0.004, 0.032, 0.075, 0.104, 0.121, 0.15, 0.161, 0.15, 0.126, 0.109, 0.107, 0.119, 0.106, 0.11, 0.13, 0.108, 0.181, 0.116, 0.132, 0.241, 0.231, 0.187, 0.259, 0.295, 0.438, 0.653};
                
                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300};
            }
        }else if( (dataset_year == "16") && !dataset_HIPM ){
            sigma_ratio = 1.049; 
            sigma_ratio_unc_up = 0.01; 
            sigma_ratio_unc_down = 0.005;                                                                                                                                                   
                                                                                                                                                                                            
            if( Njets_ISR == 0 ){
                mc_u1_mean = {-40.695, -50.426, -60.686, -71.246, -82.037, -92.929, -103.167, -113.714, -124.391}; 
                mc_u1_mean_unc_up = {0.049, 0.072, 0.062, 0.08, 0.057, 0.029, 0.008, 0.031, 0.037}; 
                mc_u1_mean_unc_down = {0.016, 0.019, 0.019, 0.02, 0.016, 0.012, 0.023, 0.013, 0.003};
                
                diff_u1 = {0.738, 0.697, 0.775, 0.914, 1.055, 1.14, 1.11, 0.907, 0.472}; 
                diff_u1_unc_up = {0.037, 0.127, 0.099, 0.05, 0.054, 0.053, 0.048, 0.206, 0.91}; 
                diff_u1_unc_down = {0.034, 0.059, 0.081, 0.119, 0.23, 0.254, 0.334, 0.093, 0.059};
                
                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130};
            }else if( Njets_ISR == 1 ){
                mc_u1_mean = {-45.345, -54.845, -64.503, -74.243, -84.119, -94.11, -104.359, -114.371, -124.479, -134.765, -144.927, -155.156, -165.375, -175.521, -185.656, -195.95, -206.013, -216.341, -226.424, -236.578, -246.575, -256.437, -267.205}; 
                mc_u1_mean_unc_up = {0.003, 0.002, 0.002, 0.004, 0.006, 0.011, 0.01, 0.002, 0.002, 0.002, 0.008, 0.005, 0.007, 0.007, 0.004, 0.023, 0.02, 0.017, 0.008, 0.023, 0.022, 0.038, 0.042}; 
                mc_u1_mean_unc_down = {0.004, 0.008, 0.001, 0.006, 0.006, 0.002, 0.007, 0.002, 0.014, 0.019, 0.03, 0.035, 0.023, 0.009, 0.113, 0.057, 0.079, 0.036, 0.046, 0.127, 0.084, 0.141, 0.121};
                
                diff_u1 = {0.197, 0.301, 0.384, 0.45, 0.501, 0.54, 0.569, 0.591, 0.609, 0.626, 0.645, 0.667, 0.696, 0.735, 0.786, 0.852, 0.935, 1.039, 1.166, 1.319, 1.501, 1.714, 1.961}; 
                diff_u1_unc_up = {0.022, 0.007, 0.011, 0.028, 0.048, 0.052, 0.069, 0.112, 0.094, 0.096, 0.091, 0.052, 0.119, 0.024, 0.023, 0.023, 0.023, 0.024, 0.205, 0.275, 0.332, 0.607, 1.007}; 
                diff_u1_unc_down = {0.003, 0.024, 0.029, 0.032, 0.039, 0.045, 0.049, 0.052, 0.054, 0.054, 0.053, 0.051, 0.048, 0.114, 0.243, 0.385, 0.135, 0.195, 0.019, 0.018, 0.02, 0.024, 0.027};
                
                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270};
            }else if( Njets_ISR >= 2 ){
                mc_u1_mean = {-45.229, -55.225, -65.321, -75.267, -85.205, -95.159, -105.218, -114.877, -125.062, -135.17, -144.999, -155.107, -165.15, -175.022, -185.472, -195.444, -205.297, -215.151, -225.278, -235.805, -245.752, -256.145, -265.924, -276.019, -286.297, -296.471}; 
                mc_u1_mean_unc_up = {0.022, 0.027, 0.004, 0.02, 0.007, 0.005, 0.006, 0.007, 0.055, 0.002, 0.015, 0.054, 0.039, 0.019, 0.048, 0.011, 0.038, 0.002, 0.012, 0.012, 0.011, 0.01, 0.019, 0.017, 0.017, 0.02};
                mc_u1_mean_unc_down = {0.008, 0.006, 0.006, 0.004, 0.011, 0.005, 0.011, 0.005, 0.006, 0.014, 0.001, 0.008, 0.007, 0.001, 0.024, 0.003, 0.016, 0.039, 0.049, 0.088, 0.037, 0.095, 0.086, 0.074, 0.064, 0.044};
                
                diff_u1 = {0.039, 0.128, 0.2, 0.255, 0.296, 0.324, 0.341, 0.348, 0.347, 0.34, 0.329, 0.314, 0.297, 0.281, 0.266, 0.255, 0.249, 0.25, 0.259, 0.277, 0.307, 0.351, 0.409, 0.483, 0.576, 0.688}; 
                diff_u1_unc_up = {0.143, 0.056, 0.025, 0.032, 0.036, 0.124, 0.102, 0.115, 0.16, 0.159, 0.138, 0.176, 0.135, 0.098, 0.174, 0.216, 0.108, 0.124, 0.131, 0.129, 0.115, 0.088, 0.312, 0.285, 0.569, 1.192}; 
                diff_u1_unc_down = {0.0, 0.016, 0.086, 0.161, 0.145, 0.146, 0.146, 0.137, 0.12, 0.097, 0.069, 0.038, 0.005, 0.001, 0.022, 0.054, 0.192, 0.295, 0.309, 0.518, 0.245, 0.314, 0.035, 0.017, 0.09, 0.185};
                
                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300};
            }
        }else if( dataset_year == "17" ){
            sigma_ratio = 1.035; 
            sigma_ratio_unc_up = 0.008; 
            sigma_ratio_unc_down = 0.006;
                                                                                                                                                                                    
            if( Njets_ISR == 0 ){
                mc_u1_mean = {-39.38, -49.063, -59.216, -69.701, -80.247, -91.053, -101.6, -112.28, -122.362, -132.611, -143.415, -153.402, -162.914, -173.52}; mc_u1_mean_unc_up = {0.149, 0.187, 0.215, 0.186, 0.181, 0.21, 0.154, 0.119, 0.103, 0.107, 0.052, 0.086, 0.009, 0.025}; 
                mc_u1_mean_unc_down = {0.022, 0.026, 0.025, 0.025, 0.022, 0.011, 0.015, 0.009, 0.004, 0.01, 0.001, 0.016, 0.08, 0.002};
                
                diff_u1 = {1.467, 1.631, 1.731, 1.769, 1.745, 1.658, 1.509, 1.297, 1.023, 0.686, 0.286, -0.175, -0.7, -1.287}; 
                diff_u1_unc_up = {0.12, 0.114, 0.146, 0.22, 0.251, 0.318, 0.358, 0.263, 0.041, 0.035, 0.026, 0.016, 0.715, 1.547}; 
                diff_u1_unc_down = {0.03, 0.056, 0.077, 0.093, 0.104, 0.11, 0.11, 0.106, 0.278, 0.409, 0.284, 0.182, 0.007, 0.015};
                
                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180};
            }else if( Njets_ISR == 1 ){
                mc_u1_mean = {-44.236, -53.715, -63.34, -73.045, -83.05, -92.978, -103.04, -113.211, -123.481, -133.769, -143.993, -154.276, -164.349, -174.52, -184.81, -194.68, -204.913, -215.205, -225.377, -235.421, -245.836, -255.989, -266.64, -276.775, -286.41, -296.728}; 
                mc_u1_mean_unc_up = {0.032, 0.026, 0.051, 0.063, 0.07, 0.073, 0.087, 0.071, 0.056, 0.039, 0.068, 0.052, 0.005, 0.004, 0.092, 0.058, 0.024, 0.002, 0.01, 0.009, 0.023, 0.036, 0.036, 0.022, 0.015, 0.016}; 
                mc_u1_mean_unc_down = {0.002, 0.003, 0.006, 0.008, 0.008, 0.008, 0.005, 0.004, 0.004, 0.003, 0.001, 0.005, 0.001, 0.05, 0.004, 0.006, 0.002, 0.073, 0.024, 0.042, 0.133, 0.188, 0.153, 0.139, 0.076, 0.114};
                
                diff_u1 = {0.42, 0.539, 0.618, 0.663, 0.677, 0.665, 0.63, 0.578, 0.511, 0.436, 0.354, 0.272, 0.192, 0.12, 0.06, 0.015, -0.01, -0.011, 0.016, 0.076, 0.173, 0.311, 0.494, 0.726, 1.012, 1.356}; 
                diff_u1_unc_up = {0.025, 0.044, 0.067, 0.06, 0.069, 0.064, 0.07, 0.086, 0.05, 0.064, 0.046, 0.116, 0.11, 0.018, 0.016, 0.014, 0.012, 0.011, 0.28, 0.14, 0.259, 0.197, 0.278, 0.225, 0.297, 0.31}; 
                diff_u1_unc_down = {0.003, 0.012, 0.023, 0.031, 0.038, 0.044, 0.048, 0.05, 0.052, 0.052, 0.051, 0.049, 0.047, 0.079, 0.164, 0.182, 0.121, 0.156, 0.02, 0.015, 0.01, 0.009, 0.011, 0.014, 0.017, 0.021};
                
                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300};
            }else if( Njets_ISR >= 2 ){
                mc_u1_mean = {-44.396, -54.455, -64.267, -74.121, -83.969, -93.944, -103.683, -113.741, -123.568, -133.66, -143.596, -153.805, -163.803, -173.843, -183.818, -193.825, -204.302, -214.293, -224.232, -234.378, -244.531, -254.509, -264.85, -275.295, -285.343, -294.813}; 
                mc_u1_mean_unc_up = {0.057, 0.032, 0.049, 0.067, 0.058, 0.099, 0.062, 0.04, 0.1, 0.122, 0.071, 0.138, 0.111, 0.12, 0.06, 0.043, 0.019, 0.05, 0.004, 0.002, 0.046, 0.024, 0.011, 0.026, 0.107, 0.053}; 
                mc_u1_mean_unc_down = {0.004, 0.006, 0.005, 0.005, 0.007, 0.008, 0.009, 0.006, 0.006, 0.005, 0.008, 0.002, 0.004, 0.007, 0.002, 0.009, 0.0, 0.003, 0.032, 0.048, 0.001, 0.027, 0.022, 0.024, 0.024, 0.011};

                diff_u1 = {0.257, 0.309, 0.349, 0.379, 0.399, 0.41, 0.413, 0.409, 0.399, 0.384, 0.365, 0.343, 0.318, 0.293, 0.267, 0.242, 0.219, 0.198, 0.181, 0.168, 0.161, 0.161, 0.168, 0.183, 0.208, 0.243}; 
                diff_u1_unc_up = {0.007, 0.016, 0.07, 0.093, 0.089, 0.146, 0.067, 0.111, 0.069, 0.08, 0.095, 0.106, 0.014, 0.009, 0.018, 0.033, 0.044, 0.052, 0.054, 0.329, 0.166, 0.123, 0.18, 0.284, 0.191, 0.442}; 
                diff_u1_unc_down = {0.047, 0.087, 0.065, 0.086, 0.099, 0.105, 0.104, 0.097, 0.086, 0.072, 0.055, 0.037, 0.098, 0.143, 0.185, 0.198, 0.118, 0.155, 0.149, 0.016, 0.016, 0.015, 0.016, 0.049, 0.097, 0.159};
                
                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300};
            }
        }else if( dataset_year == "18" ){
            sigma_ratio = 1.037;
            sigma_ratio_unc_up = 0.008; 
            sigma_ratio_unc_down = 0.006;
            
            if( Njets_ISR == 0 ){
                mc_u1_mean = {-40.094, -49.742, -59.823, -70.243, -80.942, -91.892, -102.685, -113.354, -123.591, -134.115, -144.537, -154.196, -165.018, -175.977, -184.689, -196.282, -206.028}; 
                mc_u1_mean_unc_up = {0.136, 0.152, 0.165, 0.158, 0.185, 0.133, 0.149, 0.028, 0.02, 0.003, 0.101, 0.002, 0.007, 0.003, 0.004, 0.023, 0.279}; 
                mc_u1_mean_unc_down = {0.02, 0.022, 0.021, 0.02, 0.02, 0.011, 0.004, 0.002, 0.003, 0.096, 0.0, 0.091, 0.071, 0.187, 0.082, 0.246, 0.044};
                
                diff_u1 = {1.737, 1.946, 2.194, 2.441, 2.653, 2.807, 2.89, 2.899, 2.836, 2.718, 2.568, 2.418, 2.312, 2.3, 2.443, 2.812, 3.487}; 
                diff_u1_unc_up = {0.091, 0.187, 0.176, 0.158, 0.164, 0.156, 0.181, 0.319, 0.361, 0.449, 0.704, 0.719, 0.787, 0.841, 0.373, 0.07, 0.113}; 
                diff_u1_unc_down = {0.043, 0.019, 0.058, 0.123, 0.181, 0.209, 0.193, 0.129, 0.027, 0.016, 0.014, 0.012, 0.015, 0.021, 0.029, 1.034, 2.479};
                
                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210};
            }else if( Njets_ISR == 1 ){
                mc_u1_mean = {-44.953, -54.334, -63.927, -73.544, -83.374, -93.281, -103.345, -113.39, -123.561, -133.788, -143.845, -154.035, -164.158, -174.4, -184.788, -194.632, -204.773, -215.331, -225.39, -235.224, -245.419, -255.472, -265.771, -276.043, -286.099, -296.033}; 
                mc_u1_mean_unc_up = {0.001, 0.012, 0.032, 0.034, 0.063, 0.064, 0.069, 0.05, 0.044, 0.061, 0.043, 0.04, 0.018, 0.005, 0.032, 0.068, 0.002, 0.038, 0.009, 0.008, 0.013, 0.028, 0.033, 0.025, 0.02, 0.014}; 
                mc_u1_mean_unc_down = {0.002, 0.001, 0.002, 0.004, 0.006, 0.005, 0.002, 0.003, 0.004, 0.002, 0.002, 0.003, 0.002, 0.031, 0.014, 0.003, 0.03, 0.002, 0.029, 0.131, 0.161, 0.209, 0.266, 0.175, 0.14, 0.143};
                
                diff_u1 = {0.762, 0.892, 0.999, 1.084, 1.149, 1.198, 1.231, 1.251, 1.261, 1.263, 1.258, 1.25, 1.239, 1.229, 1.222, 1.22, 1.225, 1.239, 1.264, 1.304, 1.359, 1.432, 1.526, 1.642, 1.783, 1.951}; 
                diff_u1_unc_up = {0.033, 0.046, 0.067, 0.073, 0.076, 0.083, 0.085, 0.094, 0.1, 0.131, 0.118, 0.087, 0.123, 0.104, 0.131, 0.134, 0.134, 0.147, 0.21, 0.175, 0.229, 0.251, 0.183, 0.286, 0.284, 0.233}; 
                diff_u1_unc_down = {0.003, 0.012, 0.022, 0.031, 0.038, 0.043, 0.046, 0.049, 0.049, 0.049, 0.048, 0.045, 0.042, 0.039, 0.034, 0.029, 0.025, 0.02, 0.015, 0.013, 0.013, 0.015, 0.018, 0.021, 0.025, 0.029};
                
                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300};
            }else if( Njets_ISR >= 2 ){
                mc_u1_mean = {-44.82, -54.739, -64.834, -74.552, -84.487, -94.379, -104.128, -114.154, -123.833, -133.96, -143.752, -153.859, -163.777, -174.378, -183.954, -194.149, -204.031, -214.157, -224.314, -234.362, -244.865, -254.675, -264.781, -274.747, -284.82, -294.685}; 
                mc_u1_mean_unc_up = {0.015, 0.03, 0.067, 0.033, 0.043, 0.059, 0.052, 0.085, 0.085, 0.093, 0.118, 0.094, 0.075, 0.087, 0.132, 0.075, 0.007, 0.088, 0.204, 0.023, 0.014, 0.014, 0.018, 0.007, 0.013, 0.01}; 
                mc_u1_mean_unc_down = {0.005, 0.007, 0.006, 0.006, 0.006, 0.009, 0.008, 0.006, 0.004, 0.003, 0.006, 0.004, 0.004, 0.0, 0.002, 0.001, 0.003, 0.005, 0.001, 0.001, 0.048, 0.053, 0.029, 0.059, 0.024, 0.035};
                
                diff_u1 = {0.602, 0.707, 0.8, 0.88, 0.95, 1.01, 1.059, 1.1, 1.133, 1.159, 1.178, 1.191, 1.199, 1.202, 1.202, 1.2, 1.195, 1.189, 1.182, 1.175, 1.17, 1.166, 1.164, 1.166, 1.171, 1.182}; 
                diff_u1_unc_up = {0.079, 0.049, 0.067, 0.06, 0.11, 0.122, 0.138, 0.143, 0.159, 0.145, 0.144, 0.153, 0.117, 0.173, 0.108, 0.123, 0.038, 0.047, 0.053, 0.054, 0.05, 0.153, 0.16, 0.312, 0.556, 0.774}; 
                diff_u1_unc_down = {0.003, 0.034, 0.059, 0.077, 0.088, 0.092, 0.092, 0.087, 0.078, 0.067, 0.053, 0.037, 0.021, 0.007, 0.006, 0.009, 0.08, 0.354, 0.207, 0.247, 0.353, 0.021, 0.019, 0.019, 0.044, 0.084};
                
                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300};
            }


            /*
            // HEM region removed
            sigma_ratio = 1.038;
            sigma_ratio_unc_up = 0.009;
            sigma_ratio_unc_down = 0.004;

            if( Njets_ISR == 0 ){
                mc_u1_mean = {-39.735, -49.51, -59.656, -70.056, -80.829, -91.674, -102.623, -113.277, -123.382, -133.979, -144.568, -154.048, -164.689, -175.882, -184.68, -196.536, -205.718};
                mc_u1_mean_unc_up = {0.14, 0.161, 0.178, 0.143, 0.182, 0.164, 0.162, 0.056, 0.05, 0.003, 0.082, 0.006, 0.007, 0.003, 0.024, 0.021, 0.306};
                mc_u1_mean_unc_down = {0.021, 0.021, 0.022, 0.022, 0.021, 0.012, 0.006, 0.003, 0.009, 0.115, 0.003, 0.115, 0.176, 0.181, 0.004, 0.338, 0.049};

                diff_u1 = {1.477, 1.8, 2.074, 2.297, 2.469, 2.591, 2.662, 2.682, 2.652, 2.571, 2.44, 2.258, 2.026, 1.743, 1.41, 1.025, 0.591};
                diff_u1_unc_up = {0.085, 0.164, 0.194, 0.212, 0.207, 0.172, 0.188, 0.182, 0.043, 0.04, 0.036, 0.579, 0.473, 0.554, 0.681, 1.22, 2.866};
                diff_u1_unc_down = {0.033, 0.053, 0.07, 0.082, 0.091, 0.095, 0.096, 0.092, 0.237, 0.468, 0.489, 0.038, 0.021, 0.021, 0.028, 0.041, 0.056};

                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210};
            }else if( Njets_ISR == 1 ){
                mc_u1_mean = {-44.905, -54.308, -63.902, -73.502, -83.348, -93.232, -103.289, -113.369, -123.5, -133.734, -143.783, -153.98, -164.091, -174.382, -184.712, -194.576, -204.853, -215.352, -225.485, -235.08, -245.56, -255.488, -265.909, -276.026, -286.141, -296.153};
                mc_u1_mean_unc_up = {0.001, 0.011, 0.034, 0.038, 0.067, 0.063, 0.072, 0.047, 0.042, 0.06, 0.046, 0.032, 0.026, 0.005, 0.051, 0.053, 0.002, 0.041, 0.009, 0.009, 0.012, 0.027, 0.028, 0.023, 0.02, 0.013};
                mc_u1_mean_unc_down = {0.0, 0.001, 0.003, 0.005, 0.005, 0.005, 0.003, 0.003, 0.004, 0.002, 0.002, 0.003, 0.001, 0.019, 0.015, 0.001, 0.05, 0.003, 0.025, 0.147, 0.133, 0.213, 0.222, 0.148, 0.123, 0.119};

                diff_u1 = {0.753, 0.899, 1.014, 1.101, 1.164, 1.205, 1.228, 1.236, 1.231, 1.217, 1.197, 1.174, 1.151, 1.131, 1.117, 1.113, 1.121, 1.144, 1.187, 1.25, 1.339, 1.455, 1.602, 1.783, 2.0, 2.258};
                diff_u1_unc_up = {0.032, 0.047, 0.067, 0.072, 0.074, 0.086, 0.082, 0.095, 0.096, 0.098, 0.102, 0.101, 0.106, 0.113, 0.126, 0.155, 0.133, 0.164, 0.225, 0.167, 0.141, 0.337, 0.192, 0.212, 0.269, 0.316};
                diff_u1_unc_down = {0.003, 0.011, 0.02, 0.028, 0.034, 0.038, 0.041, 0.043, 0.044, 0.044, 0.042, 0.04, 0.037, 0.034, 0.03, 0.026, 0.021, 0.017, 0.014, 0.012, 0.013, 0.015, 0.018, 0.022, 0.026, 0.03};

                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300};
            }else if( Njets_ISR >= 2 ){
                mc_u1_mean = {-44.767, -54.68, -64.813, -74.552, -84.422, -94.344, -104.104, -114.079, -123.788, -133.916, -143.752, -153.866, -163.719, -174.363, -183.957, -194.173, -204.098, -214.138, -224.253, -234.457, -244.755, -254.754, -264.721, -274.824, -284.833, -294.78};
                mc_u1_mean_unc_up = {0.013, 0.03, 0.06, 0.029, 0.053, 0.062, 0.049, 0.088, 0.081, 0.091, 0.099, 0.094, 0.061, 0.065, 0.117, 0.097, 0.004, 0.062, 0.199, 0.005, 0.011, 0.012, 0.02, 0.006, 0.031, 0.008};
                mc_u1_mean_unc_down = {0.005, 0.007, 0.006, 0.005, 0.006, 0.007, 0.008, 0.006, 0.004, 0.003, 0.006, 0.005, 0.005, 0.001, 0.003, 0.002, 0.007, 0.006, 0.001, 0.034, 0.02, 0.026, 0.024, 0.06, 0.01, 0.03};

                diff_u1 = {0.63, 0.699, 0.777, 0.856, 0.934, 1.007, 1.072, 1.126, 1.169, 1.199, 1.215, 1.218, 1.21, 1.191, 1.164, 1.131, 1.097, 1.065, 1.04, 1.027, 1.033, 1.064, 1.128, 1.232, 1.385, 1.597};
                diff_u1_unc_up = {0.055, 0.058, 0.057, 0.057, 0.102, 0.114, 0.132, 0.118, 0.141, 0.109, 0.123, 0.119, 0.12, 0.118, 0.154, 0.129, 0.172, 0.146, 0.259, 0.197, 0.308, 0.213, 0.21, 0.212, 0.3, 0.361};
                diff_u1_unc_down = {0.002, 0.032, 0.055, 0.069, 0.077, 0.079, 0.077, 0.071, 0.061, 0.05, 0.038, 0.024, 0.012, 0.007, 0.009, 0.013, 0.016, 0.019, 0.021, 0.023, 0.024, 0.024, 0.024, 0.023, 0.022, 0.02};

                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300};
            }
            */

            /*
            sigma_ratio = 1.039;
            sigma_ratio_unc_up = 0.009;
            sigma_ratio_unc_down = 0.004;

            if( Njets_ISR == 0 ){
                mc_u1_mean = {-39.735, -49.51, -59.656, -70.056, -80.829, -91.674, -102.623, -113.278, -123.383, -133.98, -144.568, -154.048, -164.689, -175.882, -184.68, -196.536, -205.718};
                mc_u1_mean_unc_up = {0.14, 0.161, 0.178, 0.143, 0.183, 0.164, 0.162, 0.056, 0.05, 0.003, 0.082, 0.006, 0.007, 0.003, 0.024, 0.021, 0.306};
                mc_u1_mean_unc_down = {0.021, 0.021, 0.022, 0.022, 0.021, 0.012, 0.006, 0.003, 0.009, 0.115, 0.003, 0.115, 0.176, 0.181, 0.004, 0.338, 0.049};

                diff_u1 = {1.477, 1.801, 2.074, 2.297, 2.469, 2.591, 2.662, 2.682, 2.652, 2.572, 2.44, 2.258, 2.026, 1.743, 1.409, 1.025, 0.591};
                diff_u1_unc_up = {0.085, 0.164, 0.194, 0.212, 0.207, 0.172, 0.188, 0.182, 0.043, 0.04, 0.036, 0.579, 0.473, 0.554, 0.681, 1.22, 2.866};
                diff_u1_unc_down = {0.033, 0.053, 0.07, 0.082, 0.091, 0.095, 0.096, 0.092, 0.237, 0.468, 0.489, 0.038, 0.021, 0.021, 0.028, 0.041, 0.056};

                mc_u2_mean = {0.013, 0.075, 0.088, 0.088, 0.096, 0.124, 0.17, 0.22, 0.247, 0.213, 0.069};
                mc_u2_mean_unc_up = {0.001, 0.003, 0.003, 0.013, 0.025, 0.013, 0.0, 0.001, 0.003, 0.008, 0.067};
                mc_u2_mean_unc_down = {0.007, 0.032, 0.014, 0.001, 0.001, 0.0, 0.023, 0.067, 0.095, 0.068, 0.008};

                data_u2_mean = {0.016, 0.011, 0.018, 0.038, 0.07, 0.114, 0.17, 0.239, 0.319, 0.412, 0.517};
                data_u2_mean_unc_up = {0.002, 0.002, 0.015, 0.029, 0.043, 0.058, 0.073, 0.089, 0.105, 0.122, 0.14};
                data_u2_mean_unc_down = {0.011, 0.0, 0.002, 0.003, 0.004, 0.005, 0.005, 0.005, 0.004, 0.002, 0.001};

                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210};
            }else if( Njets_ISR == 1 ){
                mc_u1_mean = {-44.905, -54.308, -63.902, -73.502, -83.348, -93.232, -103.289, -113.369, -123.5, -133.734, -143.783, -153.98, -164.091, -174.382, -184.712, -194.576, -204.853, -215.352, -225.485, -235.08, -245.56, -255.489, -265.909, -276.026, -286.141, -296.153};
                mc_u1_mean_unc_up = {0.001, 0.012, 0.034, 0.038, 0.067, 0.063, 0.072, 0.047, 0.042, 0.06, 0.046, 0.032, 0.026, 0.005, 0.051, 0.053, 0.002, 0.042, 0.009, 0.009, 0.012, 0.027, 0.028, 0.023, 0.02, 0.013};
                mc_u1_mean_unc_down = {0.0, 0.001, 0.003, 0.005, 0.005, 0.005, 0.003, 0.003, 0.004, 0.002, 0.002, 0.003, 0.001, 0.019, 0.015, 0.001, 0.05, 0.003, 0.025, 0.147, 0.132, 0.213, 0.222, 0.148, 0.123, 0.119};

                diff_u1 = {0.753, 0.899, 1.014, 1.101, 1.164, 1.205, 1.228, 1.236, 1.231, 1.217, 1.197, 1.174, 1.151, 1.131, 1.117, 1.113, 1.121, 1.145, 1.187, 1.25, 1.339, 1.455, 1.602, 1.783, 2.001, 2.259};
                diff_u1_unc_up = {0.032, 0.047, 0.067, 0.072, 0.074, 0.086, 0.082, 0.095, 0.096, 0.098, 0.102, 0.101, 0.106, 0.113, 0.126, 0.155, 0.133, 0.164, 0.225, 0.167, 0.141, 0.337, 0.192, 0.212, 0.269, 0.316};
                diff_u1_unc_down = {0.003, 0.011, 0.02, 0.028, 0.034, 0.038, 0.041, 0.043, 0.044, 0.044, 0.042, 0.04, 0.037, 0.034, 0.03, 0.026, 0.021, 0.017, 0.014, 0.012, 0.013, 0.015, 0.018, 0.022, 0.026, 0.03};

                mc_u2_mean = {0.013, 0.032, 0.038, 0.036, 0.03, 0.023, 0.017, 0.016, 0.019, 0.027, 0.041, 0.058, 0.078, 0.098, 0.116, 0.126, 0.126, 0.109, 0.071};
                mc_u2_mean_unc_up = {0.002, 0.001, 0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.011, 0.01, 0.009, 0.008, 0.006, 0.005, 0.005, 0.007, 0.011, 0.019, 0.032};
                mc_u2_mean_unc_down = {0.001, 0.0, 0.0, 0.001, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.002, 0.003, 0.005};

                data_u2_mean = {0.014, 0.013, 0.013, 0.016, 0.022, 0.03, 0.041, 0.054, 0.069, 0.087, 0.107, 0.13, 0.155, 0.182, 0.212, 0.245, 0.28, 0.317, 0.357};
                data_u2_mean_unc_up = {0.004, 0.0, 0.001, 0.006, 0.012, 0.016, 0.017, 0.014, 0.008, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.025, 0.083, 0.172, 0.301};
                data_u2_mean_unc_down = {0.0, 0.001, 0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.012, 0.022, 0.027, 0.024, 0.009, 0.001, 0.001, 0.001, 0.001};

                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300};
            }else if( Njets_ISR >= 2 ){
                mc_u1_mean = {-44.767, -54.68, -64.813, -74.552, -84.422, -94.344, -104.104, -114.079, -123.788, -133.916, -143.752, -153.867, -163.719, -174.363, -183.957, -194.173, -204.098, -214.138, -224.253, -234.457, -244.755, -254.754, -264.721, -274.824, -284.834, -294.78};
                mc_u1_mean_unc_up = {0.013, 0.03, 0.06, 0.029, 0.053, 0.062, 0.049, 0.088, 0.081, 0.091, 0.099, 0.094, 0.061, 0.065, 0.117, 0.097, 0.004, 0.062, 0.199, 0.005, 0.011, 0.012, 0.02, 0.006, 0.031, 0.008};
                mc_u1_mean_unc_down = {0.005, 0.007, 0.006, 0.005, 0.006, 0.007, 0.008, 0.006, 0.004, 0.003, 0.006, 0.005, 0.005, 0.001, 0.003, 0.002, 0.007, 0.006, 0.001, 0.034, 0.02, 0.026, 0.024, 0.06, 0.01, 0.03};

                diff_u1 = {0.63, 0.699, 0.777, 0.856, 0.934, 1.007, 1.072, 1.126, 1.169, 1.199, 1.215, 1.218, 1.21, 1.191, 1.164, 1.131, 1.097, 1.065, 1.04, 1.027, 1.033, 1.064, 1.128, 1.232, 1.385, 1.597};
                diff_u1_unc_up = {0.055, 0.058, 0.057, 0.057, 0.102, 0.114, 0.132, 0.118, 0.141, 0.109, 0.123, 0.119, 0.12, 0.118, 0.154, 0.129, 0.172, 0.146, 0.259, 0.197, 0.308, 0.213, 0.21, 0.211, 0.3, 0.361};
                diff_u1_unc_down = {0.002, 0.032, 0.055, 0.069, 0.077, 0.079, 0.077, 0.071, 0.061, 0.05, 0.037, 0.024, 0.012, 0.007, 0.009, 0.013, 0.016, 0.019, 0.021, 0.023, 0.024, 0.024, 0.024, 0.023, 0.022, 0.02};

                mc_u2_mean = {0.022, 0.022, 0.025, 0.03, 0.036, 0.044, 0.053, 0.063, 0.073, 0.084, 0.095, 0.106, 0.116, 0.126, 0.135, 0.143, 0.149, 0.154, 0.157, 0.157, 0.156, 0.152, 0.144, 0.134, 0.12, 0.103};
                mc_u2_mean_unc_up = {0.009, 0.007, 0.006, 0.006, 0.005, 0.006, 0.006, 0.006, 0.007, 0.008, 0.008, 0.009, 0.009, 0.01, 0.01, 0.01, 0.009, 0.009, 0.007, 0.006, 0.003, 0.001, 0.002, 0.003, 0.004, 0.006};
                mc_u2_mean_unc_down = {0.001, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0, 0.0, 0.001, 0.001, 0.004, 0.008, 0.013, 0.019};

                data_u2_mean = {0.037, 0.036, 0.036, 0.039, 0.042, 0.047, 0.053, 0.061, 0.071, 0.081, 0.094, 0.107, 0.122, 0.139, 0.157, 0.177, 0.198, 0.22, 0.244, 0.269, 0.296, 0.324, 0.354, 0.385, 0.418, 0.452};
                data_u2_mean_unc_up = {0.005, 0.003, 0.002, 0.003, 0.004, 0.005, 0.006, 0.005, 0.004, 0.003, 0.001, 0.002, 0.003, 0.005, 0.006, 0.008, 0.011, 0.013, 0.016, 0.019, 0.023, 0.026, 0.03, 0.034, 0.039, 0.044};
                data_u2_mean_unc_down = {0.007, 0.003, 0.001, 0.001, 0.0, 0.001, 0.001, 0.001, 0.001, 0.0, 0.0, 0.003, 0.007, 0.011, 0.016, 0.022, 0.028, 0.035, 0.043, 0.051, 0.06, 0.07, 0.08, 0.091, 0.103, 0.115};

                ZRecoPt = {40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300};
            }
            */

        }


        /*
        int idx2 = -1;
        for (unsigned int i = 0; i < mc_u2_mean.size() ; i++){
            if ( LepLep_pt >= ZRecoPt[i] && LepLep_pt < ZRecoPt[i+1]  ){
                idx2 = i;
                break;
            }
        }

        float Ux_new;
        float Uy_new;

        if( idx2 >= 0 ){

            float data_u2_mean_value = data_u2_mean[idx2];
            float mc_u2_mean_value = mc_u2_mean[idx2];

            float U2_new = data_u2_mean_value + (U2-mc_u2_mean_value)*sigma_ratio;
            Ux_new = - U2_new*sin(LepLep_phi);
            Uy_new = + U2_new*cos(LepLep_phi);

        }else{
            Ux_new = - U2*sin(LepLep_phi);
            Uy_new = + U2*cos(LepLep_phi);
        }
        */


        int idx = -1;
        for (unsigned int i = 0; i < mc_u1_mean.size() ; i++){
            if ( LepLep_pt >= ZRecoPt[i] && LepLep_pt < ZRecoPt[i+1]  ){
                idx = i;
                break;
            }
        }
        
        if( _sysName_lateral == "Recoil" ){
            if( (_Universe == 0) || (_Universe == 1) ){             // ratio up
                sigma_ratio = sigma_ratio + sigma_ratio_unc_up;
            }else if( (_Universe == 2) || (_Universe == 3) ){       // ratio down
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
    
    //=====JER Correction==========================================================================
    //if( apply_jer_corr && (dataset_group != "Data") && !(apply_met_recoil_corr && ((dsNameDY == "DYJetsToLL") || (dsNameZZ == "ZZTo2L") || (dsNameZZ == "ZZTo4L"))) ){
    if( apply_jer_corr && (dataset_group != "Data") && !(apply_met_recoil_corr && (dsNameDY == "DYJetsToLL")) ){ // use for normal analysis
    //if( apply_jer_corr && (dataset_group != "Data") && !(dsNameDY == "DYJetsToLL") ){  //uncomment this line to produce Zrecoil corrections
        
        TLorentzVector METLV;
        METLV.SetPtEtaPhiM(MET_pt, 0., MET_phi, 0.);
        METLV += METCorrectionFromJER;
        
        MET_pt = METLV.Pt();
        MET_phi = METLV.Phi();
    }
    
    MET_JER_pt = MET_pt;
    MET_JER_phi = MET_phi;
    
    //=====MET Emulation===========================================================================
    // MET Emulation for regions with more than 2 leptons
    bool control = false;
    double METxcorr;
    double METycorr;
    
    if( Nleptons == 3 ){
        if( RecoLepID == 31199 ){
            int imu = selectedMu.at(0);
            METxcorr = Muon_pt[imu]*cos(Muon_phi[imu]);
            METycorr = Muon_pt[imu]*sin(Muon_phi[imu]);
            control = true;
        }else if( RecoLepID == 31101 ){
            int iele = selectedEle.at(2);
            METxcorr = Electron_pt[iele]*cos(Electron_phi[iele]);
            METycorr = Electron_pt[iele]*sin(Electron_phi[iele]);
            control = true;
        }else if( RecoLepID == 31112 ){
            int iele = selectedEle.at(0);
            METxcorr = Electron_pt[iele]*cos(Electron_phi[iele]);
            METycorr = Electron_pt[iele]*sin(Electron_phi[iele]);
            control = true;
        }else if( RecoLepID == 31120 ){
            int iele = selectedEle.at(1);
            METxcorr = Electron_pt[iele]*cos(Electron_phi[iele]);
            METycorr = Electron_pt[iele]*sin(Electron_phi[iele]);
            control = true;   
        }else if( RecoLepID == 31399 ){
            int iele = selectedEle.at(0);
            METxcorr = Electron_pt[iele]*cos(Electron_phi[iele]);
            METycorr = Electron_pt[iele]*sin(Electron_phi[iele]);
            control = true;
        }else if( RecoLepID == 31301 ){
            int imu = selectedMu.at(2);
            METxcorr = Muon_pt[imu]*cos(Muon_phi[imu]);
            METycorr = Muon_pt[imu]*sin(Muon_phi[imu]);
            control = true;
        }else if( RecoLepID == 31312 ){
            int imu = selectedMu.at(0);
            METxcorr = Muon_pt[imu]*cos(Muon_phi[imu]);
            METycorr = Muon_pt[imu]*sin(Muon_phi[imu]);
            control = true;
        }else if( RecoLepID == 31320 ){
            int imu = selectedMu.at(1);
            METxcorr = Muon_pt[imu]*cos(Muon_phi[imu]);
            METycorr = Muon_pt[imu]*sin(Muon_phi[imu]);
            control = true;    
        }
    }
        
    if( Nleptons == 4 ){    
        if( RecoLepID == 41199 ){
            int imu1 = selectedMu.at(0);
            int imu2 = selectedMu.at(1);
            METxcorr = Muon_pt[imu1]*cos(Muon_phi[imu1]) + Muon_pt[imu2]*cos(Muon_phi[imu2]);
            METycorr = Muon_pt[imu1]*sin(Muon_phi[imu1]) + Muon_pt[imu2]*sin(Muon_phi[imu2]);
            control = true;
        }else if( RecoLepID == 41101 ){
            int iele1 = selectedEle.at(2);
            int iele2 = selectedEle.at(3);
            METxcorr = Electron_pt[iele1]*cos(Electron_phi[iele1]) + Electron_pt[iele2]*cos(Electron_phi[iele2]);
            METycorr = Electron_pt[iele1]*sin(Electron_phi[iele1]) + Electron_pt[iele2]*sin(Electron_phi[iele2]);
            control = true;
        }else if( RecoLepID == 41102 ){
            int iele1 = selectedEle.at(1);
            int iele2 = selectedEle.at(3);
            METxcorr = Electron_pt[iele1]*cos(Electron_phi[iele1]) + Electron_pt[iele2]*cos(Electron_phi[iele2]);
            METycorr = Electron_pt[iele1]*sin(Electron_phi[iele1]) + Electron_pt[iele2]*sin(Electron_phi[iele2]);
            control = true;
        }else if( RecoLepID == 41103 ){
            int iele1 = selectedEle.at(1);
            int iele2 = selectedEle.at(2);
            METxcorr = Electron_pt[iele1]*cos(Electron_phi[iele1]) + Electron_pt[iele2]*cos(Electron_phi[iele2]);
            METycorr = Electron_pt[iele1]*sin(Electron_phi[iele1]) + Electron_pt[iele2]*sin(Electron_phi[iele2]);
            control = true;
        }else if( RecoLepID == 41112 ){
            int iele1 = selectedEle.at(3);
            int iele2 = selectedEle.at(0);
            METxcorr = Electron_pt[iele1]*cos(Electron_phi[iele1]) + Electron_pt[iele2]*cos(Electron_phi[iele2]);
            METycorr = Electron_pt[iele1]*sin(Electron_phi[iele1]) + Electron_pt[iele2]*sin(Electron_phi[iele2]);
            control = true;
        }else if( RecoLepID == 41113 ){
            int iele1 = selectedEle.at(2);
            int iele2 = selectedEle.at(0);
            METxcorr = Electron_pt[iele1]*cos(Electron_phi[iele1]) + Electron_pt[iele2]*cos(Electron_phi[iele2]);
            METycorr = Electron_pt[iele1]*sin(Electron_phi[iele1]) + Electron_pt[iele2]*sin(Electron_phi[iele2]);
            control = true;
        }else if( RecoLepID == 41123 ){
            int iele1 = selectedEle.at(0);
            int iele2 = selectedEle.at(1);
            METxcorr = Electron_pt[iele1]*cos(Electron_phi[iele1]) + Electron_pt[iele2]*cos(Electron_phi[iele2]);
            METycorr = Electron_pt[iele1]*sin(Electron_phi[iele1]) + Electron_pt[iele2]*sin(Electron_phi[iele2]);
            control = true;
        }else if( RecoLepID == 41399 ){
            int iele1 = selectedEle.at(0);
            int iele2 = selectedEle.at(1);
            METxcorr = Electron_pt[iele1]*cos(Electron_phi[iele1]) + Electron_pt[iele2]*cos(Electron_phi[iele2]);
            METycorr = Electron_pt[iele1]*sin(Electron_phi[iele1]) + Electron_pt[iele2]*sin(Electron_phi[iele2]);
            control = true;
        }else if( RecoLepID == 41301 ){
            int imu1 = selectedMu.at(2);
            int imu2 = selectedMu.at(3);
            METxcorr = Muon_pt[imu1]*cos(Muon_phi[imu1]) + Muon_pt[imu2]*cos(Muon_phi[imu2]);
            METycorr = Muon_pt[imu1]*sin(Muon_phi[imu1]) + Muon_pt[imu2]*sin(Muon_phi[imu2]);
            control = true;
        }else if( RecoLepID == 41302 ){
            int imu1 = selectedMu.at(1);
            int imu2 = selectedMu.at(3);
            METxcorr = Muon_pt[imu1]*cos(Muon_phi[imu1]) + Muon_pt[imu2]*cos(Muon_phi[imu2]);
            METycorr = Muon_pt[imu1]*sin(Muon_phi[imu1]) + Muon_pt[imu2]*sin(Muon_phi[imu2]);
            control = true;
        }else if( RecoLepID == 41303 ){
            int imu1 = selectedMu.at(1);
            int imu2 = selectedMu.at(2);
            METxcorr = Muon_pt[imu1]*cos(Muon_phi[imu1]) + Muon_pt[imu2]*cos(Muon_phi[imu2]);
            METycorr = Muon_pt[imu1]*sin(Muon_phi[imu1]) + Muon_pt[imu2]*sin(Muon_phi[imu2]);
            control = true;
        }else if( RecoLepID == 41312 ){
            int imu1 = selectedMu.at(3);
            int imu2 = selectedMu.at(0);
            METxcorr = Muon_pt[imu1]*cos(Muon_phi[imu1]) + Muon_pt[imu2]*cos(Muon_phi[imu2]);
            METycorr = Muon_pt[imu1]*sin(Muon_phi[imu1]) + Muon_pt[imu2]*sin(Muon_phi[imu2]);
            control = true;
        }else if( RecoLepID == 41313 ){
            int imu1 = selectedMu.at(2);
            int imu2 = selectedMu.at(0);
            METxcorr = Muon_pt[imu1]*cos(Muon_phi[imu1]) + Muon_pt[imu2]*cos(Muon_phi[imu2]);
            METycorr = Muon_pt[imu1]*sin(Muon_phi[imu1]) + Muon_pt[imu2]*sin(Muon_phi[imu2]);
            control = true;
        }else if( RecoLepID == 41323 ){
            int imu1 = selectedMu.at(0);
            int imu2 = selectedMu.at(1);
            METxcorr = Muon_pt[imu1]*cos(Muon_phi[imu1]) + Muon_pt[imu2]*cos(Muon_phi[imu2]);
            METycorr = Muon_pt[imu1]*sin(Muon_phi[imu1]) + Muon_pt[imu2]*sin(Muon_phi[imu2]);
            control = true;
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
    
    //=====NEW MET Significance===========================================================================
    double MET_px = MET_pt*cos(MET_phi);
    double MET_py = MET_pt*sin(MET_phi);
    
    double det = MET_covXX * MET_covYY - MET_covXY * MET_covXY;
    
    // invert matrix
    double ncov_xx = MET_covYY / det;
    double ncov_xy = -MET_covXY / det;
    double ncov_yy = MET_covXX / det;

    MET_sig = MET_px * MET_px * ncov_xx + 2 * MET_px * MET_py * ncov_xy + MET_py * MET_py * ncov_yy;
    
}


//---------------------------------------------------------------------------------------------------------------
// Lepton selection
//---------------------------------------------------------------------------------------------------------------
void HEPHero::LeptonSelection(){
    
    selectedEle.clear();
    selectedMu.clear();
    selectedEleLowPt.clear();
    selectedMuLowPt.clear();
    Min_dilep_deltaR = 999999;
    
    for( unsigned int iele = 0; iele < nElectron; ++iele ) {
        if( abs(Electron_eta[iele]) >= ELECTRON_ETA_CUT ) continue;
        if( !ElectronID( iele, ELECTRON_ID_WP ) ) continue;
        if( (abs(Electron_eta[iele] + Electron_deltaEtaSC[iele]) > ELECTRON_GAP_LOWER_CUT) && 
            (abs(Electron_eta[iele] + Electron_deltaEtaSC[iele]) < ELECTRON_GAP_UPPER_CUT) ) continue;
        if( Electron_pt[iele] > ELECTRON_LOW_PT_CUT ) selectedEleLowPt.push_back(iele);
        if( Electron_pt[iele] <= ELECTRON_PT_CUT ) continue; 
        
        TLorentzVector Ele_test;
        Ele_test.SetPtEtaPhiM(Electron_pt[iele], Electron_eta[iele], Electron_phi[iele], Electron_pdg_mass);
        for( unsigned int iselele = 0; iselele < selectedEle.size(); ++iselele ) {
            int iele_sel = selectedEle[iselele];
            TLorentzVector Ele_sel;
            Ele_sel.SetPtEtaPhiM(Electron_pt[iele_sel], Electron_eta[iele_sel], Electron_phi[iele_sel], Electron_pdg_mass);
            float dilep_deltaR = Ele_test.DeltaR( Ele_sel );
            if( dilep_deltaR < Min_dilep_deltaR ) Min_dilep_deltaR = dilep_deltaR;
        }
        
        selectedEle.push_back(iele);
    }
    
    for( unsigned int imu = 0; imu < nMuon; ++imu ) {
        Muon_raw_pt[imu] = Muon_pt[imu];
        if( abs(Muon_eta[imu]) >= MUON_ETA_CUT ) continue;
        if( !MuonID( imu, MUON_ID_WP ) ) continue;
        if( !MuonISO( imu, MUON_ISO_WP ) ) continue;
        
        if( apply_muon_roc_corr ) Muon_pt[imu] = Muon_pt[imu]*muon_roc_corr.GetCorrection( Muon_charge[imu], Muon_pt[imu], Muon_eta[imu], Muon_phi[imu], (Muon_genPartIdx[imu]>=0) ? true : false, (Muon_genPartIdx[imu]>=0) ? GenPart_pt[Muon_genPartIdx[imu]] : 0., Muon_nTrackerLayers[imu], (dataset_group=="Data") );
        
        if( Muon_pt[imu] > MUON_LOW_PT_CUT ) selectedMuLowPt.push_back(imu);
        if( Muon_pt[imu] <= MUON_PT_CUT ) continue;
        
        TLorentzVector Mu_test;
        Mu_test.SetPtEtaPhiM(Muon_pt[imu], Muon_eta[imu], Muon_phi[imu], Muon_pdg_mass);
        for( unsigned int iselmu = 0; iselmu < selectedMu.size(); ++iselmu ) {
            int imu_sel = selectedMu[iselmu];
            TLorentzVector Mu_sel;
            Mu_sel.SetPtEtaPhiM(Muon_pt[imu_sel], Muon_eta[imu_sel], Muon_phi[imu_sel], Muon_pdg_mass);
            float dilep_deltaR = Mu_test.DeltaR( Mu_sel );
            if( dilep_deltaR < Min_dilep_deltaR ) Min_dilep_deltaR = dilep_deltaR;
        }
        for( unsigned int iselele = 0; iselele < selectedEle.size(); ++iselele ) {
            int iele_sel = selectedEle[iselele];
            TLorentzVector Ele_sel;
            Ele_sel.SetPtEtaPhiM(Electron_pt[iele_sel], Electron_eta[iele_sel], Electron_phi[iele_sel], Electron_pdg_mass);
            float dilep_deltaR = Mu_test.DeltaR( Ele_sel );
            if( dilep_deltaR < Min_dilep_deltaR ) Min_dilep_deltaR = dilep_deltaR;
        }
        
        selectedMu.push_back(imu);
    }
    
    Nelectrons = selectedEle.size();
    Nmuons = selectedMu.size();
    Nleptons = Nelectrons + Nmuons;
    
    int NelectronsLowPt = selectedEleLowPt.size();
    int NmuonsLowPt = selectedMuLowPt.size();
    NleptonsLowPt = NelectronsLowPt + NmuonsLowPt;
    
    RecoLepID = 0;
    if( (Nelectrons == 2) && (Nmuons == 0) ){
        int iele_1 = selectedEle.at(0);
        int iele_2 = selectedEle.at(1);
        if( Electron_charge[iele_1]*Electron_charge[iele_2] < 0 ) RecoLepID = 11;   // 2 Electrons OS
    }else if( (Nelectrons == 0) && (Nmuons == 2) ){ 
        int imu_1 = selectedMu.at(0);
        int imu_2 = selectedMu.at(1);
        if( Muon_charge[imu_1]*Muon_charge[imu_2] < 0 ) RecoLepID = 13;             // 2 Muons OS
    }else if( (Nelectrons == 1) && (Nmuons == 1) ){
        int iele = selectedEle.at(0);
        int imu = selectedMu.at(0);
        if( Electron_charge[iele]*Muon_charge[imu] < 0 ){
            if( Electron_pt[iele] > Muon_pt[imu] ){ 
                RecoLepID = 1113;                                                   // 1 Electron and 1 Muon OS
            }else{
                RecoLepID = 1311;                                                   // 1 Muon and 1 Electron OS
            }
        }
    //-----------------------------------------------------------------------------------------------------------------
    }else if( (Nelectrons == 2) && (Nmuons == 1) ){
        int iele_0 = selectedEle.at(0);
        int iele_1 = selectedEle.at(1);
        if( Electron_charge[iele_0]*Electron_charge[iele_1] < 0 ) RecoLepID = 31199;  // 2 Electrons OS & 1 Muon
    }else if( (Nelectrons == 1) && (Nmuons == 2) ){
        int imu_0 = selectedMu.at(0);
        int imu_1 = selectedMu.at(1);
        if( Muon_charge[imu_0]*Muon_charge[imu_1] < 0 ) RecoLepID = 31399;            // 2 Muons OS & 1 Electron
    }else if( (Nelectrons == 3) && (Nmuons == 0) ){
        int iele_0 = selectedEle.at(0);
        int iele_1 = selectedEle.at(1);
        int iele_2 = selectedEle.at(2);
        int ele_charge_prod_01 = Electron_charge[iele_0]*Electron_charge[iele_1];
        int ele_charge_prod_12 = Electron_charge[iele_1]*Electron_charge[iele_2];
        int ele_charge_prod_20 = Electron_charge[iele_2]*Electron_charge[iele_0];
        TLorentzVector ele_0;
        TLorentzVector ele_1;
        TLorentzVector ele_2;
        ele_0.SetPtEtaPhiM(Electron_pt[iele_0], Electron_eta[iele_0], Electron_phi[iele_0], Electron_pdg_mass);
        ele_1.SetPtEtaPhiM(Electron_pt[iele_1], Electron_eta[iele_1], Electron_phi[iele_1], Electron_pdg_mass);
        ele_2.SetPtEtaPhiM(Electron_pt[iele_2], Electron_eta[iele_2], Electron_phi[iele_2], Electron_pdg_mass);
        if( (ele_charge_prod_01 < 0) && (ele_charge_prod_12 < 0) ){ 
            float DeltaM_01 = abs( (ele_0 + ele_1).M() - Z_pdg_mass );
            float DeltaM_12 = abs( (ele_1 + ele_2).M() - Z_pdg_mass );
            if( DeltaM_01 < DeltaM_12 ){
                RecoLepID = 31101;                                                    // 2 Electrons OS & 1 Electron
            }else{
                RecoLepID = 31112;                                                    // 2 Electrons OS & 1 Electron
            }
        }else if( (ele_charge_prod_01 < 0) && (ele_charge_prod_20 < 0) ){ 
            float DeltaM_01 = abs( (ele_0 + ele_1).M() - Z_pdg_mass );
            float DeltaM_20 = abs( (ele_2 + ele_0).M() - Z_pdg_mass );
            if( DeltaM_01 < DeltaM_20 ){
                RecoLepID = 31101;                                                    // 2 Electrons OS & 1 Electron
            }else{
                RecoLepID = 31120;                                                    // 2 Electrons OS & 1 Electron
            }
        }else if( (ele_charge_prod_20 < 0) && (ele_charge_prod_12 < 0) ){ 
            float DeltaM_20 = abs( (ele_2 + ele_0).M() - Z_pdg_mass );
            float DeltaM_12 = abs( (ele_1 + ele_2).M() - Z_pdg_mass );
            if( DeltaM_20 < DeltaM_12 ){
                RecoLepID = 31120;                                                    // 2 Electrons OS & 1 Electron
            }else{
                RecoLepID = 31112;                                                    // 2 Electrons OS & 1 Electron
            }
        }
    }else if( (Nelectrons == 0) && (Nmuons == 3) ){
        int imu_0 = selectedMu.at(0);
        int imu_1 = selectedMu.at(1);
        int imu_2 = selectedMu.at(2);
        int mu_charge_prod_01 = Muon_charge[imu_0]*Muon_charge[imu_1];
        int mu_charge_prod_12 = Muon_charge[imu_1]*Muon_charge[imu_2];
        int mu_charge_prod_20 = Muon_charge[imu_2]*Muon_charge[imu_0];
        TLorentzVector mu_0;
        TLorentzVector mu_1;
        TLorentzVector mu_2;
        mu_0.SetPtEtaPhiM(Muon_pt[imu_0], Muon_eta[imu_0], Muon_phi[imu_0], Muon_pdg_mass);
        mu_1.SetPtEtaPhiM(Muon_pt[imu_1], Muon_eta[imu_1], Muon_phi[imu_1], Muon_pdg_mass);
        mu_2.SetPtEtaPhiM(Muon_pt[imu_2], Muon_eta[imu_2], Muon_phi[imu_2], Muon_pdg_mass);
        if( (mu_charge_prod_01 < 0) && (mu_charge_prod_12 < 0) ){ 
            float DeltaM_01 = abs( (mu_0 + mu_1).M() - Z_pdg_mass );
            float DeltaM_12 = abs( (mu_1 + mu_2).M() - Z_pdg_mass );
            if( DeltaM_01 < DeltaM_12 ){
                RecoLepID = 31301;                                                    // 2 Muons OS & 1 Muon
            }else{
                RecoLepID = 31312;                                                    // 2 Muons OS & 1 Muon
            }
        }else if( (mu_charge_prod_01 < 0) && (mu_charge_prod_20 < 0) ){ 
            float DeltaM_01 = abs( (mu_0 + mu_1).M() - Z_pdg_mass );
            float DeltaM_20 = abs( (mu_2 + mu_0).M() - Z_pdg_mass );
            if( DeltaM_01 < DeltaM_20 ){
                RecoLepID = 31301;                                                    // 2 Muons OS & 1 Muon
            }else{
                RecoLepID = 31320;                                                    // 2 Muons OS & 1 Muon
            }
        }else if( (mu_charge_prod_20 < 0) && (mu_charge_prod_12 < 0) ){ 
            float DeltaM_20 = abs( (mu_2 + mu_0).M() - Z_pdg_mass );
            float DeltaM_12 = abs( (mu_1 + mu_2).M() - Z_pdg_mass );
            if( DeltaM_20 < DeltaM_12 ){
                RecoLepID = 31320;                                                    // 2 Muons OS & 1 Muon
            }else{
                RecoLepID = 31312;                                                    // 2 Muons OS & 1 Muon
            }
        }
    //-----------------------------------------------------------------------------------------------------------------
    }else if( (Nelectrons == 2) && (Nmuons == 2) ){
        int iele_0 = selectedEle.at(0);
        int iele_1 = selectedEle.at(1);
        int imu_0 = selectedMu.at(0);
        int imu_1 = selectedMu.at(1);
        int ele_charge_prod = Electron_charge[iele_0]*Electron_charge[iele_1];
        int mu_charge_prod = Muon_charge[imu_0]*Muon_charge[imu_1];
        //if( (ele_charge_prod < 0) && (mu_charge_prod > 0) ){ 
        //    RecoLepID = 411;                                                        // 2 Electrons OS & 2 Muons
        //}else if( (ele_charge_prod > 0) && (mu_charge_prod < 0) ){
        //    RecoLepID = 413;                                                        // 2 Muons OS & 2 Electrons
        if( (ele_charge_prod < 0) && (mu_charge_prod < 0) ){
            TLorentzVector ele_0;
            TLorentzVector ele_1;
            TLorentzVector mu_0;
            TLorentzVector mu_1;
            ele_0.SetPtEtaPhiM(Electron_pt[iele_0], Electron_eta[iele_0], Electron_phi[iele_0], Electron_pdg_mass);
            ele_1.SetPtEtaPhiM(Electron_pt[iele_1], Electron_eta[iele_1], Electron_phi[iele_1], Electron_pdg_mass);
            mu_0.SetPtEtaPhiM(Muon_pt[imu_0], Muon_eta[imu_0], Muon_phi[imu_0], Muon_pdg_mass);
            mu_1.SetPtEtaPhiM(Muon_pt[imu_1], Muon_eta[imu_1], Muon_phi[imu_1], Muon_pdg_mass);
            float DeltaM_2ele = abs( (ele_0 + ele_1).M() - Z_pdg_mass );
            float DeltaM_2mu = abs( (mu_0 + mu_1).M() - Z_pdg_mass );
            if( DeltaM_2ele < DeltaM_2mu ){
                RecoLepID = 41199;                                                    // 2 Electrons OS & 2 Muons OS
            }else{
                RecoLepID = 41399;                                                    // 2 Muons OS & 2 Electrons OS
            }
        }
    }else if( (Nelectrons == 4) && (Nmuons == 0) ){
        int iele_0 = selectedEle.at(0);
        int iele_1 = selectedEle.at(1);
        int iele_2 = selectedEle.at(2);
        int iele_3 = selectedEle.at(3);
        int ele_charge_prod_01 = Electron_charge[iele_0]*Electron_charge[iele_1];
        int ele_charge_prod_02 = Electron_charge[iele_0]*Electron_charge[iele_2];
        int ele_charge_prod_03 = Electron_charge[iele_0]*Electron_charge[iele_3];
        int ele_charge_prod_12 = Electron_charge[iele_1]*Electron_charge[iele_2];
        int ele_charge_prod_13 = Electron_charge[iele_1]*Electron_charge[iele_3];
        int ele_charge_prod_23 = Electron_charge[iele_2]*Electron_charge[iele_3];
        TLorentzVector ele_0;
        TLorentzVector ele_1;
        TLorentzVector ele_2;
        TLorentzVector ele_3;
        ele_0.SetPtEtaPhiM(Electron_pt[iele_0], Electron_eta[iele_0], Electron_phi[iele_0], Electron_pdg_mass);
        ele_1.SetPtEtaPhiM(Electron_pt[iele_1], Electron_eta[iele_1], Electron_phi[iele_1], Electron_pdg_mass);
        ele_2.SetPtEtaPhiM(Electron_pt[iele_2], Electron_eta[iele_2], Electron_phi[iele_2], Electron_pdg_mass);
        ele_3.SetPtEtaPhiM(Electron_pt[iele_3], Electron_eta[iele_3], Electron_phi[iele_3], Electron_pdg_mass);
        float DeltaM_A = 9999999999.;
        float DeltaM_B = 9999999999.;
        float DeltaM_C = 9999999999.;
        int position_A;
        int position_B;
        int position_C;
        if( (ele_charge_prod_01 < 0) && (ele_charge_prod_23 < 0) ){ 
            float DeltaM_01 = abs( (ele_0 + ele_1).M() - Z_pdg_mass );
            float DeltaM_23 = abs( (ele_2 + ele_3).M() - Z_pdg_mass );
            if( DeltaM_01 < DeltaM_23 ){
                DeltaM_A = DeltaM_01; 
                position_A = 1;
            }else{
                DeltaM_A = DeltaM_23; 
                position_A = 2;
            }
        }else if( (ele_charge_prod_02 < 0) && (ele_charge_prod_13 < 0) ){ 
            float DeltaM_02 = abs( (ele_0 + ele_2).M() - Z_pdg_mass );
            float DeltaM_13 = abs( (ele_1 + ele_3).M() - Z_pdg_mass );
            if( DeltaM_02 < DeltaM_13 ){
                DeltaM_B = DeltaM_02; 
                position_B = 1;
            }else{
                DeltaM_B = DeltaM_13; 
                position_B = 2;
            }
        }else if( (ele_charge_prod_03 < 0) && (ele_charge_prod_12 < 0) ){ 
            float DeltaM_03 = abs( (ele_0 + ele_3).M() - Z_pdg_mass );
            float DeltaM_12 = abs( (ele_1 + ele_2).M() - Z_pdg_mass );
            if( DeltaM_03 < DeltaM_12 ){
                DeltaM_C = DeltaM_03; 
                position_C = 1;
            }else{
                DeltaM_C = DeltaM_12; 
                position_C = 2;
            }
        }
        
        if( (DeltaM_A < DeltaM_B) && (DeltaM_A < DeltaM_C) ){
            if( position_A == 1 ){
                RecoLepID = 41101;                                                  // 2 Electrons OS & 2 Electrons OS
            }else if( position_A == 2 ){
                RecoLepID = 41123;                                                  // 2 Electrons OS & 2 Electrons OS
            }
        }else if( (DeltaM_B < DeltaM_A) && (DeltaM_B < DeltaM_C) ){
            if( position_B == 1 ){
                RecoLepID = 41102;                                                  // 2 Electrons OS & 2 Electrons OS
            }else if( position_B == 2 ){
                RecoLepID = 41113;                                                  // 2 Electrons OS & 2 Electrons OS
            }
        }else if( (DeltaM_C < DeltaM_A) && (DeltaM_C < DeltaM_B) ){
            if( position_C == 1 ){
                RecoLepID = 41103;                                                  // 2 Electrons OS & 2 Electrons OS
            }else if( position_C == 2 ){
                RecoLepID = 41112;                                                  // 2 Electrons OS & 2 Electrons OS
            }
        }
    }
    
    else if( (Nelectrons == 0) && (Nmuons == 4) ){
        int imu_0 = selectedMu.at(0);
        int imu_1 = selectedMu.at(1);
        int imu_2 = selectedMu.at(2);
        int imu_3 = selectedMu.at(3);
        int mu_charge_prod_01 = Muon_charge[imu_0]*Muon_charge[imu_1];
        int mu_charge_prod_02 = Muon_charge[imu_0]*Muon_charge[imu_2];
        int mu_charge_prod_03 = Muon_charge[imu_0]*Muon_charge[imu_3];
        int mu_charge_prod_12 = Muon_charge[imu_1]*Muon_charge[imu_2];
        int mu_charge_prod_13 = Muon_charge[imu_1]*Muon_charge[imu_3];
        int mu_charge_prod_23 = Muon_charge[imu_2]*Muon_charge[imu_3];
        TLorentzVector mu_0;
        TLorentzVector mu_1;
        TLorentzVector mu_2;
        TLorentzVector mu_3;
        mu_0.SetPtEtaPhiM(Muon_pt[imu_0], Muon_eta[imu_0], Muon_phi[imu_0], Muon_pdg_mass);
        mu_1.SetPtEtaPhiM(Muon_pt[imu_1], Muon_eta[imu_1], Muon_phi[imu_1], Muon_pdg_mass);
        mu_2.SetPtEtaPhiM(Muon_pt[imu_2], Muon_eta[imu_2], Muon_phi[imu_2], Muon_pdg_mass);
        mu_3.SetPtEtaPhiM(Muon_pt[imu_3], Muon_eta[imu_3], Muon_phi[imu_3], Muon_pdg_mass);
        float DeltaM_A = 9999999999.;
        float DeltaM_B = 9999999999.;
        float DeltaM_C = 9999999999.;
        int position_A;
        int position_B;
        int position_C;
        if( (mu_charge_prod_01 < 0) && (mu_charge_prod_23 < 0) ){ 
            float DeltaM_01 = abs( (mu_0 + mu_1).M() - Z_pdg_mass );
            float DeltaM_23 = abs( (mu_2 + mu_3).M() - Z_pdg_mass );
            if( DeltaM_01 < DeltaM_23 ){
                DeltaM_A = DeltaM_01; 
                position_A = 1;
            }else{
                DeltaM_A = DeltaM_23; 
                position_A = 2;
            }
        }else if( (mu_charge_prod_02 < 0) && (mu_charge_prod_13 < 0) ){ 
            float DeltaM_02 = abs( (mu_0 + mu_2).M() - Z_pdg_mass );
            float DeltaM_13 = abs( (mu_1 + mu_3).M() - Z_pdg_mass );
            if( DeltaM_02 < DeltaM_13 ){
                DeltaM_B = DeltaM_02; 
                position_B = 1;
            }else{
                DeltaM_B = DeltaM_13; 
                position_B = 2;
            }
        }else if( (mu_charge_prod_03 < 0) && (mu_charge_prod_12 < 0) ){ 
            float DeltaM_03 = abs( (mu_0 + mu_3).M() - Z_pdg_mass );
            float DeltaM_12 = abs( (mu_1 + mu_2).M() - Z_pdg_mass );
            if( DeltaM_03 < DeltaM_12 ){
                DeltaM_C = DeltaM_03; 
                position_C = 1;
            }else{
                DeltaM_C = DeltaM_12; 
                position_C = 2;
            }
        }
        
        if( (DeltaM_A < DeltaM_B) && (DeltaM_A < DeltaM_C) ){
            if( position_A == 1 ){
                RecoLepID = 41301;                                                  // 2 Muons OS & 2 Muons OS
            }else if( position_A == 2 ){
                RecoLepID = 41323;                                                  // 2 Muons OS & 2 Muons OS
            }
        }else if( (DeltaM_B < DeltaM_A) && (DeltaM_B < DeltaM_C) ){
            if( position_B == 1 ){
                RecoLepID = 41302;                                                  // 2 Muons OS & 2 Muons OS
            }else if( position_B == 2 ){
                RecoLepID = 41313;                                                  // 2 Muons OS & 2 Muons OS
            }
        }else if( (DeltaM_C < DeltaM_A) && (DeltaM_C < DeltaM_B) ){
            if( position_C == 1 ){
                RecoLepID = 41303;                                                  // 2 Muons OS & 2 Muons OS
            }else if( position_C == 2 ){
                RecoLepID = 41312;                                                  // 2 Muons OS & 2 Muons OS
            }
        }
    }
    
    
    
}


//---------------------------------------------------------------------------------------------------------------
// Get Leading and Trailing Leptons
//--------------------------------------------------------------------------------------------------------------- 
void HEPHero::Get_LeadingAndTrailing_Lepton_Variables(){  
    
    IdxLeadingLep = -1;
    IdxTrailingLep = -1;
    if( RecoLepID == 11 ){
        IdxLeadingLep = selectedEle.at(0);
        IdxTrailingLep = selectedEle.at(1);
        LeadingLep_pt = Electron_pt[IdxLeadingLep];
        LeadingLep_eta = Electron_eta[IdxLeadingLep];
        TrailingLep_pt = Electron_pt[IdxTrailingLep];
        TrailingLep_eta = Electron_eta[IdxTrailingLep];
        lep_1.SetPtEtaPhiM(Electron_pt[IdxLeadingLep], Electron_eta[IdxLeadingLep], Electron_phi[IdxLeadingLep], Electron_pdg_mass);
        lep_2.SetPtEtaPhiM(Electron_pt[IdxTrailingLep], Electron_eta[IdxTrailingLep], Electron_phi[IdxTrailingLep], Electron_pdg_mass);
    }
    else if( RecoLepID == 13 ){
        IdxLeadingLep = selectedMu.at(0);
        IdxTrailingLep = selectedMu.at(1);
        LeadingLep_pt = Muon_pt[IdxLeadingLep];
        LeadingLep_eta = Muon_eta[IdxLeadingLep];
        TrailingLep_pt = Muon_pt[IdxTrailingLep];
        TrailingLep_eta = Muon_eta[IdxTrailingLep];
        lep_1.SetPtEtaPhiM(Muon_pt[IdxLeadingLep], Muon_eta[IdxLeadingLep], Muon_phi[IdxLeadingLep], Muon_pdg_mass);
        lep_2.SetPtEtaPhiM(Muon_pt[IdxTrailingLep], Muon_eta[IdxTrailingLep], Muon_phi[IdxTrailingLep], Muon_pdg_mass);
    }
    else if( RecoLepID == 1113 ){
        IdxLeadingLep = selectedEle.at(0);
        IdxTrailingLep = selectedMu.at(0);
        LeadingLep_pt = Electron_pt[IdxLeadingLep];
        LeadingLep_eta = Electron_eta[IdxLeadingLep];
        TrailingLep_pt = Muon_pt[IdxTrailingLep];
        TrailingLep_eta = Muon_eta[IdxTrailingLep];
        lep_1.SetPtEtaPhiM(Electron_pt[IdxLeadingLep], Electron_eta[IdxLeadingLep], Electron_phi[IdxLeadingLep], Electron_pdg_mass);
        lep_2.SetPtEtaPhiM(Muon_pt[IdxTrailingLep], Muon_eta[IdxTrailingLep], Muon_phi[IdxTrailingLep], Muon_pdg_mass);
    }
    else if( RecoLepID == 1311 ){
        IdxLeadingLep = selectedMu.at(0);
        IdxTrailingLep = selectedEle.at(0);
        LeadingLep_pt = Muon_pt[IdxLeadingLep];
        LeadingLep_eta = Muon_eta[IdxLeadingLep];
        TrailingLep_pt = Electron_pt[IdxTrailingLep];
        TrailingLep_eta = Electron_eta[IdxTrailingLep];
        lep_1.SetPtEtaPhiM(Muon_pt[IdxLeadingLep], Muon_eta[IdxLeadingLep], Muon_phi[IdxLeadingLep], Muon_pdg_mass);
        lep_2.SetPtEtaPhiM(Electron_pt[IdxTrailingLep], Electron_eta[IdxTrailingLep], Electron_phi[IdxTrailingLep], Electron_pdg_mass);
    }
    else if( (RecoLepID == 31199) || (RecoLepID == 41199) || (RecoLepID == 31101) || (RecoLepID == 41101) ){
        IdxLeadingLep = selectedEle.at(0);
        IdxTrailingLep = selectedEle.at(1);
        LeadingLep_pt = Electron_pt[IdxLeadingLep];
        LeadingLep_eta = Electron_eta[IdxLeadingLep];
        TrailingLep_pt = Electron_pt[IdxTrailingLep];
        TrailingLep_eta = Electron_eta[IdxTrailingLep];
        lep_1.SetPtEtaPhiM(Electron_pt[IdxLeadingLep], Electron_eta[IdxLeadingLep], Electron_phi[IdxLeadingLep], Electron_pdg_mass);
        lep_2.SetPtEtaPhiM(Electron_pt[IdxTrailingLep], Electron_eta[IdxTrailingLep], Electron_phi[IdxTrailingLep], Electron_pdg_mass);
    }
    else if( (RecoLepID == 31399) || (RecoLepID == 41399) || (RecoLepID == 31301) || (RecoLepID == 41301) ){
        IdxLeadingLep = selectedMu.at(0);
        IdxTrailingLep = selectedMu.at(1);
        LeadingLep_pt = Muon_pt[IdxLeadingLep];
        LeadingLep_eta = Muon_eta[IdxLeadingLep];
        TrailingLep_pt = Muon_pt[IdxTrailingLep];
        TrailingLep_eta = Muon_eta[IdxTrailingLep];
        lep_1.SetPtEtaPhiM(Muon_pt[IdxLeadingLep], Muon_eta[IdxLeadingLep], Muon_phi[IdxLeadingLep], Muon_pdg_mass);
        lep_2.SetPtEtaPhiM(Muon_pt[IdxTrailingLep], Muon_eta[IdxTrailingLep], Muon_phi[IdxTrailingLep], Muon_pdg_mass);
    }
    else if( (RecoLepID == 31112) || (RecoLepID == 41112) ){
        IdxLeadingLep = selectedEle.at(1);
        IdxTrailingLep = selectedEle.at(2);
        LeadingLep_pt = Electron_pt[IdxLeadingLep];
        LeadingLep_eta = Electron_eta[IdxLeadingLep];
        TrailingLep_pt = Electron_pt[IdxTrailingLep];
        TrailingLep_eta = Electron_eta[IdxTrailingLep];
        lep_1.SetPtEtaPhiM(Electron_pt[IdxLeadingLep], Electron_eta[IdxLeadingLep], Electron_phi[IdxLeadingLep], Electron_pdg_mass);
        lep_2.SetPtEtaPhiM(Electron_pt[IdxTrailingLep], Electron_eta[IdxTrailingLep], Electron_phi[IdxTrailingLep], Electron_pdg_mass);
    }
    else if( (RecoLepID == 31312) || (RecoLepID == 41312) ){
        IdxLeadingLep = selectedMu.at(1);
        IdxTrailingLep = selectedMu.at(2);
        LeadingLep_pt = Muon_pt[IdxLeadingLep];
        LeadingLep_eta = Muon_eta[IdxLeadingLep];
        TrailingLep_pt = Muon_pt[IdxTrailingLep];
        TrailingLep_eta = Muon_eta[IdxTrailingLep];
        lep_1.SetPtEtaPhiM(Muon_pt[IdxLeadingLep], Muon_eta[IdxLeadingLep], Muon_phi[IdxLeadingLep], Muon_pdg_mass);
        lep_2.SetPtEtaPhiM(Muon_pt[IdxTrailingLep], Muon_eta[IdxTrailingLep], Muon_phi[IdxTrailingLep], Muon_pdg_mass);
    }
    else if( (RecoLepID == 31120) || (RecoLepID == 41102) ){
        IdxLeadingLep = selectedEle.at(0);
        IdxTrailingLep = selectedEle.at(2);
        LeadingLep_pt = Electron_pt[IdxLeadingLep];
        LeadingLep_eta = Electron_eta[IdxLeadingLep];
        TrailingLep_pt = Electron_pt[IdxTrailingLep];
        TrailingLep_eta = Electron_eta[IdxTrailingLep];
        lep_1.SetPtEtaPhiM(Electron_pt[IdxLeadingLep], Electron_eta[IdxLeadingLep], Electron_phi[IdxLeadingLep], Electron_pdg_mass);
        lep_2.SetPtEtaPhiM(Electron_pt[IdxTrailingLep], Electron_eta[IdxTrailingLep], Electron_phi[IdxTrailingLep], Electron_pdg_mass);
    }
    else if( (RecoLepID == 31320) || (RecoLepID == 41302) ){
        IdxLeadingLep = selectedMu.at(0);
        IdxTrailingLep = selectedMu.at(2);
        LeadingLep_pt = Muon_pt[IdxLeadingLep];
        LeadingLep_eta = Muon_eta[IdxLeadingLep];
        TrailingLep_pt = Muon_pt[IdxTrailingLep];
        TrailingLep_eta = Muon_eta[IdxTrailingLep];
        lep_1.SetPtEtaPhiM(Muon_pt[IdxLeadingLep], Muon_eta[IdxLeadingLep], Muon_phi[IdxLeadingLep], Muon_pdg_mass);
        lep_2.SetPtEtaPhiM(Muon_pt[IdxTrailingLep], Muon_eta[IdxTrailingLep], Muon_phi[IdxTrailingLep], Muon_pdg_mass);
    }
    else if( RecoLepID == 41103 ){
        IdxLeadingLep = selectedEle.at(0);
        IdxTrailingLep = selectedEle.at(3);
        LeadingLep_pt = Electron_pt[IdxLeadingLep];
        LeadingLep_eta = Electron_eta[IdxLeadingLep];
        TrailingLep_pt = Electron_pt[IdxTrailingLep];
        TrailingLep_eta = Electron_eta[IdxTrailingLep];
        lep_1.SetPtEtaPhiM(Electron_pt[IdxLeadingLep], Electron_eta[IdxLeadingLep], Electron_phi[IdxLeadingLep], Electron_pdg_mass);
        lep_2.SetPtEtaPhiM(Electron_pt[IdxTrailingLep], Electron_eta[IdxTrailingLep], Electron_phi[IdxTrailingLep], Electron_pdg_mass);
    }
    else if( RecoLepID == 41303 ){
        IdxLeadingLep = selectedMu.at(0);
        IdxTrailingLep = selectedMu.at(3);
        LeadingLep_pt = Muon_pt[IdxLeadingLep];
        LeadingLep_eta = Muon_eta[IdxLeadingLep];
        TrailingLep_pt = Muon_pt[IdxTrailingLep];
        TrailingLep_eta = Muon_eta[IdxTrailingLep];
        lep_1.SetPtEtaPhiM(Muon_pt[IdxLeadingLep], Muon_eta[IdxLeadingLep], Muon_phi[IdxLeadingLep], Muon_pdg_mass);
        lep_2.SetPtEtaPhiM(Muon_pt[IdxTrailingLep], Muon_eta[IdxTrailingLep], Muon_phi[IdxTrailingLep], Muon_pdg_mass);
    }
    else if( RecoLepID == 41113 ){
        IdxLeadingLep = selectedEle.at(1);
        IdxTrailingLep = selectedEle.at(3);
        LeadingLep_pt = Electron_pt[IdxLeadingLep];
        LeadingLep_eta = Electron_eta[IdxLeadingLep];
        TrailingLep_pt = Electron_pt[IdxTrailingLep];
        TrailingLep_eta = Electron_eta[IdxTrailingLep];
        lep_1.SetPtEtaPhiM(Electron_pt[IdxLeadingLep], Electron_eta[IdxLeadingLep], Electron_phi[IdxLeadingLep], Electron_pdg_mass);
        lep_2.SetPtEtaPhiM(Electron_pt[IdxTrailingLep], Electron_eta[IdxTrailingLep], Electron_phi[IdxTrailingLep], Electron_pdg_mass);
    }
    else if( RecoLepID == 41313 ){
        IdxLeadingLep = selectedMu.at(1);
        IdxTrailingLep = selectedMu.at(3);
        LeadingLep_pt = Muon_pt[IdxLeadingLep];
        LeadingLep_eta = Muon_eta[IdxLeadingLep];
        TrailingLep_pt = Muon_pt[IdxTrailingLep];
        TrailingLep_eta = Muon_eta[IdxTrailingLep];
        lep_1.SetPtEtaPhiM(Muon_pt[IdxLeadingLep], Muon_eta[IdxLeadingLep], Muon_phi[IdxLeadingLep], Muon_pdg_mass);
        lep_2.SetPtEtaPhiM(Muon_pt[IdxTrailingLep], Muon_eta[IdxTrailingLep], Muon_phi[IdxTrailingLep], Muon_pdg_mass);
    }
    else if( RecoLepID == 41123 ){
        IdxLeadingLep = selectedEle.at(2);
        IdxTrailingLep = selectedEle.at(3);
        LeadingLep_pt = Electron_pt[IdxLeadingLep];
        LeadingLep_eta = Electron_eta[IdxLeadingLep];
        TrailingLep_pt = Electron_pt[IdxTrailingLep];
        TrailingLep_eta = Electron_eta[IdxTrailingLep];
        lep_1.SetPtEtaPhiM(Electron_pt[IdxLeadingLep], Electron_eta[IdxLeadingLep], Electron_phi[IdxLeadingLep], Electron_pdg_mass);
        lep_2.SetPtEtaPhiM(Electron_pt[IdxTrailingLep], Electron_eta[IdxTrailingLep], Electron_phi[IdxTrailingLep], Electron_pdg_mass);
    }
    else if( RecoLepID == 41323 ){
        IdxLeadingLep = selectedMu.at(2);
        IdxTrailingLep = selectedMu.at(3);
        LeadingLep_pt = Muon_pt[IdxLeadingLep];
        LeadingLep_eta = Muon_eta[IdxLeadingLep];
        TrailingLep_pt = Muon_pt[IdxTrailingLep];
        TrailingLep_eta = Muon_eta[IdxTrailingLep];
        lep_1.SetPtEtaPhiM(Muon_pt[IdxLeadingLep], Muon_eta[IdxLeadingLep], Muon_phi[IdxLeadingLep], Muon_pdg_mass);
        lep_2.SetPtEtaPhiM(Muon_pt[IdxTrailingLep], Muon_eta[IdxTrailingLep], Muon_phi[IdxTrailingLep], Muon_pdg_mass);
    } 
    
}


//---------------------------------------------------------------------------------------------------------------
// Get Third and Fourth Leptons
//--------------------------------------------------------------------------------------------------------------- 
void HEPHero::Get_ThirdAndFourth_Lepton_Variables(){  
    
    IdxThirdLep = -1;
    IdxFourthLep = -1;
    
    //-----------------------------------------------------------------------------------
    // 3 Leptons
    //-----------------------------------------------------------------------------------
    if( RecoLepID == 31199 ){
        IdxThirdLep = selectedMu.at(0);
        lep_3.SetPtEtaPhiM(Muon_pt[IdxThirdLep], Muon_eta[IdxThirdLep], Muon_phi[IdxThirdLep], Muon_pdg_mass);
    }
    else if( RecoLepID == 31101 ){
        IdxThirdLep = selectedEle.at(2);
        lep_3.SetPtEtaPhiM(Electron_pt[IdxThirdLep], Electron_eta[IdxThirdLep], Electron_phi[IdxThirdLep], Electron_pdg_mass);
    }
    else if( RecoLepID == 31112 ){
        IdxThirdLep = selectedEle.at(0);
        lep_3.SetPtEtaPhiM(Electron_pt[IdxThirdLep], Electron_eta[IdxThirdLep], Electron_phi[IdxThirdLep], Electron_pdg_mass);
    }
    else if( RecoLepID == 31120 ){
        IdxThirdLep = selectedEle.at(1);
        lep_3.SetPtEtaPhiM(Electron_pt[IdxThirdLep], Electron_eta[IdxThirdLep], Electron_phi[IdxThirdLep], Electron_pdg_mass);
    }
    else if( RecoLepID == 31399 ){
        IdxThirdLep = selectedEle.at(0);
        lep_3.SetPtEtaPhiM(Electron_pt[IdxThirdLep], Electron_eta[IdxThirdLep], Electron_phi[IdxThirdLep], Electron_pdg_mass);
    }
    else if( RecoLepID == 31301 ){
        IdxThirdLep = selectedMu.at(2);
        lep_3.SetPtEtaPhiM(Muon_pt[IdxThirdLep], Muon_eta[IdxThirdLep], Muon_phi[IdxThirdLep], Muon_pdg_mass);
    }
    else if( RecoLepID == 31312 ){
        IdxThirdLep = selectedMu.at(0);
        lep_3.SetPtEtaPhiM(Muon_pt[IdxThirdLep], Muon_eta[IdxThirdLep], Muon_phi[IdxThirdLep], Muon_pdg_mass);
    }
    else if( RecoLepID == 31320 ){
        IdxThirdLep = selectedMu.at(1);
        lep_3.SetPtEtaPhiM(Muon_pt[IdxThirdLep], Muon_eta[IdxThirdLep], Muon_phi[IdxThirdLep], Muon_pdg_mass);
    }
    //-----------------------------------------------------------------------------------
    // 4 Leptons
    //-----------------------------------------------------------------------------------
    else if( RecoLepID == 41199 ){
        IdxThirdLep = selectedMu.at(0);
        IdxFourthLep = selectedMu.at(1);
        lep_3.SetPtEtaPhiM(Muon_pt[IdxThirdLep], Muon_eta[IdxThirdLep], Muon_phi[IdxThirdLep], Muon_pdg_mass);
        lep_4.SetPtEtaPhiM(Muon_pt[IdxFourthLep], Muon_eta[IdxFourthLep], Muon_phi[IdxFourthLep], Muon_pdg_mass);
    }
    else if( RecoLepID == 41101 ){
        IdxThirdLep = selectedEle.at(2);
        IdxFourthLep = selectedEle.at(3);
        lep_3.SetPtEtaPhiM(Electron_pt[IdxThirdLep], Electron_eta[IdxThirdLep], Electron_phi[IdxThirdLep], Electron_pdg_mass);
        lep_4.SetPtEtaPhiM(Electron_pt[IdxFourthLep], Electron_eta[IdxFourthLep], Electron_phi[IdxFourthLep], Electron_pdg_mass);
    }
    else if( RecoLepID == 41102 ){
        IdxThirdLep = selectedEle.at(1);
        IdxFourthLep = selectedEle.at(3);
        lep_3.SetPtEtaPhiM(Electron_pt[IdxThirdLep], Electron_eta[IdxThirdLep], Electron_phi[IdxThirdLep], Electron_pdg_mass);
        lep_4.SetPtEtaPhiM(Electron_pt[IdxFourthLep], Electron_eta[IdxFourthLep], Electron_phi[IdxFourthLep], Electron_pdg_mass);
    }
    else if( RecoLepID == 41103 ){
        IdxThirdLep = selectedEle.at(1);
        IdxFourthLep = selectedEle.at(2);
        lep_3.SetPtEtaPhiM(Electron_pt[IdxThirdLep], Electron_eta[IdxThirdLep], Electron_phi[IdxThirdLep], Electron_pdg_mass);
        lep_4.SetPtEtaPhiM(Electron_pt[IdxFourthLep], Electron_eta[IdxFourthLep], Electron_phi[IdxFourthLep], Electron_pdg_mass);
    }
    else if( RecoLepID == 41112 ){
        IdxThirdLep = selectedEle.at(0);
        IdxFourthLep = selectedEle.at(3);
        lep_3.SetPtEtaPhiM(Electron_pt[IdxThirdLep], Electron_eta[IdxThirdLep], Electron_phi[IdxThirdLep], Electron_pdg_mass);
        lep_4.SetPtEtaPhiM(Electron_pt[IdxFourthLep], Electron_eta[IdxFourthLep], Electron_phi[IdxFourthLep], Electron_pdg_mass);
    }
    else if( RecoLepID == 41113 ){
        IdxThirdLep = selectedEle.at(0);
        IdxFourthLep = selectedEle.at(2);
        lep_3.SetPtEtaPhiM(Electron_pt[IdxThirdLep], Electron_eta[IdxThirdLep], Electron_phi[IdxThirdLep], Electron_pdg_mass);
        lep_4.SetPtEtaPhiM(Electron_pt[IdxFourthLep], Electron_eta[IdxFourthLep], Electron_phi[IdxFourthLep], Electron_pdg_mass);
    }
    else if( RecoLepID == 41123 ){
        IdxThirdLep = selectedEle.at(0);
        IdxFourthLep = selectedEle.at(1);
        lep_3.SetPtEtaPhiM(Electron_pt[IdxThirdLep], Electron_eta[IdxThirdLep], Electron_phi[IdxThirdLep], Electron_pdg_mass);
        lep_4.SetPtEtaPhiM(Electron_pt[IdxFourthLep], Electron_eta[IdxFourthLep], Electron_phi[IdxFourthLep], Electron_pdg_mass);
    }
    else if( RecoLepID == 41399 ){
        IdxThirdLep = selectedEle.at(0);
        IdxFourthLep = selectedEle.at(1);
        lep_3.SetPtEtaPhiM(Electron_pt[IdxThirdLep], Electron_eta[IdxThirdLep], Electron_phi[IdxThirdLep], Electron_pdg_mass);
        lep_4.SetPtEtaPhiM(Electron_pt[IdxFourthLep], Electron_eta[IdxFourthLep], Electron_phi[IdxFourthLep], Electron_pdg_mass);
    }
    else if( RecoLepID == 41301 ){
        IdxThirdLep = selectedMu.at(2);
        IdxFourthLep = selectedMu.at(3);
        lep_3.SetPtEtaPhiM(Muon_pt[IdxThirdLep], Muon_eta[IdxThirdLep], Muon_phi[IdxThirdLep], Muon_pdg_mass);
        lep_4.SetPtEtaPhiM(Muon_pt[IdxFourthLep], Muon_eta[IdxFourthLep], Muon_phi[IdxFourthLep], Muon_pdg_mass);
    }
    else if( RecoLepID == 41302 ){
        IdxThirdLep = selectedMu.at(1);
        IdxFourthLep = selectedMu.at(3);
        lep_3.SetPtEtaPhiM(Muon_pt[IdxThirdLep], Muon_eta[IdxThirdLep], Muon_phi[IdxThirdLep], Muon_pdg_mass);
        lep_4.SetPtEtaPhiM(Muon_pt[IdxFourthLep], Muon_eta[IdxFourthLep], Muon_phi[IdxFourthLep], Muon_pdg_mass);
    }
    else if( RecoLepID == 41303 ){
        IdxThirdLep = selectedMu.at(1);
        IdxFourthLep = selectedMu.at(2);
        lep_3.SetPtEtaPhiM(Muon_pt[IdxThirdLep], Muon_eta[IdxThirdLep], Muon_phi[IdxThirdLep], Muon_pdg_mass);
        lep_4.SetPtEtaPhiM(Muon_pt[IdxFourthLep], Muon_eta[IdxFourthLep], Muon_phi[IdxFourthLep], Muon_pdg_mass);
    }
    else if( RecoLepID == 41312 ){
        IdxThirdLep = selectedMu.at(0);
        IdxFourthLep = selectedMu.at(3);
        lep_3.SetPtEtaPhiM(Muon_pt[IdxThirdLep], Muon_eta[IdxThirdLep], Muon_phi[IdxThirdLep], Muon_pdg_mass);
        lep_4.SetPtEtaPhiM(Muon_pt[IdxFourthLep], Muon_eta[IdxFourthLep], Muon_phi[IdxFourthLep], Muon_pdg_mass);
    }
    else if( RecoLepID == 41313 ){
        IdxThirdLep = selectedMu.at(0);
        IdxFourthLep = selectedMu.at(2);
        lep_3.SetPtEtaPhiM(Muon_pt[IdxThirdLep], Muon_eta[IdxThirdLep], Muon_phi[IdxThirdLep], Muon_pdg_mass);
        lep_4.SetPtEtaPhiM(Muon_pt[IdxFourthLep], Muon_eta[IdxFourthLep], Muon_phi[IdxFourthLep], Muon_pdg_mass);
    }
    else if( RecoLepID == 41323 ){
        IdxThirdLep = selectedMu.at(0);
        IdxFourthLep = selectedMu.at(1);
        lep_3.SetPtEtaPhiM(Muon_pt[IdxThirdLep], Muon_eta[IdxThirdLep], Muon_phi[IdxThirdLep], Muon_pdg_mass);
        lep_4.SetPtEtaPhiM(Muon_pt[IdxFourthLep], Muon_eta[IdxFourthLep], Muon_phi[IdxFourthLep], Muon_pdg_mass);
    }
     
}


//---------------------------------------------------------------------------------------------------------------
// Get Trailing Lepton, and calculate LepLep variables
//--------------------------------------------------------------------------------------------------------------- 
void HEPHero::Get_Leptonic_Info( bool get12, bool get34){  
    
    if( get12 ) Get_LeadingAndTrailing_Lepton_Variables();
    
    if( get34 ) Get_ThirdAndFourth_Lepton_Variables();
    
}


//---------------------------------------------------------------------------------------------------------------
// Get Trailing Lepton, and calculate LepLep variables
//--------------------------------------------------------------------------------------------------------------- 
void HEPHero::Get_LepLep_Variables( bool get12, bool get34){  
    
    if( get12 ){
        Get_LeadingAndTrailing_Lepton_Variables();
        
        //float Delta_eta = abs( lep_1.Eta() - lep_2.Eta() );
        //float Delta_phi = abs( lep_1.Phi() - lep_2.Phi() );
        //if( Delta_phi > M_PI ) Delta_phi = 2*M_PI - Delta_phi;
        //LepLep_deltaR = sqrt( pow(Delta_phi,2) + pow(Delta_eta,2) );
        LepLep_deltaR = lep_1.DeltaR( lep_2 );

        LepLep = lep_1 + lep_2;
        LepLep_mass = LepLep.M();
        LepLep_deltaM = abs( LepLep_mass - Z_pdg_mass );
        LepLep_pt = LepLep.Pt();
        LepLep_phi = LepLep.Phi();
        LepLep_eta = LepLep.Eta();
        
        MET_LepLep_deltaPhi = abs( LepLep_phi - MET_phi );
        if( MET_LepLep_deltaPhi > M_PI ) MET_LepLep_deltaPhi = 2*M_PI - MET_LepLep_deltaPhi;
        
        MET_LepLep_deltaPt = abs(LepLep_pt - MET_pt)/LepLep_pt;
        
        MET_LepLep_Mt = sqrt( 2 * LepLep_pt * MET_pt * ( 1 - cos( MET_LepLep_deltaPhi ) ) ) ;
        
        Lep1_tight = false;
        Lep2_tight = false;
        if( (RecoLepID == 11) || ((RecoLepID >= 31100) && (RecoLepID <= 31199)) || ((RecoLepID >= 41100) && (RecoLepID <= 41199)) ){
            Lep1_tight = Electron_mvaFall17V2Iso_WP80[IdxLeadingLep];
            Lep2_tight = Electron_mvaFall17V2Iso_WP80[IdxTrailingLep];
        }else if( (RecoLepID == 13) || ((RecoLepID >= 31300) && (RecoLepID <= 31399)) || ((RecoLepID >= 41300) && (RecoLepID <= 41399)) ){
            Lep1_tight = Muon_tightId[IdxLeadingLep];
            Lep2_tight = Muon_tightId[IdxTrailingLep];
        }else if( RecoLepID == 1113 ){
            Lep1_tight = Electron_mvaFall17V2Iso_WP80[IdxLeadingLep];
            Lep2_tight = Muon_tightId[IdxTrailingLep];
        }else if( RecoLepID == 1311 ){
            Lep1_tight = Muon_tightId[IdxLeadingLep];
            Lep2_tight = Electron_mvaFall17V2Iso_WP80[IdxTrailingLep];
        }
    }
    
    if( get34 ){
        if( !get12 ) Get_LeadingAndTrailing_Lepton_Variables();
        Get_ThirdAndFourth_Lepton_Variables();
        
        LepLep_Lep3_M = -1;
        LepLep_Lep3_deltaR = -1;
        LepLep_Lep3_deltaPhi = -1;
        LepLep_Lep3_deltaPt = -1;
        MET_Lep3_deltaPhi = -1;
        MET_Lep3_Mt = -1;
        Lep3Lep4_deltaM = -1;
        Lep3Lep4_M = -1;
        Lep3Lep4_pt = -1;
        Lep3Lep4_phi = -1;
        Lep3Lep4_deltaR = -1;
        Lep3_pt = -1;
        Lep3_tight = false;
        Lep4_tight = false;
        Lep4_pt = -1;
        Lep3_Jet_deltaR = -1;
        Lep3_dxy = -1;
        Lep3_dz = -1;
        
        if( (RecoLepID > 30000) && (RecoLepID < 39999) ){
            TLorentzVector LepLep_Lep3;
            LepLep_Lep3 = LepLep + lep_3;
            LepLep_Lep3_M = LepLep_Lep3.M();
            LepLep_Lep3_deltaR = LepLep.DeltaR( -lep_3 );
            Lep3_pt = lep_3.Pt();
            float Lep3_phi = lep_3.Phi();
            float Lep3_eta = lep_3.Eta();
            
            LepLep_Lep3_deltaPhi = abs( LepLep_phi - Lep3_phi );
            if( LepLep_Lep3_deltaPhi > M_PI ) LepLep_Lep3_deltaPhi = 2*M_PI - LepLep_Lep3_deltaPhi;
            
            LepLep_Lep3_deltaPt = abs(LepLep_pt - Lep3_pt)/LepLep_pt;
            
            MET_Lep3_deltaPhi = abs( Lep3_phi - MET_JER_phi );
            if( MET_Lep3_deltaPhi > M_PI ) MET_Lep3_deltaPhi = 2*M_PI - MET_Lep3_deltaPhi;
            
            MET_Lep3_Mt = sqrt( 2 * Lep3_pt * MET_JER_pt * ( 1 - cos( MET_Lep3_deltaPhi ) ) ) ;
            
            if( (RecoLepID == 31399) || ((RecoLepID > 31100) && (RecoLepID < 31199)) ){
                Lep3_tight = Electron_mvaFall17V2Iso_WP80[IdxThirdLep];
                Lep3_dxy = Electron_dxy[IdxThirdLep];
                Lep3_dz = Electron_dz[IdxThirdLep];
            }else if( (RecoLepID == 31199) || ((RecoLepID > 31300) && (RecoLepID < 31399)) ){
                Lep3_tight = Muon_tightId[IdxThirdLep];
                Lep3_dxy = Muon_dxy[IdxThirdLep];
                Lep3_dz = Muon_dz[IdxThirdLep];
            }
            
            Lep3_Jet_deltaR = 99;
            for( unsigned int ijet = 0; ijet < nJet; ++ijet ) {
                if( Jet_pt[ijet] <= 20 ) continue;
                double deta = fabs(Lep3_eta - Jet_eta[ijet]);
                double dphi = fabs(Lep3_phi - Jet_phi[ijet]);
                if( dphi > M_PI ) dphi = 2*M_PI - dphi;
                double dr = sqrt( deta*deta + dphi*dphi );
                if( dr < Lep3_Jet_deltaR ) Lep3_Jet_deltaR = dr;
            }
            
        }else if( (RecoLepID > 40000) && (RecoLepID < 49999) ){
            TLorentzVector Lep3Lep4;
            Lep3Lep4 = lep_3 + lep_4;
            Lep3Lep4_deltaM = abs( Lep3Lep4.M() - Z_pdg_mass );
            Lep3Lep4_M = Lep3Lep4.M();
            Lep3Lep4_pt = Lep3Lep4.Pt();
            Lep3Lep4_phi = Lep3Lep4.Phi();
            Lep3Lep4_deltaR = lep_3.DeltaR( lep_4 );
            Lep3_pt = lep_3.Pt();
            Lep4_pt = lep_4.Pt();
            if( (RecoLepID == 41399) || ((RecoLepID > 41100) && (RecoLepID < 41199)) ){
                Lep3_tight = Electron_mvaFall17V2Iso_WP80[IdxThirdLep];
                Lep4_tight = Electron_mvaFall17V2Iso_WP80[IdxFourthLep];
            }else if( (RecoLepID == 41199) || ((RecoLepID > 41300) && (RecoLepID < 41399)) ){
                Lep3_tight = Muon_tightId[IdxThirdLep];
                Lep4_tight = Muon_tightId[IdxFourthLep];
            }
        }
    }
    
}
    
 
//---------------------------------------------------------------------------------------------------------------
// Calculate ttbar variables
//---------------------------------------------------------------------------------------------------------------    
void HEPHero::Get_ttbar_Variables(){  
    
    asymm_mt2_lester_bisect::disableCopyrightMessage();
    //MT2LL = asymm_mt2_lester_bisect::get_mT2( mVisA, pxA, pyA, mVisB, pxB, pyB, pxMiss, pyMiss, chiA, chiB, desiredPrecisionOnMt2);
    MT2LL = asymm_mt2_lester_bisect::get_mT2( lep_1.M(), lep_1.Px(), lep_1.Py(), lep_2.M(), lep_2.Px(), lep_2.Py(), MET_pt*cos(MET_phi), MET_pt*sin(MET_phi), 0, 0, 0);
    
    //------------------------------------------------------------------------------------------------
    // ttbar reconstruction using Sonnenschein's analytical solution
    // https://arxiv.org/abs/hep-ph/0603011
    vector<TLorentzVector> vecJets;
    for( unsigned int iseljet = 0; iseljet < selectedJet.size(); ++iseljet ) {
        int ijet = selectedJet.at(iseljet);
        TLorentzVector vecJet;
        vecJet.SetPtEtaPhiM(Jet_pt[ijet], Jet_eta[ijet], Jet_phi[ijet], Jet_mass[ijet]);
        if( JetBTAG( ijet, JET_BTAG_WP ) ){ 
            vecJet.SetPtEtaPhiM(Jet_pt[ijet], Jet_eta[ijet], Jet_phi[ijet], -1*Jet_mass[ijet]);
        }
        vecJets.push_back(vecJet);
    } 
    
    TLorentzVector t;
    TLorentzVector tbar;
    TH1D* hInacc = new TH1D("hInacc", "KinReco inaccuracy", 1000, 0.0, 100.0);
    TH1D* hAmbig = new TH1D("hAmbig", "KinReco ambiguity", 100, 0.0, 100.0);
    
    ttbar_reco = 0;
    ttbar_score = 0.;
    ttbar_mass = 0.;
    ttbar_reco_v2 = 0;
    ttbar_score_v2 = 0.;
    ttbar_mass_v2 = 0.;
    
    
    if(selectedJet.size() >= 2){
        ttbar_reco = KinRecoDilepton(lep_1, lep_2, vecJets, MET_pt*cos(MET_phi), MET_pt*sin(MET_phi), t, tbar, ttbar_score, hInacc, hAmbig);
        ttbar_mass = (t+tbar).M();
    }
    
    
    // ---------- DESY  ttbar_2016 code ----------------------------
    if(selectedJet.size() >= 2){
        KinematicReconstruction myTTbarObject;
        TLorentzVector met;
        met.SetPxPyPzE ( MET_pt*cos(MET_phi) , MET_pt*sin(MET_phi) , 0., 0.);
        myTTbarObject.kinReco( lep_1, lep_2, vecJets, met ) ; // To turn on/off the smearing process, you have to change the NoSmearingFlag in the "KinematicReconstruction.h". Remember that using smeatinf process, you need a root file with infotmation about the resolutions.
        if ( myTTbarObject.getNSol() > 0 ){
            ttbar_reco_v2 = 1;
            ttbar_score_v2 = myTTbarObject.getWeight( );
            t =  myTTbarObject.getLorentzVector( )[0];
            tbar =  myTTbarObject.getLorentzVector( )[1];
            ttbar_mass_v2 = (t+tbar).M();
        }
    }
    
    
    
    
    
}


//---------------------------------------------------------------------------------------------------------------
// Calculate Jet variables
//--------------------------------------------------------------------------------------------------------------- 
void HEPHero::Get_Dijet_Variables(){
    
    Dijet_pt = 0; 
    Dijet_M = 0;
    Dijet_deltaEta = 0;
    if( selectedJet.size() == 1 ){
        int ijet = selectedJet.at(0);
        Dijet_pt = Jet_pt[ijet];
        Dijet_M = Jet_mass[ijet];
        Dijet_deltaEta = 0;
    }else if( selectedJet.size() >= 2 ){
        int btagFlag;
        if(      JET_BTAG_WP >= 0 and JET_BTAG_WP <= 2 ) btagFlag = 1;
        else if( JET_BTAG_WP >= 3 and JET_BTAG_WP <= 5 ) btagFlag = 0;
        vector<int> Dijet_indexes = findLeadingAndTrailingBJets(btagFlag);
        int ijet1 = Dijet_indexes.at(0);
        int ijet2 = Dijet_indexes.at(1);
        TLorentzVector jet_1;
        TLorentzVector jet_2;
        jet_1.SetPtEtaPhiM(Jet_pt[ijet1], Jet_eta[ijet1], Jet_phi[ijet1], Jet_mass[ijet1]);
        jet_2.SetPtEtaPhiM(Jet_pt[ijet2], Jet_eta[ijet2], Jet_phi[ijet2], Jet_mass[ijet2]);
        Dijet = jet_1 + jet_2;
        Dijet_pt = Dijet.Pt();
        Dijet_M = Dijet.M();
        Dijet_deltaEta = abs(Jet_eta[ijet1] - Jet_eta[ijet2]);
        
        MET.SetPxPyPzE(MET_pt*cos(MET_phi), MET_pt*sin(MET_phi), 0, MET_pt);
        TLorentzVector H = MET + LepLep;
        Dijet_H_deltaPhi = abs( H.Phi() - Dijet.Phi() );
        if( Dijet_H_deltaPhi > M_PI ) Dijet_H_deltaPhi = 2*M_PI - Dijet_H_deltaPhi;
        TLorentzVector Dijet_H = H + Dijet;
        Dijet_H_pt = Dijet_H.Pt();
    }
}


//---------------------------------------------------------------------------------------------------------------
// Return Truth lepton ID of the leptons produced in the Z boson decay
//--------------------------------------------------------------------------------------------------------------- 
int HEPHero::TruthLepID(){
    int LepID = 0;
    if( dataset_group == "Signal" ) {
        for( unsigned int ipart = 0; ipart < nGenPart; ++ipart ) {
            if( (abs(GenPart_pdgId[ipart]) == 11) && (GenPart_pdgId[GenPart_genPartIdxMother[ipart]] == 23) ){
                LepID = 11;
                break;
            }else if( (abs(GenPart_pdgId[ipart]) == 13) && (GenPart_pdgId[GenPart_genPartIdxMother[ipart]] == 23) ){
                LepID = 13;
                break;
            }else if( (abs(GenPart_pdgId[ipart]) == 15) && (GenPart_pdgId[GenPart_genPartIdxMother[ipart]] == 23) ){
                LepID = 15;
                break;
            }
        }
    }
    
    return LepID;
}


//-------------------------------------------------------------------------
// Trigger selection
//-------------------------------------------------------------------------
bool HEPHero::Trigger(){
    bool triggered = false;
        
    // https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgHLTRunIISummary (electron triggers recommendation)
    
    //=====TRIGGERS FOR 2016=====================================================================
    if( dataset_year == "16" ){
        
        HLT_SingleEle = HLT_Ele27_WPTight_Gsf || HLT_Ele115_CaloIdVT_GsfTrkIdT;// || HLT_Photon175;
        
        // || HLT_Photon175_v

        //if( (dataset_group == "Data") && (run >= 276453) && (run <= 278822) ){
        //    HLT_DoubleEle = HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ || HLT_DoubleEle33_CaloIdL_MW || HLT_DoubleEle33_CaloIdL_GsfTrkIdVL;
        //}else{
        //    HLT_DoubleEle = HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ || HLT_DoubleEle33_CaloIdL_MW; // HLT_DoubleEle37_Ele27_CaloIdL_GsfTrkIdVL;
        //}
        HLT_DoubleEle = HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ;
        
        //HLT_DoubleEle33_CaloIdL_MW_v (ORed with HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_v in the run range [276453, 278822] to compensate the prescaling of the former)
            
        HLT_SingleMu = HLT_IsoMu24 || HLT_IsoTkMu24 || HLT_Mu50 || HLT_TkMu50;
            
        HLT_DoubleMu = HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL || HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL || HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ || HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ;

        HLT_EleMu = HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL || HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL || HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ || HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ;

        /*
        if( (dataset_group == "Data") && (dataset_era == "H") ){
            HLT_DoubleMu = HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ || HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ;
            
            HLT_EleMu = HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ || HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ;
        }else{
            HLT_DoubleMu = HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL || HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL;
            //HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ || HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ


            HLT_EleMu = HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL || HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL;
        }
        */
        
        HLT_MET = HLT_PFMET300 || HLT_MET200 || HLT_PFHT300_PFMET110 || HLT_PFMET170_HBHECleaned || HLT_PFMET120_PFMHT120_IDTight || HLT_PFMETNoMu120_PFMHTNoMu120_IDTight;
    }
    
    //=====TRIGGERS FOR 2017=====================================================================
    if( dataset_year == "17" ){
        
        HLT_SingleEle = HLT_Ele35_WPTight_Gsf;// || HLT_Ele115_CaloIdVT_GsfTrkIdT || HLT_Photon200; 
        
        // HLT_Ele115_CaloIdVT_GsfTrkIdT_v || HLT_Photon200_v

        HLT_DoubleEle = HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL || HLT_DoubleEle33_CaloIdL_MW;
            
        HLT_SingleMu = HLT_IsoMu27 || HLT_Mu50 || HLT_OldMu100 || HLT_TkMu100;
            
        HLT_DoubleMu = HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8 || HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8;
        
        HLT_EleMu = HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ || HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ;
            
        HLT_MET = HLT_PFMET200_HBHECleaned || HLT_PFMET200_HBHE_BeamHaloCleaned || HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned || HLT_PFMET120_PFMHT120_IDTight || HLT_PFMET120_PFMHT120_IDTight_PFHT60 || HLT_PFMETNoMu120_PFMHTNoMu120_IDTight || HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60 || HLT_PFHT500_PFMET100_PFMHT100_IDTight || HLT_PFHT700_PFMET85_PFMHT85_IDTight || HLT_PFHT800_PFMET75_PFMHT75_IDTight;
    }
    
    //=====TRIGGERS FOR 2018=====================================================================
    if( dataset_year == "18" ){
        
        HLT_SingleEle = HLT_Ele32_WPTight_Gsf || HLT_Ele115_CaloIdVT_GsfTrkIdT;

        HLT_DoubleEle = HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL;// || HLT_DoubleEle25_CaloIdL_MW;  
        
        // HLT_DoubleEle25_CaloIdL_MW
            
        HLT_SingleMu = HLT_IsoMu24 || HLT_Mu50 || HLT_OldMu100 || HLT_TkMu100;
            
        HLT_DoubleMu = HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8;
            
        HLT_EleMu = HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ || HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL;
                
        HLT_MET = HLT_PFMET200_HBHECleaned || HLT_PFMET200_HBHE_BeamHaloCleaned || HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned || HLT_PFMET120_PFMHT120_IDTight || HLT_PFMET120_PFMHT120_IDTight_PFHT60 || HLT_PFMETNoMu120_PFMHTNoMu120_IDTight || HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60 || HLT_PFHT500_PFMET100_PFMHT100_IDTight || HLT_PFHT700_PFMET85_PFMHT85_IDTight || HLT_PFHT800_PFMET75_PFMHT75_IDTight;
    }
  
    HLT_LEP = false;
    if( (dataset_group != "Data") || (dataset_sample == "MET") ){                                 
        //======= MC ======================================================================
        if( (RecoLepID == 11) || ((RecoLepID > 31100) && (RecoLepID <= 31199)) || ((RecoLepID > 41100) && (RecoLepID <= 41199)) ){          // ele ele
            if( HLT_SingleEle || HLT_DoubleEle ) HLT_LEP = true;
        }
        if( (RecoLepID == 13) || ((RecoLepID > 31300) && (RecoLepID <= 31399)) || ((RecoLepID > 41300) && (RecoLepID <= 41399)) ){          // mu mu
            if( HLT_SingleMu || HLT_DoubleMu ) HLT_LEP = true;
        }
        if( (RecoLepID > 1000) && (RecoLepID < 1999) ){                                                                                     // ele mu
            if( HLT_SingleMu || HLT_SingleEle || HLT_EleMu ) HLT_LEP = true;
        }
    }else if( (dataset_group == "Data") && (dataset_sample != "MET") ){                            
        //======= DATA ====================================================================
        if( (RecoLepID == 11) || ((RecoLepID > 31100) && (RecoLepID <= 31199)) || ((RecoLepID > 41100) && (RecoLepID <= 41199)) ){          // ele ele
            if( dataset_sample == "DoubleEle" ){
                if( HLT_DoubleEle ) HLT_LEP = true;
            }else if( dataset_sample == "SingleEle" ){
                if( HLT_SingleEle && !HLT_DoubleEle ) HLT_LEP = true;
            }else if( dataset_sample == "Electrons" ){
                if( HLT_SingleEle || HLT_DoubleEle ) HLT_LEP = true;
            }
        }
        if( (RecoLepID == 13) || ((RecoLepID > 31300) && (RecoLepID <= 31399)) || ((RecoLepID > 41300) && (RecoLepID <= 41399)) ){          // mu mu
            if( dataset_sample == "DoubleMu" ){
                if( HLT_DoubleMu ) HLT_LEP = true;
            }else if( dataset_sample == "SingleMu" ){
                if( HLT_SingleMu && !HLT_DoubleMu ) HLT_LEP = true;
            }
        }
        if( (RecoLepID > 1000) && (RecoLepID < 1999) ){                                                                                     // ele mu
            if( dataset_sample == "EleMu" ){
                if( HLT_EleMu ) HLT_LEP = true;
            }else if( dataset_sample == "SingleEle" ){
                if( HLT_SingleEle && !HLT_EleMu ) HLT_LEP = true;    
            }else if( dataset_sample == "SingleMu" ){
                if( HLT_SingleMu && !HLT_SingleEle && !HLT_EleMu ) HLT_LEP = true;
            }
        }
    }

    if( HLT_LEP ) triggered = true;
    
    return triggered;
}


//-------------------------------------------------------------------------
// Return idx for Leading and Trailing BJets
//
// btagFlag -> 0 -> DeepCSV
// btagFlag -> 1 -> DeepJet
//-------------------------------------------------------------------------
vector<int> HEPHero::findLeadingAndTrailingBJets( int btagFlag ) {

    // Result vector to store indexes
    vector<int> iLeadingAndTrailing;

    // Jet Index
    int iJetLeading = -1;
    int iJetTrailing = -1;

    // Largest and second largest Btag values i.e. Leading and Trailing
    float LeadingBtag = -999.;
    float TrailingBtag = -999.;

    // Btag value
    float Jet_btag = 0;

    if (selectedJet.size() == 1) {
        iJetLeading = selectedJet.at(0);
        if ( btagFlag == 0 ) {
            LeadingBtag = Jet_btagDeepB[iJetLeading];
        } else if ( btagFlag == 1 ) {
            LeadingBtag = Jet_btagDeepFlavB[iJetLeading];
        }
    } else {
        for( unsigned int iselJet = 0; iselJet < selectedJet.size(); ++iselJet ) {
            int iJet = selectedJet.at(iselJet);
            if ( btagFlag == 0 ) {
                Jet_btag = Jet_btagDeepB[iJet];
            } else if ( btagFlag == 1 ) {
                Jet_btag = Jet_btagDeepFlavB[iJet];
            }
            if (Jet_btag > LeadingBtag) {
                TrailingBtag = LeadingBtag;
                iJetTrailing = iJetLeading;
                LeadingBtag = Jet_btag;
                iJetLeading = iJet;
            } else if (Jet_btag > TrailingBtag) {
                iJetTrailing = iJet;
                TrailingBtag = Jet_btag;
            }
        }
    }

    iLeadingAndTrailing.push_back(iJetLeading);
    iLeadingAndTrailing.push_back(iJetTrailing);

    return iLeadingAndTrailing;
}



//-------------------------------------------------------------------------
// Return minimum jet-lepton DR
//-------------------------------------------------------------------------
float HEPHero::JetLepDR( int iJet ){
    float MinDR = 9;
    
    for( unsigned int iselEle = 0; iselEle < selectedEle.size(); ++iselEle ) {
        int iEle = selectedEle.at(iselEle);
        float DEta = fabs( Electron_eta[iEle] - Jet_eta[iJet] );
        float DPhi = fabs( Electron_phi[iEle] - Jet_phi[iJet] );
        if( DPhi > M_PI ) DPhi = 2*M_PI - DPhi;
        float DR = sqrt( DEta*DEta + DPhi*DPhi );
        if( DR < MinDR ) MinDR = DR;
    }
     
    for( unsigned int iselMu = 0; iselMu < selectedMu.size(); ++iselMu ) {
        int iMu = selectedMu.at(iselMu); 
        float DEta = fabs( Muon_eta[iMu] - Jet_eta[iJet] );
        float DPhi = fabs( Muon_phi[iMu] - Jet_phi[iJet] );
        if( DPhi > M_PI ) DPhi = 2*M_PI - DPhi;
        float DR = sqrt( DEta*DEta + DPhi*DPhi );
        if( DR < MinDR ) MinDR = DR;
    }
    
    return MinDR;
}



//-------------------------------------------------------------------------
// Return boolean informing if the reco jet is a signal b jet or not
//-------------------------------------------------------------------------
bool HEPHero::SignalBJet( int iJet ){
    bool isSignal = false;
    int idxGenJet = Jet_genJetIdx[iJet];
    if( idxGenJet >= 0 ){
        if( GenJet_partonFlavour[idxGenJet] == 5 ){
            float DEtab = GenJet_eta[idxGenJet] - GenPart_eta[3];
            float DPhib = GenJet_phi[idxGenJet] - GenPart_phi[3];
            if( DPhib > M_PI ) DPhib = 2*M_PI - DPhib;
            float DRb = sqrt(pow(DEtab,2) + pow(DPhib,2));
            if( DRb < 0.4 ) isSignal = true;
        }else if( GenJet_partonFlavour[idxGenJet] == -5 ){
            float DEtaantib = GenJet_eta[idxGenJet] - GenPart_eta[4];
            float DPhiantib = GenJet_phi[idxGenJet] - GenPart_phi[4];
            if( DPhiantib > M_PI ) DPhiantib = 2*M_PI - DPhiantib;
            float DRantib = sqrt(pow(DEtaantib,2) + pow(DPhiantib,2));
            if( DRantib < 0.4 ) isSignal = true;
        }
    }
    return isSignal;
}


//-------------------------------------------------------------------------
// Return boolean informing if the reco electron is a signal electron or not
//-------------------------------------------------------------------------
bool HEPHero::SignalEle( int iEle ){
    bool isSignal = false;
    int idxGenEle = Electron_genPartIdx[iEle];
    if( (idxGenEle >= 0) && (abs(GenPart_pdgId[idxGenEle]) == 11) && (GenPart_pdgId[GenPart_genPartIdxMother[idxGenEle]] == 23) ){
        isSignal = true;
    }
    return isSignal;
}



//-------------------------------------------------------------------------
// Return boolean informing if the reco muon is a signal muon or not
//-------------------------------------------------------------------------
bool HEPHero::SignalMu( int iMu ){
    bool isSignal = false;
    int idxGenMu = Muon_genPartIdx[iMu];
    if( (idxGenMu >= 0) && (abs(GenPart_pdgId[idxGenMu]) == 13) && (GenPart_pdgId[GenPart_genPartIdxMother[idxGenMu]] == 23) ){
        isSignal = true;
    }
    return isSignal;
}



