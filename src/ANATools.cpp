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
        (Nbjets >= 1) && 
        (ttbar_reco == 0) && 
        (dataset_group != "Data") 
      ){                                        // [Signal Region]
        RegionID = 0;
    }
    else if( (RecoLepID < 100) && 
        (Nbjets == 0) &&
        (Njets >= 1) &&  
        (MET_pt < 100)
      ){                                        // [DY - Control Region]
        RegionID = 1;
    }
    else if( (RecoLepID > 1000) && (RecoLepID < 1999) && 
        (Nbjets >= 1) 
      ){                                        // [ttbar - Control Region]
        RegionID = 2;
    }
    if( (RecoLepID > 30000) && (RecoLepID < 39999) && 
        (Nbjets == 0) &&
        (Njets >= 1) &&
        (MET_Lep3_Mt > 50) && (MET_Lep3_Mt < 100)
      ){                                        // [WZ - Control Region]
        RegionID = 3;
    }
    else if( (RecoLepID > 40000) && (RecoLepID < 49999) && 
        (Nbjets == 0) &&
        (Njets >= 1) &&
        (Lep3Lep4_M > 10)
      ){                                        // [ZZ - Control Region] 
        RegionID = 4;
    }
    
}


//---------------------------------------------------------------------------------------------------------------
// MLP Model for signal discrimination (Keras)
//---------------------------------------------------------------------------------------------------------------
void HEPHero::Signal_discriminators(){
    
    float floatC = 1.;
    
    //MLP_score_keras = MLP_keras.predict({LeadingLep_pt, LepLep_pt, LepLep_deltaR, LepLep_deltaM, MET_pt, MET_LepLep_Mt, MET_LepLep_deltaPhi});
    
    MLP_score_torch = MLP_torch.predict({LeadingLep_pt, LepLep_pt, LepLep_deltaR, LepLep_deltaM, MET_pt, MET_LepLep_Mt, MET_LepLep_deltaPhi, TrailingLep_pt, MT2LL});//, Nbjets*floatC});
    
    MLP4_score_torch = pow(1.e4,MLP_score_torch)/1.e4;
    
}


//---------------------------------------------------------------------------------------------------------------
// Jet lep overlap
//---------------------------------------------------------------------------------------------------------------
bool HEPHero::Jet_lep_overlap( int ijet, float deltaR ){
    
    bool overlap = false;
    double drMin = 10000;
    
    //for( unsigned int iele = 0; iele < nElectron; ++iele ) {
    for( unsigned int iselEle = 0; iselEle < selectedEle.size(); ++iselEle ) {
        int iele = selectedEle.at(iselEle);    
        double deta = fabs(Electron_eta[iele] - Jet_eta[ijet]);
        double dphi = fabs(Electron_phi[iele] - Jet_phi[ijet]);
        if( dphi > M_PI ) dphi = 2*M_PI - dphi;
        double dr = sqrt( deta*deta + dphi*dphi );
        if( dr < drMin ) drMin = dr;
    }
    //for( unsigned int imu = 0; imu < nMuon; ++imu ) {
    for( unsigned int iselMu = 0; iselMu < selectedMu.size(); ++iselMu ) {
        int imu = selectedMu.at(iselMu);
        double deta = fabs(Muon_eta[imu] - Jet_eta[ijet]);
        double dphi = fabs(Muon_phi[imu] - Jet_phi[ijet]);
        if( dphi > M_PI ) dphi = 2*M_PI - dphi;
        double dr = sqrt( deta*deta + dphi*dphi );
        if( dr < drMin ) drMin = dr;
    }
    if( drMin < deltaR ) overlap = true;
    
    return overlap;
}


//---------------------------------------------------------------------------------------------------------------
// Jet selection
//---------------------------------------------------------------------------------------------------------------
void HEPHero::JetSelection(){

    selectedJet.clear();
    
    JESvariation();
    JERvariation();
    
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
        
        if( Jet_lep_overlap( ijet, JET_LEP_DR_ISO_CUT ) ) continue;
        if( (Jet_pt[ijet] < 50) && (Jet_puId[ijet] < JET_PUID_WP) ) continue;
        
        if( abs(Jet_eta[ijet]) >= 5.0 ) continue;
        if( abs(Jet_eta[ijet]) > 1.4 ) Njets_forward += 1;
        if( abs(Jet_eta[ijet]) > Jet_abseta_max ) Jet_abseta_max = abs(Jet_eta[ijet]);
        if( abs(Jet_eta[ijet]) >= JET_ETA_CUT ) continue;
        selectedJet.push_back(ijet);
        TLorentzVector Jet;
        Jet.SetPtEtaPhiE(Jet_pt[ijet], Jet_eta[ijet], Jet_phi[ijet], 0);
        
        Njets += 1;
        if( !Jet_lep_overlap(ijet, 0.4) ) Njets_LepIso04 += 1;
        if( PileupJet( ijet ) ) NPUjets += 1;
        HT += Jet_pt[ijet];
        HPx += Jet.Px();
        HPy += Jet.Py();
        if( Jet_pt[ijet] > 30 ){ 
            Njets30 += 1;
            if( !Jet_lep_overlap(ijet, 0.4) ) Njets30_LepIso04 += 1;
            HT30 += Jet_pt[ijet];
            HPx30 += Jet.Px();
            HPy30 += Jet.Py();
        }
        if( Jet_pt[ijet] > 40 ){ 
            Njets40 += 1;
            if( !Jet_lep_overlap(ijet, 0.4) ) Njets40_LepIso04 += 1;
            HT40 += Jet_pt[ijet];
            HPx40 += Jet.Px();
            HPy40 += Jet.Py();
        }
        if( JetBTAG( ijet, JET_BTAG_WP ) ){ 
            Nbjets += 1;
            if( Jet_pt[ijet] > 30 ) Nbjets30 += 1;
            if( !Jet_lep_overlap(ijet, 0.4) ){
                Nbjets_LepIso04 += 1;
                if( Jet_pt[ijet] > 30 ) Nbjets30_LepIso04 += 1;
            }
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
    if( Njets > 0 ) LeadingJet_pt = Jet_pt[selectedJet.at(0)];
    if( Njets > 1 ) SubLeadingJet_pt = Jet_pt[selectedJet.at(1)];
}


//---------------------------------------------------------------------------------------------------------------
// Jet angular variables
//---------------------------------------------------------------------------------------------------------------
void HEPHero::Get_Jet_Angular_Variables( int pt_cut ){
    
    if( (pt_cut != 20) && (pt_cut != 30) && (pt_cut != 40) ){
        cout << "Sorry, for angular variables the only cuts acceptable are 20, 30, or 40. Let's consider your cut equal to 20 GeV!" << endl;
        pt_cut = 20;
    }
  
    float omegaMin = 999999;
    float chiMin = 999999;
    float fMax = 0;
    for( unsigned int iselJet = 0; iselJet < selectedJet.size(); ++iselJet ) {
        int iJet = selectedJet.at(iselJet);
        if( Jet_pt[iJet] < pt_cut ) continue;
        double HPx = 0;
        double HPy = 0;
        for( unsigned int iselJet2 = 0; iselJet2 < selectedJet.size(); ++iselJet2 ) {
            int iJet2 = selectedJet.at(iselJet2);
            if( Jet_pt[iJet2] < pt_cut ) continue;
            TLorentzVector Jet;
            Jet.SetPtEtaPhiE(Jet_pt[iJet2], Jet_eta[iJet2], Jet_phi[iJet2], 0);
            HPx += Jet.Px();
            HPy += Jet.Py();
        }
        double MHT_i = sqrt(HPx*HPx + HPy*HPy);
        
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
    if( (_sysName_lateral == "JES") && (dataset_group != "Data") && !(apply_met_recoil_corr && (dsNameDY == "DYJetsToLL")) ){
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
        
        /*
        float CorrectedMET_x;
        float CorrectedMET_y;
        
        float pfmet_ex = MET_pt*cos(MET_phi);
        float pfmet_ey = MET_pt*sin(MET_phi);
        
        float visPx = LepLep_pt*cos(LepLep_phi);
        float visPy = LepLep_pt*sin(LepLep_phi);
        
        float boson_pt = -999;
        float boson_phi = -999;
        for( unsigned int ipart = 0; ipart < nGenPart; ++ipart ) {
            if( ((abs(GenPart_pdgId[ipart]) == 11) || (abs(GenPart_pdgId[ipart]) == 13) || (abs(GenPart_pdgId[ipart]) == 15)) &&
                ((GenPart_pdgId[GenPart_genPartIdxMother[ipart]] == 22) || (GenPart_pdgId[GenPart_genPartIdxMother[ipart]] == 23)) ){
                int boson_idx = GenPart_genPartIdxMother[ipart];
                boson_pt = GenPart_pt[boson_idx];
                boson_phi = GenPart_phi[boson_idx];
                break;
            }else if( ((abs(GenPart_pdgId[ipart]) == 11) || (abs(GenPart_pdgId[ipart]) == 13) || (abs(GenPart_pdgId[ipart]) == 15)) &&
                ((GenPart_genPartIdxMother[ipart] == 0) || (GenPart_genPartIdxMother[ipart] == 1)) ){
                TLorentzVector lep1;
                TLorentzVector lep2;
                lep1.SetPtEtaPhiM(GenPart_pt[ipart], GenPart_eta[ipart], GenPart_phi[ipart], GenPart_mass[ipart]);
                lep2.SetPtEtaPhiM(GenPart_pt[ipart+1], GenPart_eta[ipart+1], GenPart_phi[ipart+1], GenPart_mass[ipart+1]);
                boson_pt = (lep1+lep2).Pt();
                boson_phi = (lep1+lep2).Phi();;
                break;
            }
        }
        
        float genPx = boson_pt*cos(boson_phi);
        float genPy = boson_pt*sin(boson_phi);
        
        Z_recoil.CorrectByMeanResolution(
            pfmet_ex, // uncorrected type I pf met px (float)
            pfmet_ey, // uncorrected type I pf met py (float)
            genPx, // generator Z/W/Higgs px (float)
            genPy, // generator Z/W/Higgs py (float)
            visPx, // generator visible Z/W/Higgs px (float)
            visPy, // generator visible Z/W/Higgs py (float)
            Njets30,  // number of jets (hadronic jet multiplicity) (int)
            CorrectedMET_x, // corrected type I pf met px (float)
            CorrectedMET_y  // corrected type I pf met py (float)
        );
        
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
        */
        
        Ux = -(MET_pt*cos(MET_phi) + LepLep_pt*cos(LepLep_phi));
        Uy = -(MET_pt*sin(MET_phi) + LepLep_pt*sin(LepLep_phi));
        
        U1 =  Ux*cos(LepLep_phi) + Uy*sin(LepLep_phi);
        U2 = -Ux*sin(LepLep_phi) + Uy*cos(LepLep_phi);
        
        
        /*
        if( dataset_year == "17" ){
            if( Njets_ISR == 0 ){
                mc_u1_mean = Recoil_linears( LepLep_pt, -0.94156673, 1.75265787, -1.03413641, 55.89113125 );
                data_u1_mean = Recoil_linears( LepLep_pt, -0.87956373, 1.97453781, -1.04102595, 48.53804473 );
                ratio_u1_sigma = 1.05;
                ratio_u2_sigma = 1.05;
            }else if( Njets_ISR == 1 ){
                mc_u1_mean = Recoil_linears( LepLep_pt, -0.937999, -2.87487082, -1.01418559, 69.88669753 );
                data_u1_mean = Recoil_linears( LepLep_pt, -0.92980303, -2.45308022, -1.01699646, 69.88488229 );
                ratio_u1_sigma = 1.05;
                ratio_u2_sigma = 1.05;
            }else if( Njets_ISR >= 2 ){
                mc_u1_mean = -1.00520746*LepLep_pt + 0.74503521;
                data_u1_mean = -1.00169317*LepLep_pt + 1.0902607;
                ratio_u1_sigma = 1.05;
                ratio_u2_sigma = 1.05;
            }
        }else if( dataset_year == "18" ){
            if( Njets_ISR == 0 ){
                mc_u1_mean = Recoil_linears( LepLep_pt, -0.95352544, 1.5965469, -1.04030247, 59.63387475 );
                data_u1_mean = Recoil_linears( LepLep_pt, -0.89354413, 2.09950711, -1.02613109, 48.84773165 );
                ratio_u1_sigma = 1.05;
                ratio_u2_sigma = 1.05;
            }else if( Njets_ISR == 1 ){
                mc_u1_mean = Recoil_linears( LepLep_pt, -0.93972472, -3.4133597, -1.01111812, 72.21242867 );
                data_u1_mean = Recoil_linears( LepLep_pt, -0.93472784, -2.58310229, -1.00859591, 77.58574812 );
                ratio_u1_sigma = 1.05;
                ratio_u2_sigma = 1.05;
            }else if( Njets_ISR >= 2 ){
                mc_u1_mean = -1.00276055*LepLep_pt + 0.13789201;
                data_u1_mean = -0.99480348*LepLep_pt + 0.61119181;
                ratio_u1_sigma = 1.05;
                ratio_u2_sigma = 1.05;
            }
        }
        */
        
        
        vector<float> mc_u1_mean; 
        vector<float> mc_u1_mean_unc_up;
        vector<float> mc_u1_mean_unc_down;
        vector<float> data_u1_mean;
        vector<float> data_u1_mean_unc_up;
        vector<float> data_u1_mean_unc_down;
        vector<float> ratio_u1_sigma;
        vector<float> ratio_u1_sigma_unc_up;
        vector<float> ratio_u1_sigma_unc_down;
        vector<float> ratio_u2_sigma;
        vector<float> ratio_u2_sigma_unc_up;
        vector<float> ratio_u2_sigma_unc_down;
        
        
        if( (dataset_year == "16") && dataset_HIPM ){
            if( Njets_ISR == 0 ){
                mc_u1_mean = { -39.905678,  -49.777471,  -59.980332,  -70.866129,  -81.543262,
                        -93.096595, -103.418582, -114.367718, -131.723455};
                mc_u1_mean_unc_down = {0.076476, 0.055177, 0.098362, 0.135582, 0.181867, 0.249757,
                    0.206197, 0.20038 , 0.203744};
                mc_u1_mean_unc_up = {0.076639, 0.055612, 0.098412, 0.135807, 0.181866, 0.250517,
                    0.207524, 0.202101, 0.203745};
                data_u1_mean = { -38.619509,  -48.319686,  -58.488187,  -68.545536,  -80.064364,
                        -90.762309, -102.419492, -112.998105, -131.259676};
                data_u1_mean_unc_down = {0.056497, 0.065499, 0.094793, 0.148455, 0.168527, 0.262938,
                    0.309866, 0.506041, 0.390571};
                data_u1_mean_unc_up = {0.056506, 0.065566, 0.094773, 0.148453, 0.168497, 0.262978,
                    0.31004 , 0.506159, 0.390993};
                ratio_u1_sigma = {1.057156, 1.050333, 1.056863, 1.065487, 1.064072, 1.066866,
                    1.083291, 1.07448 , 1.060924};
                ratio_u1_sigma_unc_down = {0.003701, 0.003296, 0.005085, 0.007189, 0.008723, 0.012366,
                    0.012467, 0.017639, 0.013216};
                ratio_u1_sigma_unc_up = {0.003704, 0.003296, 0.005086, 0.00719 , 0.008725, 0.012367,
                    0.012469, 0.017642, 0.013216};
                ratio_u2_sigma = {1.055301, 1.051136, 1.053954, 1.059011, 1.063427, 1.039059,
                    1.048895, 1.052477, 1.087352};
                ratio_u2_sigma_unc_down = {0.00517 , 0.005379, 0.005551, 0.007613, 0.008258, 0.01045 ,
                    0.013321, 0.014657, 0.010913};
                ratio_u2_sigma_unc_up = {0.00517 , 0.00538 , 0.005552, 0.007618, 0.008258, 0.010453,
                    0.013324, 0.014665, 0.010918};
            }else if( Njets_ISR == 1 ){
                mc_u1_mean = { -45.059289,  -54.5109  ,  -64.311458,  -74.145412,  -84.12014 ,
                        -94.277027, -104.42766 , -114.448007, -132.814766, -163.862574,
                    -194.412293, -225.128116, -256.01769 , -286.471907};
                mc_u1_mean_unc_down = {0.061837, 0.060191, 0.06315 , 0.061322, 0.074481, 0.085916,
                    0.077138, 0.059273, 0.061557, 0.095016, 0.145684, 0.216756,
                    0.230593, 0.212321};
                mc_u1_mean_unc_up = {0.061982, 0.06061 , 0.063404, 0.061405, 0.074764, 0.086394,
                    0.080098, 0.060089, 0.06194 , 0.095104, 0.145729, 0.216867,
                    0.242004, 0.215391};
                data_u1_mean = { -44.713345,  -54.058346,  -63.606555,  -73.164915,  -83.243078,
                        -93.151458, -103.327589, -113.29406 , -131.569931, -162.78397 ,
                    -193.834704, -224.218538, -254.029428, -283.739504};
                data_u1_mean_unc_down = {0.059102, 0.052745, 0.055683, 0.073609, 0.089991, 0.109277,
                    0.106662, 0.163251, 0.112237, 0.26627 , 0.301892, 0.368483,
                    0.664432, 0.96877 };
                data_u1_mean_unc_up = {0.059116, 0.052773, 0.055686, 0.07361 , 0.090019, 0.10928 ,
                    0.10668 , 0.163271, 0.112246, 0.266328, 0.302085, 0.368483,
                    0.664506, 0.968898};
                ratio_u1_sigma = {1.046014, 1.047295, 1.047236, 1.045778, 1.049734, 1.054039,
                    1.045537, 1.045375, 1.044414, 1.047932, 1.040544, 1.033043,
                    1.0213  , 1.036906};
                ratio_u1_sigma_unc_down = {0.003316, 0.003056, 0.003165, 0.003505, 0.00418 , 0.004866,
                    0.004506, 0.005674, 0.003839, 0.00807 , 0.009216, 0.011282,
                    0.018194, 0.025551};
                ratio_u1_sigma_unc_up = {0.003316, 0.003057, 0.003166, 0.003506, 0.004182, 0.004871,
                    0.004508, 0.005675, 0.003842, 0.008072, 0.009216, 0.011291,
                    0.018197, 0.025576};
                ratio_u2_sigma = {1.052179, 1.048683, 1.051902, 1.052874, 1.053185, 1.049339,
                    1.053653, 1.055499, 1.053688, 1.048105, 1.069797, 1.05692 ,
                    1.048564, 1.050617};
                ratio_u2_sigma_unc_down = {0.004508, 0.004579, 0.004671, 0.004916, 0.00505 , 0.005388,
                    0.006263, 0.005935, 0.005359, 0.005934, 0.008545, 0.012716,
                    0.01762 , 0.017814};
                ratio_u2_sigma_unc_up = {0.004509, 0.004579, 0.004672, 0.004917, 0.005052, 0.005391,
                    0.006264, 0.00594 , 0.005361, 0.005935, 0.00855 , 0.012716,
                    0.01762 , 0.017814};
            }else if( Njets_ISR >= 2 ){
                mc_u1_mean = { -45.242471,  -55.459239,  -65.408892,  -75.213327,  -85.11258 ,
                        -95.188546, -105.116725, -115.118836, -133.362869, -163.904661,
                    -194.391847, -224.751022, -254.948475, -285.905   };
                mc_u1_mean_unc_down = {0.120166, 0.096818, 0.13946 , 0.111403, 0.074966, 0.093441,
                    0.112481, 0.115537, 0.068511, 0.104112, 0.1986  , 0.1541  ,
                    0.213889, 0.197565};
                mc_u1_mean_unc_up = {0.120287, 0.097557, 0.139612, 0.111437, 0.075061, 0.09384 ,
                    0.113381, 0.115873, 0.068512, 0.104202, 0.198612, 0.154469,
                    0.222286, 0.201879};
                data_u1_mean = { -44.773448,  -55.021366,  -65.031853,  -74.891458,  -84.790925,
                        -94.674725, -104.641808, -114.278209, -132.732706, -163.160288,
                    -192.99    , -222.179642, -253.327459, -283.988749};
                data_u1_mean_unc_down = {0.073708, 0.129892, 0.135387, 0.136887, 0.162164, 0.156816,
                    0.144032, 0.243466, 0.160705, 0.180151, 0.308152, 0.394109,
                    0.599388, 0.565285};
                data_u1_mean_unc_up = {0.073722, 0.129891, 0.135422, 0.137165, 0.1622  , 0.157272,
                    0.144537, 0.243499, 0.160844, 0.180162, 0.308623, 0.394989,
                    0.600647, 0.566023};
                ratio_u1_sigma = {1.052652, 1.045626, 1.05219 , 1.046471, 1.044258, 1.045388,
                    1.042382, 1.042655, 1.042252, 1.032514, 1.033873, 1.041643,
                    1.030096, 1.027653};
                ratio_u1_sigma_unc_down = {0.006036, 0.006865, 0.008004, 0.007548, 0.00733 , 0.00727 ,
                    0.006961, 0.00927 , 0.005511, 0.006083, 0.010174, 0.011418,
                    0.016496, 0.015131};
                ratio_u1_sigma_unc_up = {0.006056, 0.006891, 0.008034, 0.007588, 0.007371, 0.007311,
                    0.006995, 0.00929 , 0.005524, 0.006087, 0.010175, 0.011421,
                    0.016496, 0.015131};
                ratio_u2_sigma = {1.036153, 1.053516, 1.044228, 1.055136, 1.037933, 1.044794,
                    1.041796, 1.040243, 1.057642, 1.053659, 1.048342, 1.054939,
                    1.036677, 1.048414};
                ratio_u2_sigma_unc_down = {0.005904, 0.005598, 0.006824, 0.00728 , 0.008036, 0.008352,
                    0.007535, 0.007852, 0.005824, 0.00744 , 0.007605, 0.01044 ,
                    0.010968, 0.016639};
                ratio_u2_sigma_unc_up = {0.005924, 0.005633, 0.006863, 0.007325, 0.008088, 0.008404,
                    0.007582, 0.007886, 0.005844, 0.007443, 0.007612, 0.01044 ,
                    0.010971, 0.016644};
            }
        }else if( (dataset_year == "16") && !dataset_HIPM ){
            if( Njets_ISR == 0 ){
                mc_u1_mean = { -40.495758,  -50.311634,  -60.833259,  -71.321902,  -82.056569,
                        -92.969373, -103.543604, -113.938446, -131.897318};
                mc_u1_mean_unc_down = {0.065181, 0.072503, 0.096453, 0.153823, 0.16437 , 0.23618 ,
                    0.273633, 0.274492, 0.208255};
                mc_u1_mean_unc_up = {0.065497, 0.07304 , 0.097319, 0.154103, 0.164762, 0.23651 ,
                    0.275171, 0.275475, 0.208272};
                data_u1_mean = { -39.809393,  -49.540567,  -59.617118,  -70.228568,  -81.202453,
                        -91.558952, -102.435309, -113.292758, -131.685462};
                data_u1_mean_unc_down = {0.058588, 0.061861, 0.1239  , 0.147871, 0.255509, 0.456423,
                    0.389199, 0.612086, 0.527409};
                data_u1_mean_unc_up = {0.058608, 0.061856, 0.124088, 0.147847, 0.25556 , 0.456481,
                    0.389701, 0.61213 , 0.527534};
                ratio_u1_sigma = {1.054907, 1.054162, 1.056694, 1.063273, 1.053323, 1.06611 ,
                    1.035747, 1.077589, 1.049559};
                ratio_u1_sigma_unc_down = {0.00318 , 0.003428, 0.005441, 0.007267, 0.009982, 0.016708,
                    0.015057, 0.021247, 0.016249};
                ratio_u1_sigma_unc_up = {0.00318 , 0.003428, 0.005442, 0.007276, 0.009982, 0.01671 ,
                    0.015058, 0.021248, 0.016255};
                ratio_u2_sigma = {1.057087, 1.06179 , 1.054983, 1.066122, 1.081645, 1.072539,
                    1.073855, 1.042024, 1.061928};
                ratio_u2_sigma_unc_down = {0.004665, 0.004338, 0.005599, 0.007694, 0.009563, 0.009645,
                    0.015104, 0.017801, 0.012927};
                ratio_u2_sigma_unc_up = {0.004665, 0.004338, 0.005599, 0.007695, 0.009566, 0.009651,
                    0.015104, 0.017804, 0.012929};
            }else if( Njets_ISR == 1 ){
                mc_u1_mean = { -45.405408,  -54.857478,  -64.592102,  -74.357044,  -84.26828 ,
                        -94.298388, -104.479095, -114.719153, -132.782252, -163.603721,
                    -194.559175, -224.968286, -255.509072, -286.376422};
                mc_u1_mean_unc_down = {0.07248 , 0.062553, 0.056885, 0.062215, 0.076066, 0.092195,
                    0.071929, 0.085831, 0.07031 , 0.085376, 0.191156, 0.218964,
                    0.278107, 0.228343};
                mc_u1_mean_unc_up = {0.072584, 0.062824, 0.05712 , 0.062393, 0.076262, 0.092958,
                    0.074157, 0.087481, 0.071038, 0.085497, 0.191518, 0.219014,
                    0.283902, 0.23225 };
                data_u1_mean = { -45.085022,  -54.590471,  -64.018236,  -73.915216,  -83.780503,
                        -93.810286, -103.732058, -113.729145, -132.112345, -162.773059,
                    -193.656361, -224.1768  , -254.869688, -285.235386};
                data_u1_mean_unc_down = {0.052276, 0.059651, 0.062893, 0.085434, 0.106705, 0.108856,
                    0.1612  , 0.161891, 0.170204, 0.18491 , 0.282085, 0.466129,
                    0.880979, 1.187937};
                data_u1_mean_unc_up = {0.052283, 0.059683, 0.062928, 0.085466, 0.106735, 0.108899,
                    0.161225, 0.162115, 0.170216, 0.184917, 0.282456, 0.466235,
                    0.883666, 1.187947};
                ratio_u1_sigma = {1.055525, 1.051173, 1.04879 , 1.051433, 1.044394, 1.045781,
                    1.046849, 1.057198, 1.037396, 1.050445, 1.029754, 1.041002,
                    1.081332, 1.026036};
                ratio_u1_sigma_unc_down = {0.003281, 0.003104, 0.003001, 0.003667, 0.004425, 0.004735,
                    0.005715, 0.005815, 0.005285, 0.005739, 0.009193, 0.013532,
                    0.024301, 0.03032 };
                ratio_u1_sigma_unc_up = {0.003283, 0.003104, 0.003001, 0.003667, 0.004426, 0.004736,
                    0.005716, 0.005817, 0.005287, 0.005743, 0.009194, 0.013533,
                    0.024319, 0.030328};
                ratio_u2_sigma = {1.058139, 1.062962, 1.059208, 1.05941 , 1.058675, 1.05848 ,
                    1.067268, 1.054347, 1.061606, 1.066193, 1.051988, 1.065616,
                    1.056469, 1.08078 };
                ratio_u2_sigma_unc_down = {0.004167, 0.003813, 0.004604, 0.004089, 0.004892, 0.006059,
                    0.005456, 0.005631, 0.004552, 0.006181, 0.010587, 0.011426,
                    0.011617, 0.089657};
                ratio_u2_sigma_unc_up = {0.004167, 0.003814, 0.004605, 0.00409 , 0.004893, 0.006061,
                    0.005457, 0.005633, 0.004553, 0.006182, 0.010588, 0.011427,
                    0.011624, 0.227797};
            }else if( Njets_ISR >= 2 ){
                mc_u1_mean = { -45.407227,  -55.418524,  -65.539155,  -75.431177,  -85.544331,
                        -95.140656, -105.253035, -115.165106, -133.652387, -163.859489,
                    -194.289399, -224.404955, -254.67286 , -285.194562};
                mc_u1_mean_unc_down = {0.133143, 0.085185, 0.108647, 0.122409, 0.146547, 0.121924,
                    0.157389, 0.131591, 0.080963, 0.133456, 0.184243, 0.227963,
                    0.260626, 0.228352};
                mc_u1_mean_unc_up = {0.133357, 0.085755, 0.108895, 0.12247 , 0.147022, 0.122052,
                    0.157979, 0.131932, 0.08103 , 0.133586, 0.184493, 0.228175,
                    0.264755, 0.230207};
                data_u1_mean = { -45.270117,  -55.246886,  -65.336301,  -75.124195,  -84.952751,
                        -95.030022, -104.954782, -114.513035, -133.16327 , -163.551503,
                    -193.92606 , -223.872499, -254.205082, -284.612858};
                data_u1_mean_unc_down = {0.12877 , 0.118135, 0.105354, 0.115746, 0.15572 , 0.227769,
                    0.183586, 0.212273, 0.149522, 0.204903, 0.304915, 0.458345,
                    0.581713, 0.83885 };
                data_u1_mean_unc_up = {0.128979, 0.118157, 0.105351, 0.115735, 0.155721, 0.227795,
                    0.183681, 0.212319, 0.149636, 0.204905, 0.305179, 0.458586,
                    0.58215 , 0.838849};
                ratio_u1_sigma = {1.056247, 1.042216, 1.048177, 1.04023 , 1.052042, 1.04404 ,
                    1.036931, 1.030647, 1.044322, 1.030366, 1.014897, 1.031738,
                    1.034165, 1.036194};
                ratio_u1_sigma_unc_down = {0.006953, 0.005794, 0.006105, 0.006651, 0.00789 , 0.008988,
                    0.008216, 0.008255, 0.005206, 0.006947, 0.00965 , 0.013521,
                    0.016685, 0.021988};
                ratio_u1_sigma_unc_up = {0.006967, 0.005814, 0.006128, 0.006678, 0.007917, 0.009011,
                    0.008236, 0.008272, 0.005216, 0.00695 , 0.009652, 0.013523,
                    0.01669 , 0.021988};
                ratio_u2_sigma = {1.054817, 1.058403, 1.045484, 1.056674, 1.04726 , 1.037224,
                    1.044615, 1.058487, 1.053144, 1.054138, 1.070324, 1.051908,
                    1.043446, 1.028916};
                ratio_u2_sigma_unc_down = {0.005759, 0.006305, 0.00646 , 0.006419, 0.006733, 0.008191,
                    0.007232, 0.00743 , 0.005475, 0.005411, 0.009207, 0.012548,
                    0.014408, 0.011006};
                ratio_u2_sigma_unc_up = {0.005772, 0.006326, 0.006485, 0.006455, 0.006767, 0.008226,
                    0.007265, 0.007455, 0.00549 , 0.005419, 0.009209, 0.012549,
                    0.014408, 0.011016};
            }
        }else if( dataset_year == "17" ){
            if( Njets_ISR == 0 ){
                mc_u1_mean = { -39.006112,  -48.843511,  -59.150917,  -69.816297,  -80.370661,
                        -91.708845, -101.663109, -112.430475, -130.345983, -160.658249};
                mc_u1_mean_unc_down = {0.046463, 0.06781 , 0.068707, 0.139089, 0.125588, 0.145172,
                    0.201016, 0.187915, 0.162357, 0.274863};
                mc_u1_mean_unc_up = {0.046507, 0.068113, 0.069169, 0.139272, 0.125714, 0.148017,
                    0.201582, 0.188438, 0.162979, 0.274862};
                data_u1_mean = { -37.642498,  -47.248844,  -57.235538,  -67.602223,  -78.378398,
                        -89.224508, -100.242379, -111.276247, -129.523774, -160.808648};
                data_u1_mean_unc_down = {0.055544, 0.08032 , 0.130225, 0.14815 , 0.193906, 0.240626,
                    0.262858, 0.294667, 0.360997, 0.684365};
                data_u1_mean_unc_up = {0.055648, 0.080426, 0.130272, 0.148161, 0.19389 , 0.240612,
                    0.26302 , 0.294693, 0.361049, 0.684367};
                ratio_u1_sigma = {1.038702, 1.041395, 1.042102, 1.050902, 1.04718 , 1.061381,
                    1.068954, 1.060241, 1.07028 , 1.029099};
                ratio_u1_sigma_unc_down = {0.00238 , 0.003411, 0.004647, 0.006383, 0.007082, 0.008486,
                    0.010037, 0.010453, 0.010894, 0.019514};
                ratio_u1_sigma_unc_up = {0.00238 , 0.003412, 0.004648, 0.006384, 0.007085, 0.008488,
                    0.010039, 0.010459, 0.010896, 0.019514};
                ratio_u2_sigma = {1.043136, 1.043348, 1.049467, 1.054299, 1.03428 , 1.030273,
                    1.049066, 1.054248, 1.049709, 1.011775};
                ratio_u2_sigma_unc_down = {0.00315 , 0.003407, 0.004314, 0.005991, 0.006911, 0.007918,
                    0.008849, 0.011469, 0.009784, 0.015888};
                ratio_u2_sigma_unc_up = {0.003151, 0.003408, 0.004317, 0.005993, 0.006916, 0.007919,
                    0.00885 , 0.01147 , 0.009788, 0.015907};
            }else if( Njets_ISR == 1 ){
                mc_u1_mean = { -44.32835 ,  -53.837358,  -63.395174,  -73.143601,  -83.123013,
                        -93.109435, -103.127187, -113.49582 , -131.76748 , -162.720366,
                    -193.158209, -223.974891, -255.318339, -285.529914, -315.972581};
                mc_u1_mean_unc_down = {0.053706, 0.041707, 0.046022, 0.062321, 0.0634  , 0.077879,
                    0.07737 , 0.092353, 0.07989 , 0.097964, 0.177314, 0.191999,
                    0.242018, 0.217193, 0.290247};
                mc_u1_mean_unc_up = {0.053713, 0.04241 , 0.046118, 0.062429, 0.063654, 0.078557,
                    0.079282, 0.09389 , 0.080224, 0.098135, 0.177315, 0.192044,
                    0.25513 , 0.218982, 0.290673};
                data_u1_mean = { -43.826824,  -53.115142,  -62.681302,  -72.463003,  -82.323329,
                        -92.30461 , -102.587505, -112.937296, -131.249892, -162.179269,
                    -193.49791 , -223.987917, -253.781567, -285.47274 , -317.11925 };
                data_u1_mean_unc_down = {0.052477, 0.05367 , 0.061969, 0.075573, 0.093563, 0.112691,
                    0.124063, 0.152559, 0.122616, 0.181162, 0.248253, 0.429247,
                    0.597577, 0.74293 , 0.998883};
                data_u1_mean_unc_up = {0.052506, 0.053685, 0.061985, 0.07565 , 0.093583, 0.112728,
                    0.124085, 0.152819, 0.122675, 0.181168, 0.2486  , 0.429317,
                    0.597586, 0.744118, 1.000797};
                ratio_u1_sigma = {1.038498, 1.043948, 1.04074 , 1.040599, 1.038499, 1.035778,
                    1.039128, 1.043406, 1.036446, 1.035339, 1.036738, 1.034581,
                    1.020998, 0.987458, 1.048728};
                ratio_u1_sigma_unc_down = {0.002504, 0.002242, 0.002501, 0.003129, 0.00353 , 0.004183,
                    0.004401, 0.005276, 0.003999, 0.005446, 0.007921, 0.011854,
                    0.015798, 0.018475, 0.024733};
                ratio_u1_sigma_unc_up = {0.002504, 0.002243, 0.002501, 0.00313 , 0.003531, 0.004184,
                    0.004402, 0.005279, 0.004   , 0.005447, 0.007921, 0.011854,
                    0.015802, 0.018479, 0.024734};
                ratio_u2_sigma = {1.043804, 1.04146 , 1.046355, 1.045662, 1.0534  , 1.050136,
                    1.051458, 1.042247, 1.045738, 1.052822, 1.03547 , 1.045632,
                    1.054777, 1.013663, 1.025936};
                ratio_u2_sigma_unc_down = {0.002985, 0.003061, 0.003131, 0.003577, 0.003194, 0.00467 ,
                    0.004355, 0.004282, 0.003662, 0.005142, 0.004986, 0.00938 ,
                    0.010868, 0.016736, 0.016716};
                ratio_u2_sigma_unc_up = {0.002986, 0.003061, 0.003132, 0.003578, 0.003196, 0.004672,
                    0.004357, 0.004284, 0.003665, 0.005144, 0.004992, 0.009382,
                    0.01087 , 0.016742, 0.016727};
            }else if( Njets_ISR >= 2 ){
                mc_u1_mean = { -44.438598,  -54.567424,  -64.481671,  -74.318243,  -84.158928,
                        -94.068653, -103.900156, -113.986093, -132.142774, -162.545739,
                    -192.546358, -223.065273, -253.531468, -284.335036, -314.595487};
                mc_u1_mean_unc_down = {0.078886, 0.071599, 0.074873, 0.09149 , 0.109891, 0.087955,
                    0.103711, 0.098118, 0.084744, 0.12085 , 0.157385, 0.19876 ,
                    0.202289, 0.230079, 0.267014};
                mc_u1_mean_unc_up = {0.078999, 0.07209 , 0.075137, 0.091715, 0.109902, 0.088452,
                    0.105048, 0.098762, 0.084745, 0.120867, 0.157609, 0.198798,
                    0.209424, 0.232642, 0.267534};
                data_u1_mean = { -44.222829,  -54.289413,  -64.104948,  -73.832165,  -83.709451,
                        -93.554383, -103.210104, -113.329196, -131.674744, -162.235783,
                    -192.85192 , -222.8593  , -253.660887, -283.741062, -312.595375};
                data_u1_mean_unc_down = {0.08205 , 0.104565, 0.104399, 0.113056, 0.150028, 0.13335 ,
                    0.147958, 0.204343, 0.131258, 0.17641 , 0.247295, 0.308304,
                    0.400392, 0.535847, 0.69191 };
                data_u1_mean_unc_up = {0.082103, 0.10459 , 0.10453 , 0.113166, 0.15013 , 0.133446,
                    0.148131, 0.204525, 0.131388, 0.17667 , 0.247378, 0.308305,
                    0.400597, 0.536042, 0.691991};
                ratio_u1_sigma = {1.044246, 1.042321, 1.036888, 1.042157, 1.049047, 1.04143 ,
                    1.042622, 1.037284, 1.038742, 1.041422, 1.035167, 1.042925,
                    1.023492, 1.033575, 1.046816};
                ratio_u1_sigma_unc_down = {0.004387, 0.004824, 0.004984, 0.005428, 0.006447, 0.005466,
                    0.005794, 0.006963, 0.004452, 0.005813, 0.007692, 0.009439,
                    0.011149, 0.014236, 0.017942};
                ratio_u1_sigma_unc_up = {0.004403, 0.004845, 0.005013, 0.005457, 0.006478, 0.005494,
                    0.005815, 0.00698 , 0.004461, 0.005814, 0.007695, 0.00944 ,
                    0.011152, 0.014237, 0.017945};
                ratio_u2_sigma = {1.032064, 1.038758, 1.037176, 1.038305, 1.048244, 1.034888,
                    1.046234, 1.033118, 1.037534, 1.039309, 1.054584, 1.037517,
                    1.046283, 1.019106, 1.027043};
                ratio_u2_sigma_unc_down = {0.00417 , 0.004405, 0.004982, 0.005807, 0.005562, 0.006821,
                    0.005991, 0.006574, 0.004411, 0.00548 , 0.005262, 0.008128,
                    0.009326, 0.006487, 0.013857};
                ratio_u2_sigma_unc_up = {0.004188, 0.00443 , 0.005013, 0.005842, 0.005604, 0.006857,
                    0.006026, 0.006598, 0.004426, 0.005482, 0.005263, 0.008131,
                    0.009327, 0.006488, 0.013858};
            }
        }else if( dataset_year == "18" ){
            if( Njets_ISR == 0 ){
                mc_u1_mean = { -39.801591,  -49.602626,  -59.88805 ,  -70.40703 ,  -81.027062,
                        -92.178838, -103.026086, -113.898017, -131.614174, -162.512376};
                mc_u1_mean_unc_down = {0.059946, 0.061972, 0.085123, 0.134309, 0.147351, 0.200098,
                    0.225306, 0.260251, 0.192151, 0.21628 };
                mc_u1_mean_unc_up = {0.060092, 0.062366, 0.085807, 0.134552, 0.147376, 0.20045 ,
                    0.226124, 0.261098, 0.192347, 0.216289};
                data_u1_mean = { -38.178522,  -47.671736,  -57.387219,  -67.675299,  -78.233536,
                        -88.706219,  -99.829534, -109.704288, -128.943768, -160.209686};
                data_u1_mean_unc_down = {0.052914, 0.070964, 0.098502, 0.125914, 0.170331, 0.230415,
                    0.264596, 0.311978, 0.235241, 0.599403};
                data_u1_mean_unc_up = {0.052961, 0.070978, 0.09849 , 0.125949, 0.170377, 0.230415,
                    0.264606, 0.311982, 0.235503, 0.599533};
                ratio_u1_sigma = {1.044684, 1.050138, 1.046029, 1.052363, 1.048219, 1.077027,
                    1.057671, 1.098257, 1.050351, 1.074306};
                ratio_u1_sigma_unc_down = {0.002653, 0.003092, 0.004176, 0.005796, 0.006904, 0.009411,
                    0.010393, 0.01245 , 0.008402, 0.017071};
                ratio_u1_sigma_unc_up = {0.002654, 0.003094, 0.004179, 0.005798, 0.006906, 0.009419,
                    0.010394, 0.012451, 0.008405, 0.017072};
                ratio_u2_sigma = {1.048344, 1.053209, 1.04997 , 1.047282, 1.045463, 1.046092,
                    1.071624, 1.043858, 1.064873, 1.020884};
                ratio_u2_sigma_unc_down = {0.003193, 0.003651, 0.003662, 0.004684, 0.004719, 0.006816,
                    0.008456, 0.009813, 0.008826, 0.010665};
                ratio_u2_sigma_unc_up = {0.003194, 0.003652, 0.003667, 0.004684, 0.004719, 0.006817,
                    0.008457, 0.009815, 0.008827, 0.010667};
            }else if( Njets_ISR == 1 ){
                mc_u1_mean = { -45.04065 ,  -54.464126,  -64.097302,  -73.724075,  -83.548978,
                        -93.459864, -103.567286, -113.662894, -131.797528, -162.513931,
                    -193.308678, -223.903484, -254.392609, -284.991216, -315.534355};
                mc_u1_mean_unc_down = {0.055571, 0.05355 , 0.046969, 0.058019, 0.06162 , 0.084254,
                    0.080448, 0.092016, 0.075741, 0.136853, 0.145023, 0.231938,
                    0.206938, 0.238174, 0.263958};
                mc_u1_mean_unc_up = {0.055673, 0.054042, 0.047391, 0.058322, 0.062044, 0.084748,
                    0.082993, 0.093479, 0.076028, 0.136987, 0.145178, 0.232188,
                    0.214894, 0.240191, 0.264754};
                data_u1_mean = { -44.231705,  -53.459597,  -62.877168,  -72.498984,  -82.196908,
                        -92.077008, -102.233087, -112.337572, -130.265323, -160.994161,
                    -191.985636, -222.691605, -253.67638 , -283.241612, -313.111005};
                data_u1_mean_unc_down = {0.044367, 0.047774, 0.055678, 0.068785, 0.092917, 0.107773,
                    0.131083, 0.133237, 0.134073, 0.190993, 0.241147, 0.314376,
                    0.488458, 0.644045, 0.704651};
                data_u1_mean_unc_up = {0.044396, 0.047793, 0.05568 , 0.068817, 0.092965, 0.1078  ,
                    0.131115, 0.133261, 0.134136, 0.191003, 0.241205, 0.31459 ,
                    0.488465, 0.644053, 0.704794};
                ratio_u1_sigma = {1.041766, 1.044344, 1.045533, 1.045119, 1.046138, 1.043799,
                    1.045407, 1.034196, 1.042709, 1.034969, 1.049684, 1.021414,
                    1.042675, 1.049572, 1.057851};
                ratio_u1_sigma_unc_down = {0.002391, 0.002387, 0.002387, 0.002901, 0.003505, 0.004236,
                    0.004649, 0.004791, 0.004228, 0.006265, 0.007367, 0.009839,
                    0.013149, 0.016697, 0.017989};
                ratio_u1_sigma_unc_up = {0.002392, 0.002388, 0.00239 , 0.002903, 0.003508, 0.004238,
                    0.004653, 0.004792, 0.004231, 0.006266, 0.007368, 0.009841,
                    0.01315 , 0.0167  , 0.017992};
                ratio_u2_sigma = {1.047499, 1.048094, 1.045392, 1.054331, 1.049999, 1.05065 ,
                    1.047497, 1.056944, 1.050694, 1.051759, 1.055664, 1.056861,
                    1.059045, 1.056126, 1.050848};
                ratio_u2_sigma_unc_down = {0.003152, 0.003315, 0.003284, 0.003826, 0.003385, 0.003605,
                    0.003327, 0.00409 , 0.003946, 0.004205, 0.004955, 0.008776,
                    0.008634, 0.009012, 0.011952};
                ratio_u2_sigma_unc_up = {0.003154, 0.003317, 0.003284, 0.003829, 0.003387, 0.003606,
                    0.00333 , 0.004094, 0.003946, 0.004205, 0.004957, 0.008779,
                    0.008634, 0.009016, 0.011955};
            }else if( Njets_ISR >= 2 ){
                mc_u1_mean = { -44.955025,  -54.988554,  -65.021574,  -74.818546,  -84.787377,
                        -94.575902, -104.294989, -114.221851, -132.523961, -162.707224,
                    -192.893285, -223.081514, -253.923616, -283.768446, -314.202239};
                mc_u1_mean_unc_down = {0.109459, 0.105081, 0.081304, 0.099468, 0.118532, 0.102863,
                    0.092909, 0.100529, 0.08064 , 0.111914, 0.154381, 0.201557,
                    0.217729, 0.221691, 0.267966};
                mc_u1_mean_unc_up = {0.109762, 0.105894, 0.081603, 0.099748, 0.119161, 0.103408,
                    0.094142, 0.100631, 0.080708, 0.111953, 0.154481, 0.20162 ,
                    0.225304, 0.223221, 0.268517};
                data_u1_mean = { -44.287135,  -54.183397,  -64.226373,  -73.816658,  -83.594201,
                        -93.418082, -103.193197, -112.945756, -131.313126, -161.380921,
                    -191.489404, -222.282834, -252.396502, -282.45291 , -311.676689};
                data_u1_mean_unc_down = {0.095072, 0.098353, 0.095663, 0.109848, 0.110294, 0.123913,
                    0.14503 , 0.151955, 0.13077 , 0.167833, 0.205365, 0.252511,
                    0.377382, 0.373493, 0.654551};
                data_u1_mean_unc_up = {0.095129, 0.098367, 0.095866, 0.109947, 0.110388, 0.12409 ,
                    0.145156, 0.152063, 0.130897, 0.167899, 0.205375, 0.25263 ,
                    0.377449, 0.373651, 0.655472};
                ratio_u1_sigma = {1.038824, 1.042326, 1.040449, 1.050138, 1.047111, 1.038838,
                    1.044293, 1.038002, 1.039014, 1.032724, 1.036816, 1.033796,
                    1.042472, 1.023729, 1.038677};
                ratio_u1_sigma_unc_down = {0.005255, 0.005428, 0.005043, 0.005591, 0.005907, 0.005613,
                    0.005701, 0.005748, 0.004418, 0.005464, 0.006788, 0.008264,
                    0.010937, 0.010533, 0.016976};
                ratio_u1_sigma_unc_up = {0.005272, 0.00545 , 0.005074, 0.005624, 0.005942, 0.005645,
                    0.005727, 0.005765, 0.004429, 0.005466, 0.006789, 0.008264,
                    0.010938, 0.010535, 0.016978};
                ratio_u2_sigma = {1.046448, 1.04071 , 1.043785, 1.042813, 1.046508, 1.050302,
                    1.047961, 1.043346, 1.045012, 1.044569, 1.03826 , 1.03554 ,
                    1.04621 , 1.039906, 1.03637 };
                ratio_u2_sigma_unc_down = {0.004574, 0.004742, 0.005038, 0.005423, 0.005415, 0.006198,
                    0.005535, 0.005889, 0.004141, 0.004257, 0.005888, 0.005195,
                    0.008342, 0.010157, 0.011665};
                ratio_u2_sigma_unc_up = {0.004595, 0.004769, 0.005072, 0.005464, 0.005465, 0.006242,
                    0.005577, 0.005918, 0.00416 , 0.004261, 0.00589 , 0.005196,
                    0.008343, 0.010158, 0.011665};
            }
        }
                
        
        vector<float> ZRecoPt_J0;
        vector<float> ZRecoPt_J12;
        if( dataset_year == "16" ){
            ZRecoPt_J0 = {40, 50, 60, 70, 80, 90, 100, 110, 120, 150};
            ZRecoPt_J12 = {40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 180, 210, 240, 270, 300};
        }else if( (dataset_year == "17") || (dataset_year == "18") ){
            ZRecoPt_J0 = {40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 180};
            ZRecoPt_J12 = {40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 180, 210, 240, 270, 300, 330};
        }
        
        int idx = -1;
        if( Njets_ISR == 0 ){
            for (unsigned int i = 0; i < mc_u1_mean.size() ; i++){
                if ( LepLep_pt >= ZRecoPt_J0[i] && LepLep_pt < ZRecoPt_J0[i+1]  ){
                    idx = i;
                    break;
                }
            }
        }else{
            for (unsigned int i = 0; i < mc_u1_mean.size() ; i++){
                if ( LepLep_pt >= ZRecoPt_J12[i] && LepLep_pt < ZRecoPt_J12[i+1]  ){
                    idx = i;
                    break;
                }
            }
        }
        
        
        if( idx >= 0 ){
            
            float data_u1_mean_value = data_u1_mean[idx];
            float mc_u1_mean_value = mc_u1_mean[idx];
            float ratio_u1_sigma_value = ratio_u1_sigma[idx];
            float ratio_u2_sigma_value = ratio_u2_sigma[idx];
            
            if( _sysName_lateral == "Recoil" ){
                if( _Universe == 0 ){ // up down up up   
                    data_u1_mean_value = data_u1_mean[idx] + data_u1_mean_unc_up[idx];
                    mc_u1_mean_value = mc_u1_mean[idx] - mc_u1_mean_unc_down[idx];
                    ratio_u1_sigma_value = ratio_u1_sigma[idx] + ratio_u1_sigma_unc_up[idx];
                    ratio_u2_sigma_value = ratio_u2_sigma[idx] + ratio_u2_sigma_unc_up[idx];    
                }else if( _Universe == 1 ){ // up down down down
                    data_u1_mean_value = data_u1_mean[idx] + data_u1_mean_unc_up[idx];
                    mc_u1_mean_value = mc_u1_mean[idx] - mc_u1_mean_unc_down[idx];
                    ratio_u1_sigma_value = ratio_u1_sigma[idx] - ratio_u1_sigma_unc_down[idx];
                    ratio_u2_sigma_value = ratio_u2_sigma[idx] - ratio_u2_sigma_unc_down[idx]; 
                }else if( _Universe == 2 ){ // down up up up   
                    data_u1_mean_value = data_u1_mean[idx] - data_u1_mean_unc_down[idx];
                    mc_u1_mean_value = mc_u1_mean[idx] + mc_u1_mean_unc_up[idx];
                    ratio_u1_sigma_value = ratio_u1_sigma[idx] + ratio_u1_sigma_unc_up[idx];
                    ratio_u2_sigma_value = ratio_u2_sigma[idx] + ratio_u2_sigma_unc_up[idx];    
                }else if( _Universe == 3 ){ // down up down down
                    data_u1_mean_value = data_u1_mean[idx] - data_u1_mean_unc_down[idx];
                    mc_u1_mean_value = mc_u1_mean[idx] + mc_u1_mean_unc_up[idx];
                    ratio_u1_sigma_value = ratio_u1_sigma[idx] - ratio_u1_sigma_unc_down[idx];
                    ratio_u2_sigma_value = ratio_u2_sigma[idx] - ratio_u2_sigma_unc_down[idx]; 
                }
            }
            
            float U1_new = data_u1_mean_value + (U1-mc_u1_mean_value)*ratio_u1_sigma_value; 
            float U2_new = U2*ratio_u2_sigma_value;
            
            float Ux_new = U1_new*cos(LepLep_phi) - U2_new*sin(LepLep_phi);;
            float Uy_new = U1_new*sin(LepLep_phi) + U2_new*cos(LepLep_phi);
            
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
        
    //=====TRIGGERS FOR 2016=====================================================================
    if( dataset_year == "16" ){
        
        HLT_SingleEle = HLT_Ele27_WPTight_Gsf || HLT_Ele115_CaloIdVT_GsfTrkIdT;

        HLT_DoubleEle = HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ; // HLT_DoubleEle37_Ele27_CaloIdL_GsfTrkIdVL;
            
        HLT_SingleMu = HLT_IsoMu24 || HLT_IsoTkMu24 || HLT_Mu50;
            
        if( (dataset_group == "Data") && (dataset_era == "H") ){
            HLT_DoubleMu = HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ || HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ;
            
            HLT_EleMu = HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ || HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ;
        }else{
            HLT_DoubleMu = HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL || HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL; // HLT_Mu30_TkMu11;
            
            HLT_EleMu = HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL || HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL;
        }
        
        HLT_MET = HLT_PFMET300 || HLT_MET200 || HLT_PFHT300_PFMET110 || HLT_PFMET170_HBHECleaned || HLT_PFMET120_PFMHT120_IDTight || HLT_PFMETNoMu120_PFMHTNoMu120_IDTight;
    }
    
    //=====TRIGGERS FOR 2017=====================================================================
    if( dataset_year == "17" ){
        
        HLT_SingleEle = HLT_Ele35_WPTight_Gsf; // HLT_Ele115_CaloIdVT_GsfTrkIdT;

        HLT_DoubleEle = HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL || HLT_DoubleEle33_CaloIdL_MW;
            
        HLT_SingleMu = HLT_IsoMu27 || HLT_Mu50;
            
        HLT_DoubleMu = HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8;
        
        HLT_EleMu = HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ || HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ;
            
        HLT_MET = HLT_PFMET200_HBHECleaned || HLT_PFMET200_HBHE_BeamHaloCleaned || HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned || HLT_PFMET120_PFMHT120_IDTight || HLT_PFMET120_PFMHT120_IDTight_PFHT60 || HLT_PFMETNoMu120_PFMHTNoMu120_IDTight || HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60 || HLT_PFHT500_PFMET100_PFMHT100_IDTight || HLT_PFHT700_PFMET85_PFMHT85_IDTight || HLT_PFHT800_PFMET75_PFMHT75_IDTight;
    }
    
    //=====TRIGGERS FOR 2018=====================================================================
    if( dataset_year == "18" ){
        
        HLT_SingleEle = HLT_Ele32_WPTight_Gsf || HLT_Ele115_CaloIdVT_GsfTrkIdT;

        HLT_DoubleEle = HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL;  // HLT_DoubleEle25_CaloIdL_MW
            
        HLT_SingleMu = HLT_IsoMu24 || HLT_Mu50;
            
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















