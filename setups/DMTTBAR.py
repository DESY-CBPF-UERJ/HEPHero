#--------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset files
#--------------------------------------------------------------------------------------------------------------------------------------------------
selection = "TTJES"
analysis = "DMTTBAR"
treeName = "Events"
LumiWeights = 1

analysis_parameters = {   
"JET_ETA_CUT                ": 2.4,
"JET_PT_CUT                 ": 20,      # GeV
"JET_ID_WP                  ": 6,       # 0-loose, 2-tight, 6-tightlepveto 
"JET_PUID_WP                ": 7,       # 0-fail, 1-loose, 3-medium, 7-tight
"JET_BTAG_WP                ": 4,       # DeepJet: 0-loose, 1-medium, 2-tight; DeepCSV: 3-loose, 4-medium, 5-tight; [Set up the efficiency maps as well]
"JET_LEP_DR_ISO_CUT         ": 0.4,
"ELECTRON_GAP_LOWER_CUT     ": 1.444,   # Lower absolute limit of barrel-endcap gap
"ELECTRON_GAP_UPPER_CUT     ": 1.566,   # Upper absolute limit of barrel-endcap gap
"ELECTRON_ETA_CUT           ": 2.4, 
"ELECTRON_PT_CUT            ": 20,      # GeV  
"ELECTRON_LOW_PT_CUT        ": 15,      # GeV
"ELECTRON_ID_WP             ": 4,       # 0-veto, 1-loose, 2-medium, 3-tight, 4-WP90iso, 5-WP80iso
"MUON_ETA_CUT               ": 2.4, 
"MUON_PT_CUT                ": 20,      # GeV
"MUON_LOW_PT_CUT            ": 15,      # GeV
"MUON_ID_WP                 ": 1,       # 0-loose, 1-medium, 2-tight
"MUON_ISO_WP                ": 3,       # 0-none, 1-loose/looseID, 2-loose/mediumID, 3-tight/mediumID     
}

corrections = {  # 0-don't apply, 1-apply
"PILEUP_WGT                 ": 1, 
"ELECTRON_ID_WGT            ": 1,
"MUON_ID_WGT                ": 1,
"JET_PUID_WGT               ": 1,
"BTAG_WGT                   ": 1,
"TRIGGER_WGT                ": 1,
"PREFIRING_WGT              ": 1,
"JER_CORR                   ": 1,
"MET_XY_CORR                ": 1,
"MET_RECOIL_CORR            ": 1,
"TOP_PT_WGT                 ": 1,
#"VJETS_HT_WGT               ": 0,
"MUON_ROC_CORR              ": 1,
}

lateral_systematics = { 
"CV":          [0,  1, [], []],   # [sys_source, sys_universe, processes_ID (empty -> all), subsources] 
#"JES":         [1,  54, [], ["AbsoluteMPFBias_fc", "AbsoluteScale_fc", "AbsoluteStat", "FlavorQCD_fc", "Fragmentation_fc", "PileUpDataMC_fc", "PileUpPtBB_fc", "PileUpPtEC1_fc", "PileUpPtEC2_fc", "PileUpPtHF_fc", "PileUpPtRef_fc", "RelativeFSR_fc", "RelativeJEREC1", "RelativeJEREC2", "RelativeJERHF_fc", "RelativePtBB_fc", "RelativePtEC1", "RelativePtEC2", "RelativePtHF_fc", "RelativeBal_fc", "RelativeSample", "RelativeStatEC", "RelativeStatFSR", "RelativeStatHF", "SinglePionECAL_fc", "SinglePionHCAL_fc", "TimePtEta"]],
"JER":         [28,  2, [], []],
#"UncMET":      [29,  2, [], []],
#"Recoil":      [30,  4, ["02"], []],
"JES":         [41,  2, [], ["Total"]],
}

vertical_systematics = {
#"Pileup":      [50,   2,  [], []],
#"EleID":       [51,   2,  [], []],
#"MuID":        [52,  2,  [], []],
#"JetPU":       [53,  2,  [], []],
#"BTag":        [54,  8,  [], ["bc", "light", "bc_fc", "light_fc"]],
#"Trig":        [58,  2,  [], []],
#"PreFir":      [59,  2,  [], []],
#"PDF":         [60,  2,  [], []],
#"AlphaS":      [61,  2,  [], []],
#"Scales":      [62,  9,  [], []],    
#"ISR":         [63,  2,  [], []],
#"FSR":         [64,  2,  [], []],
#"TT1LXS":      [65,  2,  0],
#"TT2LXS":      [66,  2,  0],
#"DYXS":        [67,  2,  0],
}


#--------------------------------------------------------------------------------------------------------------------------------------------------
# Jobs setup
#--------------------------------------------------------------------------------------------------------------------------------------------------
NumMaxEvents = -1
NumFilesPerJob_Data = 1   
NumFilesPerJob_Signal = 50 
NumFilesPerJob_Bkg = 1      


#--------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset files
#--------------------------------------------------------------------------------------------------------------------------------------------------
datasets = [
#"Data_Lep",
#"Data_MET",
"Signal",
#"DYPt50ToInf",
#"DYPt0To50",
"TTFullLep",
#"TTSemiLep",
#"ST",
#"VZ",
#"ResidualSM"    
]


#--------------------------------------------------------------------------------------------------------------------------------------------------
# Metadata
#--------------------------------------------------------------------------------------------------------------------------------------------------
metadata = {       
"NN_prep_keras_XX         ": "Metadata/ML/Keras/preprocessing.json", 
"NN_model_keras_XX        ": "Metadata/ML/Keras/NN_4_100_elu_adam/model.json",
"NN_model_torch_HIPM_16   ": "Metadata/ML/Torch/DeepCSV/2016preVFP/model_scripted.pt",
"NN_model_torch_NORMAL_16 ": "Metadata/ML/Torch/DeepCSV/2016postVFP/model_scripted.pt",
"NN_model_torch_17        ": "Metadata/ML/Torch/DeepCSV/2017/model_scripted.pt",
"NN_model_torch_18        ": "Metadata/ML/Torch/DeepCSV/2018/model_scripted.pt",
"lumi_certificate_16      ": "Metadata/certificates/Cert_271036-284044_13TeV_Legacy2016_Collisions16.json",
"lumi_certificate_17      ": "Metadata/certificates/Cert_294927-306462_13TeV_UL2017_Collisions17.json",
"lumi_certificate_18      ": "Metadata/certificates/Cert_314472-325175_13TeV_Legacy2018_Collisions18.json",
"pdf_type_XX              ": "Metadata/PDF/pdf_type.json",
"pileup_HIPM_16           ": "Metadata/POG/LUM/2016preVFP_UL/puWeights.json.gz",
"pileup_NORMAL_16         ": "Metadata/POG/LUM/2016postVFP_UL/puWeights.json.gz",
"pileup_17                ": "Metadata/POG/LUM/2017_UL/puWeights.json.gz",
"pileup_18                ": "Metadata/POG/LUM/2018_UL/puWeights.json.gz",
"electron_HIPM_16         ": "Metadata/POG/EGM/2016preVFP_UL/electron.json.gz",
"electron_NORMAL_16       ": "Metadata/POG/EGM/2016postVFP_UL/electron.json.gz",
"electron_17              ": "Metadata/POG/EGM/2017_UL/electron.json.gz",
"electron_18              ": "Metadata/POG/EGM/2018_UL/electron.json.gz",
"muon_HIPM_16             ": "Metadata/POG/MUO/2016preVFP_UL/muon_Z.json.gz",
"muon_NORMAL_16           ": "Metadata/POG/MUO/2016postVFP_UL/muon_Z.json.gz",
"muon_17                  ": "Metadata/POG/MUO/2017_UL/muon_Z.json.gz",
"muon_18                  ": "Metadata/POG/MUO/2018_UL/muon_Z.json.gz",
"btag_SF_HIPM_16          ": "Metadata/POG/BTV/2016preVFP_UL/btagging.json.gz",
"btag_SF_NORMAL_16        ": "Metadata/POG/BTV/2016postVFP_UL/btagging.json.gz",
"btag_SF_17               ": "Metadata/POG/BTV/2017_UL/btagging.json.gz",
"btag_SF_18               ": "Metadata/POG/BTV/2018_UL/btagging.json.gz",
"btag_eff_HIPM_16         ": "Metadata/btag_eff/DeepCSVMedium/2016preVFP.json",
"btag_eff_NORMAL_16       ": "Metadata/btag_eff/DeepCSVMedium/2016postVFP.json",
"btag_eff_17              ": "Metadata/btag_eff/DeepCSVMedium/2017.json",
"btag_eff_18              ": "Metadata/btag_eff/DeepCSVMedium/2018.json",
"trigger_16               ": "Metadata/trigger/SF_2016_ttbar.json",
"trigger_17               ": "Metadata/trigger/SF_2017_ttbar.json",
"trigger_18               ": "Metadata/trigger/SF_2018_ttbar.json",
"JES_MC_HIPM_16           ": "Metadata/JES/JES_MC_16_preVFP.txt",
"JES_MC_NORMAL_16         ": "Metadata/JES/JES_MC_16_postVFP.txt",
"JES_MC_17                ": "Metadata/JES/JES_MC_17.txt",
"JES_MC_18                ": "Metadata/JES/JES_MC_18.txt",
"JER_MC_HIPM_16           ": "Metadata/JER/Summer20UL16APV_JRV3_MC/Summer20UL16APV_JRV3_MC_PtResolution_AK4PFchs.txt",
"JER_SF_MC_HIPM_16        ": "Metadata/JER/Summer20UL16APV_JRV3_MC/Summer20UL16APV_JRV3_MC_SF_AK4PFchs.txt",
"JER_MC_NORMAL_16         ": "Metadata/JER/Summer20UL16_JRV3_MC/Summer20UL16_JRV3_MC_PtResolution_AK4PFchs.txt",
"JER_SF_MC_NORMAL_16      ": "Metadata/JER/Summer20UL16_JRV3_MC/Summer20UL16_JRV3_MC_SF_AK4PFchs.txt",
"JER_MC_17                ": "Metadata/JER/Summer19UL17_JRV2_MC/Summer19UL17_JRV2_MC_PtResolution_AK4PFchs.txt",
"JER_SF_MC_17             ": "Metadata/JER/Summer19UL17_JRV2_MC/Summer19UL17_JRV2_MC_SF_AK4PFchs.txt",
"JER_MC_18                ": "Metadata/JER/Summer19UL18_JRV2_MC/Summer19UL18_JRV2_MC_PtResolution_AK4PFchs.txt",
"JER_SF_MC_18             ": "Metadata/JER/Summer19UL18_JRV2_MC/Summer19UL18_JRV2_MC_SF_AK4PFchs.txt",
"jet_puID_HIPM_16         ": "Metadata/POG/JME/2016preVFP_UL/jmar.json.gz",
"jet_puID_NORMAL_16       ": "Metadata/POG/JME/2016postVFP_UL/jmar.json.gz",
"jet_puID_17              ": "Metadata/POG/JME/2017_UL/jmar.json.gz",
"jet_puID_18              ": "Metadata/POG/JME/2018_UL/jmar.json.gz",
"JERC_HIPM_16             ": "Metadata/POG/JME/2016preVFP_UL/jet_jerc.json.gz",
"JERC_NORMAL_16           ": "Metadata/POG/JME/2016postVFP_UL/jet_jerc.json.gz",
"JERC_17                  ": "Metadata/POG/JME/2017_UL/jet_jerc.json.gz",
"JERC_18                  ": "Metadata/POG/JME/2018_UL/jet_jerc.json.gz",
"mu_RoccoR_HIPM_16        ": "Metadata/mu_Rochester/RoccoR2016aUL.txt",
"mu_RoccoR_NORMAL_16      ": "Metadata/mu_Rochester/RoccoR2016bUL.txt",
"mu_RoccoR_17             ": "Metadata/mu_Rochester/RoccoR2017UL.txt",
"mu_RoccoR_18             ": "Metadata/mu_Rochester/RoccoR2018UL.txt",
"Z_recoil_16              ": "Metadata/boson_recoil/Z/TypeI-PFMet_Run2016_legacy.root",
"Z_recoil_17              ": "Metadata/boson_recoil/Z/Type1_PFMET_2017.root",
"Z_recoil_18              ": "Metadata/boson_recoil/Z/TypeI-PFMet_Run2018.root",
}


#----------------------------------------------------------------------------------------------------------------------------------------
# Plots
#----------------------------------------------------------------------------------------------------------------------------------------
Get_Image_in_EPS = 0
Get_Image_in_PNG = 1
Get_Image_in_PDF = 0
