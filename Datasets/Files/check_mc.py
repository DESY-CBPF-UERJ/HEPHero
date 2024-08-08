import os
import sys
import argparse

# List of datasets generated with the command below (example for WWW). 
# dasgoclient --limit 0 --query 'dataset dataset=/WWW*/RunIISummer20UL16NanoAODv2-106X*/NANOAODSIM'
# dasgoclient --limit 0 --query 'dataset dataset=/*bbHToZaToLLChiChi_2HDMa*/RunIISummer20UL*NanoAOD*v2-106X*/NANOAODSIM'

#======SETUP=======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version")
parser.add_argument("-p", "--period")
args = parser.parse_args()

if args.version != "8" and args.version != "9":
    print("Please, enter a valid version: 8 or 9")
    sys.exit()
    
if args.period != "16" and args.period != "17" and args.period != "18":
    print("Please, enter a valid period: 16, 17, or 18")  
    sys.exit()

if args.version == "8":
    args.version = "v2"
elif args.version == "9":
    args.version = "v9"

campaign = "RunIISummer20UL"+args.period+"*NanoAOD*"+args.version+"*106X*/NANOAODSIM"

datasets = [
"DYJetsToLL_LHEFilterPtZ-*_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
#"DYJetsToLL_Pt-*_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
"DYJetsToLL_M-50_HT-*_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
"ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
"ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
"ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
"ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
"ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
"WZ_TuneCP5_13TeV-pythia8",
"WZTo3LNu_mllmin4p0_TuneCP5_13TeV-powheg-pythia8",
"WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"ZZ_TuneCP5_13TeV-pythia8",
"ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8",
"ZZTo4L_TuneCP5_13TeV_powheg_pythia8",
"ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"WW_TuneCP5_13TeV-pythia8",
"WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
"ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",
"WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
"WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8",
"WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
##"ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
##"TTGJets_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8",
##"TTGamma_Dilept_TuneCP5_13TeV-madgraph-pythia8",
"TTWW_TuneCP5_13TeV-madgraph-pythia8",
"TTWZ_TuneCP5_13TeV-madgraph-pythia8",
"TTZZ_TuneCP5_13TeV-madgraph-pythia8",
"TWZToLL_thad_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8",
"TWZToLL_tlept_Whad_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8",
"TWZToLL_tlept_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8",
"TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8",
"TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8",
"TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8",
"TTZToLL_M-1to10_TuneCP5_13TeV-amcatnlo-pythia8",
"TTZToNuNu_TuneCP5_13TeV-amcatnlo-pythia8",
"tZq_ll_4f_ckm_NLO_TuneCP5_erdON_13TeV-amcatnlo-pythia8",
"ttH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8",
"ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8",
"ttHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8",
"GluGluHToWWTo2L2Nu_M125_TuneCP5_13TeV_powheg2_JHUGenV714_pythia8",
"GluGluHToZZTo2L2Q_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8",
"GluGluHToZZTo4L_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8",
"WplusH_HToZZTo2L2X_M125_TuneCP5_13TeV_powheg2-minlo-HWJ_JHUGenV7011_pythia8",
"WplusH_HToZZTo4L_M125_TuneCP5_13TeV_powheg2-minlo-HWJ_JHUGenV7011_pythia8",
"WminusH_HToZZTo2L2X_M125_TuneCP5_13TeV_powheg2-minlo-HWJ_JHUGenV7011_pythia8",
"WminusH_HToZZTo4L_M125_TuneCP5_13TeV_powheg2-minlo-HWJ_JHUGenV7011_pythia8",
"ZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8",
"ZH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2-minlo-HZJ_JHUGenV7011_pythia8",
"QCD_Pt-15To20_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
"QCD_Pt-20To30_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
"QCD_Pt-30To50_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
"QCD_Pt-50To80_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
"QCD_Pt-80To120_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
"QCD_Pt-120To170_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
"QCD_Pt-170To300_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
"QCD_Pt-300To470_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
"QCD_Pt-470To600_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
"QCD_Pt-600To800_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
"QCD_Pt-800To1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
"QCD_Pt-1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
##"tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-amcatnlo-pythia8",
##"WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8",
"WGToLNuG_TuneCP5_13TeV-madgraphMLM-pythia8",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8",
#"QCD_Pt_15to30_TuneCP5_13TeV_pythia8",
#"QCD_Pt_30to50_TuneCP5_13TeV_pythia8",
#"QCD_Pt_50to80_TuneCP5_13TeV_pythia8",
#"QCD_Pt_80to120_TuneCP5_13TeV_pythia8",
#"QCD_Pt_120to170_TuneCP5_13TeV_pythia8",
#"QCD_Pt_170to300_TuneCP5_13TeV_pythia8",
#"QCD_Pt_300to470_TuneCP5_13TeV_pythia8",
#"QCD_Pt_470to600_TuneCP5_13TeV_pythia8",
#"QCD_Pt_600to800_TuneCP5_13TeV_pythia8",
#"QCD_Pt_800to1000_TuneCP5_13TeV_pythia8",
#"QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8",
#"QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8",
#"QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8",
#"QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8",
#"QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8",
"ZH_HToBB_ZToNuNu_M-125_TuneCP5_13TeV-powheg-pythia8",
"ZZTo2Q2Nu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
#"QCD_HT100to200_BGenFilter_TuneCP5_13TeV-madgraph-pythia8",
#"QCD_HT200to300_BGenFilter_TuneCP5_13TeV-madgraph-pythia8",
#"QCD_HT300to500_BGenFilter_TuneCP5_13TeV-madgraph-pythia8",
#"QCD_HT500to700_BGenFilter_TuneCP5_13TeV-madgraph-pythia8",
#"QCD_HT700to1000_BGenFilter_TuneCP5_13TeV-madgraph-pythia8",
#"QCD_HT1000to1500_BGenFilter_TuneCP5_13TeV-madgraph-pythia8",
#"QCD_HT1500to2000_BGenFilter_TuneCP5_13TeV-madgraph-pythia8",
#"QCD_HT2000toInf_BGenFilter_TuneCP5_13TeV-madgraph-pythia8",
"WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"WJetsToLNu_Pt-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"WJetsToLNu_Pt-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"WJetsToLNu_Pt-400To600_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"WJetsToLNu_Pt-600ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"WJetsToLNu_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"WJetsToLNu_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"WJetsToLNu_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"WHJet_HToBB_WToLNu_M-125_TuneCP5_SMEFTsim_topU3l_13TeV-madgraphMLM-pythia8",
"QCD_HT50to100_TuneCP5_PSWeights_13TeV-madgraph-pythia8",
"QCD_HT100to200_TuneCP5_PSWeights_13TeV-madgraph-pythia8",
"QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8",
"QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8",
"QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8",
"QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8",
"QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8",
"QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8",
"QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraph-pythia8",
"Z1JetsToNuNu_M-50_LHEFilterPtZ-50To150_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"Z1JetsToNuNu_M-50_LHEFilterPtZ-150To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"Z1JetsToNuNu_M-50_LHEFilterPtZ-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"Z1JetsToNuNu_M-50_LHEFilterPtZ-400ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"Z2JetsToNuNu_M-50_LHEFilterPtZ-50To150_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"Z2JetsToNuNu_M-50_LHEFilterPtZ-150To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"Z2JetsToNuNu_M-50_LHEFilterPtZ-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"Z2JetsToNuNu_M-50_LHEFilterPtZ-400ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"GluGluToContinToZZTo2e2mu_TuneCP5_13TeV-mcfm701-pythia8",
"GluGluToContinToZZTo2e2nu_TuneCP5_13TeV-mcfm701-pythia8",
"GluGluToContinToZZTo2e2tau_TuneCP5_13TeV-mcfm701-pythia8",
"GluGluToContinToZZTo2mu2nu_TuneCP5_13TeV-mcfm701-pythia8",
"GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV-mcfm701-pythia8",
"GluGluToContinToZZTo4e_TuneCP5_13TeV-mcfm701-pythia8",
"GluGluToContinToZZTo4mu_TuneCP5_13TeV-mcfm701-pythia8",
"GluGluToContinToZZTo4tau_TuneCP5_13TeV-mcfm701-pythia8",
#Signal
"bbHToZaToLLChiChi_2HDMa"
]


print("#############################################################################################")
for i in range(len(datasets)):
    print(">>>> " + datasets[i])
    command = "dasgoclient --limit 0 --query 'dataset dataset=/" + datasets[i] + "*/" + campaign + "'"
    os.system(command)
    print(" ")
