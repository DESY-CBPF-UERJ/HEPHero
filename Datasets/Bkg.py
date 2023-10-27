nano_version = 'v9'
pathAPV16 = 'Datasets/Files/bkg_16/UL_APV'+nano_version+'/'
path16 = 'Datasets/Files/bkg_16/UL_'+nano_version+'/'
path17 = 'Datasets/Files/bkg_17/UL_'+nano_version+'/'
path18 = 'Datasets/Files/bkg_18/UL_'+nano_version+'/'

# ID digits:
# 1st-2nd = 16(2016),17(2017),18(2018)                      # Year
# 3th-4th = 00(Data),01(MC-signal),02-13(MC-bkg)            # Type
# 5th-6th = 00(none),...                                    # Bkg -> Exclusive interval 
# 5th-6th = 00(none),11(250_30),12(250_40),55(1250_100)     # Signal -> Phisical process
# 5th-6th = 00(none),01(A),02(B),03(C)                      # Data -> Era
# 7th     = 1(with "HIPM"/"APV")("pre-VFP"), 0(without "HIPM"/"APV")("pos-VFP")  # 2016


periods = ["APV_16", "16", "17", "18"]
years = ["16", "16", "17", "18"]
paths = [pathAPV16, path16, path17, path18]

for period,year,path in zip(periods,years,paths):
    
    if period[0:3] == "APV":
        APVID = "1"
    else:
        APVID = "0"
    
    DYPt0To50 = [
        ["DYJetsToLL_Pt-0To3_"+period]               + [year+'0201'+APVID, path+"DYJetsToLL_Pt-Inclusive.txt"],
        ["DYJetsToLL_PtZ-3To50_"+period]             + [year+'0201'+APVID, path+"DYJetsToLL_PtZ-0To50.txt"],
        ##["DYJetsToLL_PtZ-0To50_"+period]             + [year+'0201'+APVID, path+"DYJetsToLL_PtZ-0To50.txt"],
        ##["DYJetsToLL_Pt-Inclusive_"+period]          + [year+'0201'+APVID, path+"DYJetsToLL_Pt-Inclusive.txt"],
    ]
    
    DYPt50ToInf = [
        ["DYJetsToLL_PtZ-50To100_"+period]           + [year+'0202'+APVID, path+"DYJetsToLL_PtZ-50To100.txt"],
        ["DYJetsToLL_PtZ-100To250_"+period]          + [year+'0203'+APVID, path+"DYJetsToLL_PtZ-100To250.txt"],
        ["DYJetsToLL_PtZ-250To400_"+period]          + [year+'0204'+APVID, path+"DYJetsToLL_PtZ-250To400.txt"],
        ["DYJetsToLL_PtZ-400To650_"+period]          + [year+'0205'+APVID, path+"DYJetsToLL_PtZ-400To650.txt"],
        ["DYJetsToLL_PtZ-650ToInf_"+period]          + [year+'0206'+APVID, path+"DYJetsToLL_PtZ-650ToInf.txt"],
        ##["DYJetsToLL_Pt-50To100_"+period]            + [year+'0202'+APVID, path+"DYJetsToLL_Pt-50To100.txt"],
        ##["DYJetsToLL_Pt-100To250_"+period]           + [year+'0203'+APVID, path+"DYJetsToLL_Pt-100To250.txt"],
        ##["DYJetsToLL_Pt-250To400_"+period]           + [year+'0204'+APVID, path+"DYJetsToLL_Pt-250To400.txt"],
        ##["DYJetsToLL_Pt-400To650_"+period]           + [year+'0205'+APVID, path+"DYJetsToLL_Pt-400To650.txt"],
        ##["DYJetsToLL_Pt-650ToInf_"+period]           + [year+'0206'+APVID, path+"DYJetsToLL_Pt-650ToInf.txt"],    
        ##["DYJetsToLL_HT-Inclusive_"+period]          + [year+'0207'+APVID, path+"DYJetsToLL_HT-Inclusive.txt"],
        ##["DYJetsToLL_HT-70to100_"+period]            + [year+'0208'+APVID, path+"DYJetsToLL_HT-70to100.txt"],
        ##["DYJetsToLL_HT-100to200_"+period]           + [year+'0209'+APVID, path+"DYJetsToLL_HT-100to200.txt"],
        ##["DYJetsToLL_HT-200to400_"+period]           + [year+'0210'+APVID, path+"DYJetsToLL_HT-200to400.txt"],
        ##["DYJetsToLL_HT-400to600_"+period]           + [year+'0211'+APVID, path+"DYJetsToLL_HT-400to600.txt"],
        ##["DYJetsToLL_HT-600to800_"+period]           + [year+'0212'+APVID, path+"DYJetsToLL_HT-600to800.txt"],
        ##["DYJetsToLL_HT-800to1200_"+period]          + [year+'0213'+APVID, path+"DYJetsToLL_HT-800to1200.txt"],
        ##["DYJetsToLL_HT-1200to2500_"+period]         + [year+'0214'+APVID, path+"DYJetsToLL_HT-1200to2500.txt"],
        ##["DYJetsToLL_HT-2500toInf_"+period]          + [year+'0215'+APVID, path+"DYJetsToLL_HT-2500toInf.txt"],
    ]
    
    TTFullLep = [
        ["TTTo2L2Nu_"+period]                        + [year+'0300'+APVID, path+"TTTo2L2Nu.txt"],
    ]
    TTSemiLep = [
        ["TTToSemiLeptonic_"+period]                 + [year+'0301'+APVID, path+"TTToSemiLeptonic.txt"],
    ]
    ST = [
        ["ST_tW_top_"+period]                        + [year+'0400'+APVID, path+"ST_tW_top.txt"],
        ["ST_tW_antitop_"+period]                    + [year+'0401'+APVID, path+"ST_tW_antitop.txt"],
        ["ST_t-channel_top_"+period]                 + [year+'0402'+APVID, path+"ST_t-channel_top.txt"],
        ["ST_t-channel_antitop_"+period]             + [year+'0403'+APVID, path+"ST_t-channel_antitop.txt"],
        ["ST_s-channel_"+period]                     + [year+'0404'+APVID, path+"ST_s-channel.txt"],
    ]
    VZ = [
        ["ZZTo2L2Nu_"+period]                        + [year+'0501'+APVID, path+"ZZTo2L2Nu.txt"],
        ["ZZTo4L_"+period]                           + [year+'0502'+APVID, path+"ZZTo4L.txt"],
        ["WZTo3LNu_"+period]                         + [year+'0512'+APVID, path+"WZTo3LNu.txt"],
    ]
    ResidualSM = [
        #["ZZ_Inclusive_"+period]                     + [year+'0500'+APVID, path+"ZZ.txt"],
        #["ZZ_Others_"+period]                        + [year+'0500'+APVID, path+"ZZ.txt"],
        ["ZZTo2Q2L_"+period]                         + [year+'0504'+APVID, path+"ZZTo2Q2L.txt"],
        #["WZ_Inclusive_"+period]                     + [year+'0510'+APVID, path+"WZ.txt"],
        #["WZ_Others_"+period]                        + [year+'0510'+APVID, path+"WZ.txt"],
        ["WZTo2Q2L_"+period]                         + [year+'0513'+APVID, path+"WZTo2Q2L.txt"],
        #["WW_"+period]                               + [year+'0520'+APVID, path+"WW.txt"],
        ["WWTo2L2Nu_"+period]                        + [year+'0521'+APVID, path+"WWTo2L2Nu.txt"],
        ["ZZZ_"+period]                              + [year+'0600'+APVID, path+"ZZZ.txt"],
        ["WZZ_"+period]                              + [year+'0601'+APVID, path+"WZZ.txt"],
        ["WWZ_"+period]                              + [year+'0602'+APVID, path+"WWZ.txt"],
        ["WWW_"+period]                              + [year+'0603'+APVID, path+"WWW.txt"],
        ["TTWZ_"+period]                             + [year+'0700'+APVID, path+"TTWZ.txt"],
        ["TTZZ_"+period]                             + [year+'0701'+APVID, path+"TTZZ.txt"],
        ["TTWW_"+period]                             + [year+'0702'+APVID, path+"TTWW.txt"],
        ["TWZToLL_thad_Wlept_"+period]               + [year+'0800'+APVID, path+"TWZToLL_thad_Wlept.txt"],
        ["TWZToLL_tlept_Whad_"+period]               + [year+'0801'+APVID, path+"TWZToLL_tlept_Whad.txt"],
        ["TWZToLL_tlept_Wlept_"+period]              + [year+'0802'+APVID, path+"TWZToLL_tlept_Wlept.txt"],
        ["TTWJetsToLNu_"+period]                     + [year+'1000'+APVID, path+"TTWJetsToLNu.txt"],
        ["TTWJetsToQQ_"+period]                      + [year+'1001'+APVID, path+"TTWJetsToQQ.txt"],
        ["TTZToQQ_"+period]                          + [year+'1002'+APVID, path+"TTZToQQ.txt"],
        ["TTZToNuNu_"+period]                        + [year+'1003'+APVID, path+"TTZToNuNu.txt"],
        ["TTZToLL_"+period]                          + [year+'1004'+APVID, path+"TTZToLL.txt"],
        ["tZq_ll_"+period]                           + [year+'1100'+APVID, path+"tZq_ll.txt"],
        #["ttH_HToZZ_"+period]                        + [year+'1300'+APVID, path+"ttH_HToZZ.txt"],
        #["ttHTobb_"+period]                          + [year+'1301'+APVID, path+"ttHTobb.txt"],
        #["ttHToTauTau_"+period]                      + [year+'1302'+APVID, path+"ttHToTauTau.txt"],
        #["GluGluHToWWTo2L2Nu_"+period]               + [year+'1400'+APVID, path+"GluGluHToWWTo2L2Nu.txt"],
        #["GluGluHToZZTo4L_"+period]                  + [year+'1401'+APVID, path+"GluGluHToZZTo4L.txt"],
        #["WplusH_HToZZTo4L_"+period]                 + [year+'1500'+APVID, path+"WplusH_HToZZTo4L.txt"],
        #["WminusH_HToZZTo4L_"+period]                + [year+'1501'+APVID, path+"WminusH_HToZZTo4L.txt"],
        #["ZH_HToBB_ZToLL_"+period]                   + [year+'1600'+APVID, path+"ZH_HToBB_ZToLL.txt"],
        #["ZH_HToZZ_"+period]                         + [year+'1601'+APVID, path+"ZH_HToZZ.txt"],
        #["QCD_Pt-15To20_"+period]                    + [year+'1800'+APVID, path+"QCD_Pt-15To20.txt"],
        #["QCD_Pt-20To30_"+period]                    + [year+'1801'+APVID, path+"QCD_Pt-20To30.txt"],
        #["QCD_Pt-30To50_"+period]                    + [year+'1802'+APVID, path+"QCD_Pt-30To50.txt"],
        #["QCD_Pt-50To80_"+period]                    + [year+'1803'+APVID, path+"QCD_Pt-50To80.txt"],
        #["QCD_Pt-80To120_"+period]                   + [year+'1804'+APVID, path+"QCD_Pt-80To120.txt"],
        #["QCD_Pt-120To170_"+period]                  + [year+'1805'+APVID, path+"QCD_Pt-120To170.txt"],
        #["QCD_Pt-170To300_"+period]                  + [year+'1806'+APVID, path+"QCD_Pt-170To300.txt"],
        #["QCD_Pt-300To470_"+period]                  + [year+'1807'+APVID, path+"QCD_Pt-300To470.txt"],
        #["QCD_Pt-470To600_"+period]                  + [year+'1808'+APVID, path+"QCD_Pt-470To600.txt"],
        #["QCD_Pt-600To800_"+period]                  + [year+'1809'+APVID, path+"QCD_Pt-600To800.txt"],
        #["QCD_Pt-800To1000_"+period]                 + [year+'1810'+APVID, path+"QCD_Pt-800To1000.txt"],
        #["QCD_Pt-1000ToInf_"+period]                 + [year+'1811'+APVID, path+"QCD_Pt-1000ToInf.txt"],
        #["WJetsToLNu_"+period]                       + [year+'1900'+APVID, path+"WJetsToLNu.txt"],
        #["WGToLNuG_"+period]                         + [year+'1901'+APVID, path+"WGToLNuG.txt"],
    ]
    
    
    if period == "APV_16":
        ResidualSM_preVFP_16 = ResidualSM
        VZ_preVFP_16 = VZ
        ST_preVFP_16 = ST
        TTSemiLep_preVFP_16 = TTSemiLep
        TTFullLep_preVFP_16 = TTFullLep
        DYPt0To50_preVFP_16 = DYPt0To50
        DYPt50ToInf_preVFP_16 = DYPt50ToInf
    elif period == "16":
        ResidualSM_postVFP_16 = ResidualSM
        VZ_postVFP_16 = VZ
        ST_postVFP_16 = ST
        TTSemiLep_postVFP_16 = TTSemiLep
        TTFullLep_postVFP_16 = TTFullLep
        DYPt0To50_postVFP_16 = DYPt0To50
        DYPt50ToInf_postVFP_16 = DYPt50ToInf
    elif period == "17":
        ResidualSM_17 = ResidualSM
        VZ_17 = VZ
        ST_17 = ST
        TTSemiLep_17 = TTSemiLep
        TTFullLep_17 = TTFullLep
        DYPt0To50_17 = DYPt0To50
        DYPt50ToInf_17 = DYPt50ToInf
    elif period == "18":
        ResidualSM_18 = ResidualSM
        VZ_18 = VZ
        ST_18 = ST
        TTSemiLep_18 = TTSemiLep
        TTFullLep_18 = TTFullLep
        DYPt0To50_18 = DYPt0To50
        DYPt50ToInf_18 = DYPt50ToInf




























