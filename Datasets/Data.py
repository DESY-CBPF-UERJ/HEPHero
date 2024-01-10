nano_version = 'v9'
path16 = 'Datasets/Files/data_16/UL_'+nano_version+'/'
path17 = 'Datasets/Files/data_17/UL_'+nano_version+'/'
path18 = 'Datasets/Files/data_18/UL_'+nano_version+'/'

# ID digits:
# 1st-2nd = 16(2016),17(2017),18(2018)                      # Year
# 3th-4th = 00(Data),01(MC-signal),02-13(MC-bkg)            # Group
# 5th-6th = 00(none),...                                    # Bkg -> Exclusive interval 
# 5th-6th = 00(none),11(250_30),12(250_40),55(1250_100)     # Signal -> Phisical process
# 5th-6th = 00(none),01(A),02(B),03(C)                      # Data -> Era
# 7th     = 1(with "HIPM"/"APV")("pre-VFP"), 0(without "HIPM"/"APV")("pos-VFP")  # 2016

Data_Lep_preVFP_16 = [
["Data_SingleEle_v1_B_16"]          + ['1600011', path16+"SingleEle_B_v1.txt"],
["Data_SingleEle_v2_B_16"]          + ['1600021', path16+"SingleEle_B_v2.txt"],
["Data_SingleEle_C_16"]             + ['1600031', path16+"SingleEle_C.txt"],
["Data_SingleEle_D_16"]             + ['1600041', path16+"SingleEle_D.txt"],
["Data_SingleEle_E_16"]             + ['1600051', path16+"SingleEle_E.txt"],
["Data_SingleEle_HIPM_F_16"]        + ['1600061', path16+"SingleEle_F_HIPM.txt"],
["Data_DoubleEle_v1_B_16"]          + ['1600111', path16+"DoubleEle_B_v1.txt"],
["Data_DoubleEle_v2_B_16"]          + ['1600121', path16+"DoubleEle_B_v2.txt"],
["Data_DoubleEle_C_16"]             + ['1600131', path16+"DoubleEle_C.txt"],
["Data_DoubleEle_D_16"]             + ['1600141', path16+"DoubleEle_D.txt"],
["Data_DoubleEle_E_16"]             + ['1600151', path16+"DoubleEle_E.txt"],
["Data_DoubleEle_HIPM_F_16"]        + ['1600161', path16+"DoubleEle_F_HIPM.txt"],
["Data_SingleMu_v1_B_16"]           + ['1600211', path16+"SingleMu_B_v1.txt"],
["Data_SingleMu_v2_B_16"]           + ['1600221', path16+"SingleMu_B_v2.txt"],
["Data_SingleMu_C_16"]              + ['1600231', path16+"SingleMu_C.txt"],
["Data_SingleMu_D_16"]              + ['1600241', path16+"SingleMu_D.txt"],
["Data_SingleMu_E_16"]              + ['1600251', path16+"SingleMu_E.txt"],
["Data_SingleMu_HIPM_F_16"]         + ['1600261', path16+"SingleMu_F_HIPM.txt"],
["Data_DoubleMu_v1_B_16"]           + ['1600311', path16+"DoubleMu_B_v1.txt"],
["Data_DoubleMu_v2_B_16"]           + ['1600321', path16+"DoubleMu_B_v2.txt"],
["Data_DoubleMu_C_16"]              + ['1600331', path16+"DoubleMu_C.txt"],
["Data_DoubleMu_D_16"]              + ['1600341', path16+"DoubleMu_D.txt"],
["Data_DoubleMu_E_16"]              + ['1600351', path16+"DoubleMu_E.txt"],
["Data_DoubleMu_HIPM_F_16"]         + ['1600361', path16+"DoubleMu_F_HIPM.txt"],
["Data_EleMu_v1_B_16"]              + ['1600411', path16+"EleMu_B_v1.txt"],
["Data_EleMu_v2_B_16"]              + ['1600421', path16+"EleMu_B_v2.txt"],
["Data_EleMu_C_16"]                 + ['1600431', path16+"EleMu_C.txt"],
["Data_EleMu_D_16"]                 + ['1600441', path16+"EleMu_D.txt"],
["Data_EleMu_E_16"]                 + ['1600451', path16+"EleMu_E.txt"],
["Data_EleMu_HIPM_F_16"]            + ['1600461', path16+"EleMu_F_HIPM.txt"],
]

Data_MET_preVFP_16 = [
["Data_MET_v1_B_16"]                + ['1600511', path16+"MET_B_v1.txt"],
["Data_MET_v2_B_16"]                + ['1600521', path16+"MET_B_v2.txt"],
["Data_MET_C_16"]                   + ['1600531', path16+"MET_C.txt"],
["Data_MET_D_16"]                   + ['1600541', path16+"MET_D.txt"],
["Data_MET_E_16"]                   + ['1600551', path16+"MET_E.txt"],
["Data_MET_HIPM_F_16"]              + ['1600561', path16+"MET_F_HIPM.txt"],
]

Data_Lep_postVFP_16 = [
["Data_SingleEle_F_16"]             + ['1600070', path16+"SingleEle_F.txt"],
["Data_SingleEle_G_16"]             + ['1600080', path16+"SingleEle_G.txt"],
["Data_SingleEle_H_16"]             + ['1600090', path16+"SingleEle_H.txt"],
["Data_DoubleEle_F_16"]             + ['1600170', path16+"DoubleEle_F.txt"],
["Data_DoubleEle_G_16"]             + ['1600180', path16+"DoubleEle_G.txt"],
["Data_DoubleEle_H_16"]             + ['1600190', path16+"DoubleEle_H.txt"],
["Data_SingleMu_F_16"]              + ['1600270', path16+"SingleMu_F.txt"],
["Data_SingleMu_G_16"]              + ['1600280', path16+"SingleMu_G.txt"],
["Data_SingleMu_H_16"]              + ['1600290', path16+"SingleMu_H.txt"],
["Data_DoubleMu_F_16"]              + ['1600370', path16+"DoubleMu_F.txt"],
["Data_DoubleMu_G_16"]              + ['1600380', path16+"DoubleMu_G.txt"],
["Data_DoubleMu_H_16"]              + ['1600390', path16+"DoubleMu_H.txt"],
["Data_EleMu_F_16"]                 + ['1600470', path16+"EleMu_F.txt"],
["Data_EleMu_G_16"]                 + ['1600480', path16+"EleMu_G.txt"],
["Data_EleMu_H_16"]                 + ['1600490', path16+"EleMu_H.txt"],
]

Data_MET_postVFP_16 = [
["Data_MET_F_16"]                   + ['1600570', path16+"MET_F.txt"],
["Data_MET_G_16"]                   + ['1600580', path16+"MET_G.txt"],
["Data_MET_H_16"]                   + ['1600590', path16+"MET_H.txt"],
]

Data_Lep_17 = [
["Data_SingleEle_B_17"]             + ['1700020', path17+"SingleEle_B.txt"],
["Data_SingleEle_C_17"]             + ['1700030', path17+"SingleEle_C.txt"],
["Data_SingleEle_D_17"]             + ['1700040', path17+"SingleEle_D.txt"],
["Data_SingleEle_E_17"]             + ['1700050', path17+"SingleEle_E.txt"],
["Data_SingleEle_F_17"]             + ['1700060', path17+"SingleEle_F.txt"],
["Data_DoubleEle_B_17"]             + ['1700120', path17+"DoubleEle_B.txt"],
["Data_DoubleEle_C_17"]             + ['1700130', path17+"DoubleEle_C.txt"],
["Data_DoubleEle_D_17"]             + ['1700140', path17+"DoubleEle_D.txt"],
["Data_DoubleEle_E_17"]             + ['1700150', path17+"DoubleEle_E.txt"],
["Data_DoubleEle_F_17"]             + ['1700160', path17+"DoubleEle_F.txt"],
["Data_SingleMu_B_17"]              + ['1700220', path17+"SingleMu_B.txt"],
["Data_SingleMu_C_17"]              + ['1700230', path17+"SingleMu_C.txt"],
["Data_SingleMu_D_17"]              + ['1700240', path17+"SingleMu_D.txt"],
["Data_SingleMu_E_17"]              + ['1700250', path17+"SingleMu_E.txt"],
["Data_SingleMu_F_17"]              + ['1700260', path17+"SingleMu_F.txt"],
["Data_DoubleMu_B_17"]              + ['1700320', path17+"DoubleMu_B.txt"],
["Data_DoubleMu_C_17"]              + ['1700330', path17+"DoubleMu_C.txt"],
["Data_DoubleMu_D_17"]              + ['1700340', path17+"DoubleMu_D.txt"],
["Data_DoubleMu_E_17"]              + ['1700350', path17+"DoubleMu_E.txt"],
["Data_DoubleMu_F_17"]              + ['1700360', path17+"DoubleMu_F.txt"],
["Data_EleMu_B_17"]                 + ['1700420', path17+"EleMu_B.txt"],
["Data_EleMu_C_17"]                 + ['1700430', path17+"EleMu_C.txt"],
["Data_EleMu_D_17"]                 + ['1700440', path17+"EleMu_D.txt"],
["Data_EleMu_E_17"]                 + ['1700450', path17+"EleMu_E.txt"],
["Data_EleMu_F_17"]                 + ['1700460', path17+"EleMu_F.txt"],
]

Data_MET_17 = [
["Data_MET_B_17"]                   + ['1700520', path17+"MET_B.txt"],
["Data_MET_C_17"]                   + ['1700530', path17+"MET_C.txt"],
["Data_MET_D_17"]                   + ['1700540', path17+"MET_D.txt"],
["Data_MET_E_17"]                   + ['1700550', path17+"MET_E.txt"],
["Data_MET_F_17"]                   + ['1700560', path17+"MET_F.txt"],
]

Data_Lep_18 = [
["Data_Electrons_A_18"]             + ['1800010', path18+"Ele_A.txt"],    
["Data_Electrons_B_18"]             + ['1800020', path18+"Ele_B.txt"],
["Data_Electrons_C_18"]             + ['1800030', path18+"Ele_C.txt"],
["Data_Electrons_D_18"]             + ['1800040', path18+"Ele_D.txt"],
["Data_SingleMu_A_18"]              + ['1800210', path18+"SingleMu_A.txt"],
["Data_SingleMu_B_18"]              + ['1800220', path18+"SingleMu_B.txt"],
["Data_SingleMu_C_18"]              + ['1800230', path18+"SingleMu_C.txt"],
["Data_SingleMu_D_18"]              + ['1800240', path18+"SingleMu_D.txt"],
["Data_DoubleMu_A_18"]              + ['1800310', path18+"DoubleMu_A.txt"],
["Data_DoubleMu_B_18"]              + ['1800320', path18+"DoubleMu_B.txt"],
["Data_DoubleMu_C_18"]              + ['1800330', path18+"DoubleMu_C.txt"],
["Data_DoubleMu_D_18"]              + ['1800340', path18+"DoubleMu_D.txt"],
["Data_EleMu_A_18"]                 + ['1800410', path18+"EleMu_A.txt"],
["Data_EleMu_B_18"]                 + ['1800420', path18+"EleMu_B.txt"],
["Data_EleMu_C_18"]                 + ['1800430', path18+"EleMu_C.txt"],
["Data_EleMu_D_18"]                 + ['1800440', path18+"EleMu_D.txt"],
]

Data_MET_18 = [
["Data_MET_A_18"]                   + ['1800510', path18+"MET_A.txt"],    
["Data_MET_B_18"]                   + ['1800520', path18+"MET_B.txt"],
["Data_MET_C_18"]                   + ['1800530', path18+"MET_C.txt"],
["Data_MET_D_18"]                   + ['1800540', path18+"MET_D.txt"],
]


# OPEN DATA
openpath = 'Datasets/Files/opendata/data_12/'

Data_12 = [
    ["Data_TauPlusX_B_12"]               + ['1200000', openpath+"TauPlusX_B.txt"],
    ["Data_TauPlusX_C_12"]               + ['1200010', openpath+"TauPlusX_C.txt"],
]


