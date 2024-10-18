analysis = "ABC"
nano_version = 'v9'
path_16 = analysis+'/Datasets/Files/data_16/'+nano_version+'/'
path_17 = analysis+'/Datasets/Files/data_17/'+nano_version+'/'
path_18 = analysis+'/Datasets/Files/data_18/'+nano_version+'/'

#----------------------------------------------------------------------------------------
# ID digits:
# 1st-2nd = 16(2016),17(2017),18(2018)                      # Year
# 3th-4th = 00(Data),01(MC-signal),02-13(MC-bkg)            # Group
# 5th-6th = 00(none),...                                    # Bkg -> Process
# 5th-6th = 00(none),11(250_30),12(250_40),55(1250_100)     # Signal -> Signal point
# 5th-6th = 00(none),01(A),02(B),03(C)                      # Data -> Era
# 7th     = 0,1,2,...                                       # Data taking interval (DTI)

# 2016 DTIs = 0(with "HIPM"/"APV")("pre-VFP"), 1(without "HIPM"/"APV")("pos-VFP")
#----------------------------------------------------------------------------------------


Data_Lep_0_16 = [
["Data_SingleEle_v1_B_0_16"]          + ['1600010', path_16+"SingleEle_B_v1.txt"],
["Data_SingleEle_v2_B_0_16"]          + ['1600020', path_16+"SingleEle_B_v2.txt"],
["Data_SingleEle_C_0_16"]             + ['1600030', path_16+"SingleEle_C.txt"],
["Data_SingleEle_D_0_16"]             + ['1600040', path_16+"SingleEle_D.txt"],
["Data_SingleEle_E_0_16"]             + ['1600050', path_16+"SingleEle_E.txt"],
["Data_SingleEle_F_0_16"]             + ['1600060', path_16+"SingleEle_F_HIPM.txt"],
["Data_DoubleEle_v1_B_0_16"]          + ['1600110', path_16+"DoubleEle_B_v1.txt"],
["Data_DoubleEle_v2_B_0_16"]          + ['1600120', path_16+"DoubleEle_B_v2.txt"],
["Data_DoubleEle_C_0_16"]             + ['1600130', path_16+"DoubleEle_C.txt"],
["Data_DoubleEle_D_0_16"]             + ['1600140', path_16+"DoubleEle_D.txt"],
["Data_DoubleEle_E_0_16"]             + ['1600150', path_16+"DoubleEle_E.txt"],
["Data_DoubleEle_F_0_16"]             + ['1600160', path_16+"DoubleEle_F_HIPM.txt"],
["Data_SingleMu_v1_B_0_16"]           + ['1600210', path_16+"SingleMu_B_v1.txt"],
["Data_SingleMu_v2_B_0_16"]           + ['1600220', path_16+"SingleMu_B_v2.txt"],
["Data_SingleMu_C_0_16"]              + ['1600230', path_16+"SingleMu_C.txt"],
["Data_SingleMu_D_0_16"]              + ['1600240', path_16+"SingleMu_D.txt"],
["Data_SingleMu_E_0_16"]              + ['1600250', path_16+"SingleMu_E.txt"],
["Data_SingleMu_F_0_16"]              + ['1600260', path_16+"SingleMu_F_HIPM.txt"],
["Data_DoubleMu_v1_B_0_16"]           + ['1600310', path_16+"DoubleMu_B_v1.txt"],
["Data_DoubleMu_v2_B_0_16"]           + ['1600320', path_16+"DoubleMu_B_v2.txt"],
["Data_DoubleMu_C_0_16"]              + ['1600330', path_16+"DoubleMu_C.txt"],
["Data_DoubleMu_D_0_16"]              + ['1600340', path_16+"DoubleMu_D.txt"],
["Data_DoubleMu_E_0_16"]              + ['1600350', path_16+"DoubleMu_E.txt"],
["Data_DoubleMu_F_0_16"]              + ['1600360', path_16+"DoubleMu_F_HIPM.txt"],
["Data_EleMu_v1_B_0_16"]              + ['1600410', path_16+"EleMu_B_v1.txt"],
["Data_EleMu_v2_B_0_16"]              + ['1600420', path_16+"EleMu_B_v2.txt"],
["Data_EleMu_C_0_16"]                 + ['1600430', path_16+"EleMu_C.txt"],
["Data_EleMu_D_0_16"]                 + ['1600440', path_16+"EleMu_D.txt"],
["Data_EleMu_E_0_16"]                 + ['1600450', path_16+"EleMu_E.txt"],
["Data_EleMu_F_0_16"]                 + ['1600460', path_16+"EleMu_F_HIPM.txt"],
]

Data_MET_0_16 = [
["Data_MET_v1_B_0_16"]                + ['1600510', path_16+"MET_B_v1.txt"],
["Data_MET_v2_B_0_16"]                + ['1600520', path_16+"MET_B_v2.txt"],
["Data_MET_C_0_16"]                   + ['1600530', path_16+"MET_C.txt"],
["Data_MET_D_0_16"]                   + ['1600540', path_16+"MET_D.txt"],
["Data_MET_E_0_16"]                   + ['1600550', path_16+"MET_E.txt"],
["Data_MET_F_0_16"]                   + ['1600560', path_16+"MET_F_HIPM.txt"],
]

Data_Lep_1_16 = [
["Data_SingleEle_F_1_16"]             + ['1600071', path_16+"SingleEle_F.txt"],
["Data_SingleEle_G_1_16"]             + ['1600081', path_16+"SingleEle_G.txt"],
["Data_SingleEle_H_1_16"]             + ['1600091', path_16+"SingleEle_H.txt"],
["Data_DoubleEle_F_1_16"]             + ['1600171', path_16+"DoubleEle_F.txt"],
["Data_DoubleEle_G_1_16"]             + ['1600181', path_16+"DoubleEle_G.txt"],
["Data_DoubleEle_H_1_16"]             + ['1600191', path_16+"DoubleEle_H.txt"],
["Data_SingleMu_F_1_16"]              + ['1600271', path_16+"SingleMu_F.txt"],
["Data_SingleMu_G_1_16"]              + ['1600281', path_16+"SingleMu_G.txt"],
["Data_SingleMu_H_1_16"]              + ['1600291', path_16+"SingleMu_H.txt"],
["Data_DoubleMu_F_1_16"]              + ['1600371', path_16+"DoubleMu_F.txt"],
["Data_DoubleMu_G_1_16"]              + ['1600381', path_16+"DoubleMu_G.txt"],
["Data_DoubleMu_H_1_16"]              + ['1600391', path_16+"DoubleMu_H.txt"],
["Data_EleMu_F_1_16"]                 + ['1600471', path_16+"EleMu_F.txt"],
["Data_EleMu_G_1_16"]                 + ['1600481', path_16+"EleMu_G.txt"],
["Data_EleMu_H_1_16"]                 + ['1600491', path_16+"EleMu_H.txt"],
]

Data_MET_1_16 = [
["Data_MET_F_1_16"]                   + ['1600571', path_16+"MET_F.txt"],
["Data_MET_G_1_16"]                   + ['1600581', path_16+"MET_G.txt"],
["Data_MET_H_1_16"]                   + ['1600591', path_16+"MET_H.txt"],
]

Data_Lep_0_17 = [
["Data_SingleEle_B_0_17"]             + ['1700020', path_17+"SingleEle_B.txt"],
["Data_SingleEle_C_0_17"]             + ['1700030', path_17+"SingleEle_C.txt"],
["Data_SingleEle_D_0_17"]             + ['1700040', path_17+"SingleEle_D.txt"],
["Data_SingleEle_E_0_17"]             + ['1700050', path_17+"SingleEle_E.txt"],
["Data_SingleEle_F_0_17"]             + ['1700060', path_17+"SingleEle_F.txt"],
["Data_DoubleEle_B_0_17"]             + ['1700120', path_17+"DoubleEle_B.txt"],
["Data_DoubleEle_C_0_17"]             + ['1700130', path_17+"DoubleEle_C.txt"],
["Data_DoubleEle_D_0_17"]             + ['1700140', path_17+"DoubleEle_D.txt"],
["Data_DoubleEle_E_0_17"]             + ['1700150', path_17+"DoubleEle_E.txt"],
["Data_DoubleEle_F_0_17"]             + ['1700160', path_17+"DoubleEle_F.txt"],
["Data_SingleMu_B_0_17"]              + ['1700220', path_17+"SingleMu_B.txt"],
["Data_SingleMu_C_0_17"]              + ['1700230', path_17+"SingleMu_C.txt"],
["Data_SingleMu_D_0_17"]              + ['1700240', path_17+"SingleMu_D.txt"],
["Data_SingleMu_E_0_17"]              + ['1700250', path_17+"SingleMu_E.txt"],
["Data_SingleMu_F_0_17"]              + ['1700260', path_17+"SingleMu_F.txt"],
["Data_DoubleMu_B_0_17"]              + ['1700320', path_17+"DoubleMu_B.txt"],
["Data_DoubleMu_C_0_17"]              + ['1700330', path_17+"DoubleMu_C.txt"],
["Data_DoubleMu_D_0_17"]              + ['1700340', path_17+"DoubleMu_D.txt"],
["Data_DoubleMu_E_0_17"]              + ['1700350', path_17+"DoubleMu_E.txt"],
["Data_DoubleMu_F_0_17"]              + ['1700360', path_17+"DoubleMu_F.txt"],
["Data_EleMu_B_0_17"]                 + ['1700420', path_17+"EleMu_B.txt"],
["Data_EleMu_C_0_17"]                 + ['1700430', path_17+"EleMu_C.txt"],
["Data_EleMu_D_0_17"]                 + ['1700440', path_17+"EleMu_D.txt"],
["Data_EleMu_E_0_17"]                 + ['1700450', path_17+"EleMu_E.txt"],
["Data_EleMu_F_0_17"]                 + ['1700460', path_17+"EleMu_F.txt"],
]

Data_MET_0_17 = [
["Data_MET_B_0_17"]                   + ['1700520', path_17+"MET_B.txt"],
["Data_MET_C_0_17"]                   + ['1700530', path_17+"MET_C.txt"],
["Data_MET_D_0_17"]                   + ['1700540', path_17+"MET_D.txt"],
["Data_MET_E_0_17"]                   + ['1700550', path_17+"MET_E.txt"],
["Data_MET_F_0_17"]                   + ['1700560', path_17+"MET_F.txt"],
]

Data_Lep_0_18 = [
["Data_Electrons_A_0_18"]             + ['1800010', path_18+"Ele_A.txt"],
["Data_Electrons_B_0_18"]             + ['1800020', path_18+"Ele_B.txt"],
["Data_Electrons_C_0_18"]             + ['1800030', path_18+"Ele_C.txt"],
["Data_Electrons_D_0_18"]             + ['1800040', path_18+"Ele_D.txt"],
["Data_SingleMu_A_0_18"]              + ['1800210', path_18+"SingleMu_A.txt"],
["Data_SingleMu_B_0_18"]              + ['1800220', path_18+"SingleMu_B.txt"],
["Data_SingleMu_C_0_18"]              + ['1800230', path_18+"SingleMu_C.txt"],
["Data_SingleMu_D_0_18"]              + ['1800240', path_18+"SingleMu_D.txt"],
["Data_DoubleMu_A_0_18"]              + ['1800310', path_18+"DoubleMu_A.txt"],
["Data_DoubleMu_B_0_18"]              + ['1800320', path_18+"DoubleMu_B.txt"],
["Data_DoubleMu_C_0_18"]              + ['1800330', path_18+"DoubleMu_C.txt"],
["Data_DoubleMu_D_0_18"]              + ['1800340', path_18+"DoubleMu_D.txt"],
["Data_EleMu_A_0_18"]                 + ['1800410', path_18+"EleMu_A.txt"],
["Data_EleMu_B_0_18"]                 + ['1800420', path_18+"EleMu_B.txt"],
["Data_EleMu_C_0_18"]                 + ['1800430', path_18+"EleMu_C.txt"],
["Data_EleMu_D_0_18"]                 + ['1800440', path_18+"EleMu_D.txt"],
]

Data_MET_0_18 = [
["Data_MET_A_0_18"]                   + ['1800510', path_18+"MET_A.txt"],
["Data_MET_B_0_18"]                   + ['1800520', path_18+"MET_B.txt"],
["Data_MET_C_0_18"]                   + ['1800530', path_18+"MET_C.txt"],
["Data_MET_D_0_18"]                   + ['1800540', path_18+"MET_D.txt"],
]

