import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as pat
import copy
import os
import re
from matplotlib.ticker import AutoMinorLocator
#from scipy.optimize import curve_fit

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 99999999)

class harvester:
    """
    Gather information from distributions and systematic uncertainties
    """

    #==============================================================================================================
    def __init__(self, region, var, period, bins, backgrounds, bkg_labels, bkg_colors, signal, signal_label, signal_color, regions_labels, systematics, smooth_factor=1, smooth_repeat=2, symmetric_factor=999999999999, analysis_name=None, signal_cross_section=None, groups=None, full_year_correlated=[], processes_scale_factors=None, allow_syst_shape_only=True, allow_single_bin_template=False):
        
        systematics_temp = systematics.copy()
        for iSource in systematics.keys():
            if len(systematics[iSource][3]) > 0:
                isub = 0
                for iSubSource in systematics[iSource][3]:
                    systematics_temp[iSource+"_"+iSubSource] = [systematics[iSource][0]+isub, 2, systematics[iSource][2], systematics[iSource][0]]
                    isub += 1
                del systematics_temp[iSource]
        systematics = systematics_temp.copy()
        
        if groups is not None:
            for igroup in groups:
                if isinstance(igroup, list) and len(igroup) > 1:
                    systematics_loop = systematics.copy()
                    for iSource in systematics_loop.keys():
                        if igroup[0] in iSource and iSource not in igroup[1:]:
                            del systematics[iSource]
        
        #reorder sys_sources
        list_keys = []
        list_0 = []
        list_1 = []
        list_2 = []
        list_3 = []
        for sys_source in systematics.keys():
            list_keys.append(sys_source)
            list_0.append(systematics[sys_source][0])
            list_1.append(systematics[sys_source][1])
            list_2.append(systematics[sys_source][2])
            list_3.append(systematics[sys_source][3])
        list_0, list_1, list_2, list_3, list_keys = (list(t) for t in zip(*sorted(zip(list_0, list_1, list_2, list_3, list_keys))))
        list_values = []
        for i in range(len(list_keys)):
            list_values.append([list_0[i], list_1[i], list_2[i], list_3[i]])
        systematics = dict(zip(list_keys, list_values))
        #print(systematics)
        
        datasets = [signal] + backgrounds 
        labels =  [signal_label] + bkg_labels
        colors =  [signal_color] + bkg_colors
        
        # Maximum of 14 systematics
        # Get maximum source ID:
        ID_max = 0
        for iSource in systematics.keys():
            ID = systematics[iSource][0]
            if ID_max < ID:
                ID_max = ID
    
        #Initialize tables (first index is source ID, second index is universe, and third index is process)
        self.hist_table3D = []
        self.unc_table3D = []       # Statistical unc. in the histograms
        self.norm_table3D = []      # Normalization factor for Scales, PDF, ...
        for i in range(ID_max+1):
            self.hist_table3D.append([0, 0])
            self.unc_table3D.append([0, 0])
            self.norm_table3D.append([0, 0])
        
        self.XS_syst_list = []
        for jSource in systematics.keys():    
            if jSource[-2:] == "XS":   
                self.XS_syst_list.append(jSource)
        
        self.sys_IDs = []
        self.sys_labels = []
        self.sys_colors = ["#7F7F7F", "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#BCBD22", "#17BECF", "#C7C7C7", "#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5", "#C49C94", "#F7B6D2", "#DBDB8D", "#9EDAE5", "dimgrey", "limegreen", "blue", "red", "darkgoldenrod", "darkgreen", "darkorange", "skyblue", "darkviolet", "aqua", "gold", "pink", "magenta", "darkkhaki", "lime", "greenyellow", "sandybrown", "brown", "mediumpurple", "forestgreen", "fuchsia", "goldenrod", "springgreen", "tomato", "royalblue", "chocolate", "aquamarine", "orange"]
        
        #self.smear_factor = smear_factor
        self.groups = groups
        self.full_year_correlated = full_year_correlated
        self.allow_single_bin_template = allow_single_bin_template
        
        #--------------------------------------------------------------------------------
        for iProcType in range(len(datasets)):
            item_type = type(datasets[iProcType]).__name__
            if item_type != "list": 
                datasets[iProcType] = [datasets[iProcType]]
        
        self.regions_labels = regions_labels
        self.processes = labels
        
        for j in range(len(self.regions_labels)):
            self.regions_labels[j] = re.sub('[^A-Za-z0-9]+', '', self.regions_labels[j])
        
        for j in range(len(self.processes)):
            self.processes[j] = re.sub('[^A-Za-z0-9]+', '', self.processes[j])
        
        ProcType_has_norm_param = [False]*len(datasets)
        if allow_syst_shape_only:
            syst_shape_only = [value for value in self.processes if value+"CR" in self.regions_labels]
            if signal_cross_section is not None:
                syst_shape_only.append(self.processes[0])
            #print(syst_shape_only)
            for iProcType in range(len(datasets)):
                if self.processes[iProcType] in syst_shape_only:
                    ProcType_has_norm_param[iProcType] = True
        print(self.processes)
        print(ProcType_has_norm_param)
        
        #----------------------------------------------------------------------------------
        # Get the normalization factors from the nominal histograms
        list_ProcType_sum = []
        for iProcType in range(len(datasets)):  # loop in the proc_type_lists
            Hist_ProcType = np.zeros(len(bins)-1)
            for iProcDic in range(len(datasets[iProcType])):  # loop in the proc_dictionaries inside the lists
                proc_dic = datasets[iProcType][iProcDic][var+"_"+str(region)+"_0_0"]
                Hist_raw = proc_dic["Hist"]
                Start_raw = proc_dic["Start"]
                End_raw = proc_dic["End"]
                Nbins_raw = proc_dic["Nbins"]
                Delta_raw = (End_raw - Start_raw)/Nbins_raw
                
                Hist_new = [0]*(len(bins)-1)
                            
                for iRawBin in range(Nbins_raw):
                    inf_raw = Start_raw + iRawBin*Delta_raw
                    sup_raw = Start_raw + (iRawBin+1)*Delta_raw
                    for iNewBin in range(len(Hist_new)):
                        if( (inf_raw >= bins[iNewBin]) and (sup_raw <= bins[iNewBin+1]) ):
                            Hist_new[iNewBin] = Hist_new[iNewBin] + Hist_raw[iRawBin]
            
                for iNewBin in range(len(Hist_new)):
                    if Hist_new[iNewBin] < 0:
                        Hist_new[iNewBin] = 0
                
                Hist_ProcType = Hist_ProcType + np.array(Hist_new)
                
            if Hist_ProcType.sum() == 0:
                Hist_ProcType[0] = 0.0001
            if (signal_cross_section is not None) and (iProcType == 0):
                Hist_ProcType = Hist_ProcType/signal_cross_section
            if processes_scale_factors is not None:
                for ProcSF in processes_scale_factors:
                    if iProcType == ProcSF[0]:
                        Hist_ProcType = Hist_ProcType*ProcSF[1]
                        break

                
            list_ProcType_sum.append(Hist_ProcType.sum())
            
        #----------------------------------------------------------------------------------
        for iSource in systematics.keys():    # loop in the systematic sources
            #--------------------------------------------------------------------------------
            if iSource == "CV":  
            
                list_Hist_ProcType = []
                list_Unc_ProcType = []
            
                # Process SF uncertainty
                list_Hist_ProcType_ProcSF_up = []
                list_Hist_ProcType_ProcSF_down = []

                # Luminosity uncertainty 
                list_Hist_ProcType_lumi_up = []
                list_Hist_ProcType_lumi_down = []
                    
                # Statistical uncertainty 
                list_Hist_ProcType_stat_up = []
                list_Hist_ProcType_stat_down = []
            
                for iProcType in range(len(datasets)):  # loop in the proc_type_lists
                
                    Hist_ProcType = np.zeros(len(bins)-1)
                    Unc_ProcType = np.zeros(len(bins)-1)  # Stat. Uncertainty of Hist
                    for iProcDic in range(len(datasets[iProcType])):  # loop in the proc_dictionaries inside the lists
                        proc_dic = datasets[iProcType][iProcDic][var+"_"+str(region)+"_0_0"]
                        Hist_raw = proc_dic["Hist"]
                        Unc_raw = proc_dic["Unc"]
                        Start_raw = proc_dic["Start"]
                        End_raw = proc_dic["End"]
                        Nbins_raw = proc_dic["Nbins"]
                        LumiUnc = proc_dic["LumiUnc"]*0.01
                        LumiValuesUnc = np.array(proc_dic["LumiValuesUnc"])*0.01
                        LumiTagsUnc = proc_dic["LumiTagsUnc"]
                        Delta_raw = (End_raw - Start_raw)/Nbins_raw
                    
                        Hist_new = [0]*(len(bins)-1)
                        Unc_new = [0]*(len(bins)-1)
                                    
                        for iRawBin in range(Nbins_raw):
                            inf_raw = Start_raw + iRawBin*Delta_raw
                            sup_raw = Start_raw + (iRawBin+1)*Delta_raw
                            for iNewBin in range(len(Hist_new)):
                                if( (inf_raw >= bins[iNewBin]) and (sup_raw <= bins[iNewBin+1]) ):
                                    Hist_new[iNewBin] = Hist_new[iNewBin] + Hist_raw[iRawBin]
                                    Unc_new[iNewBin] = np.sqrt( Unc_new[iNewBin]**2 + Unc_raw[iRawBin]**2 )
                    
                        for iNewBin in range(len(Hist_new)):
                            if Hist_new[iNewBin] < 0:
                                Hist_new[iNewBin] = 0
                                Unc_new[iNewBin] = 0
                        
                        Hist_ProcType = Hist_ProcType + np.array(Hist_new)
                        Unc_ProcType = np.sqrt(Unc_ProcType**2 + np.array(Unc_new)**2)
                
                    if Hist_ProcType.sum() == 0:
                        Hist_ProcType[0] = 0.0001
                        Unc_ProcType[0] = 0.0001
                    if (signal_cross_section is not None) and (iProcType == 0):
                        Hist_ProcType = Hist_ProcType/signal_cross_section
                        Unc_ProcType = Unc_ProcType/signal_cross_section
                    if processes_scale_factors is not None:
                        for ProcSF in processes_scale_factors:
                            if iProcType == ProcSF[0]:
                                Hist_ProcType = Hist_ProcType*ProcSF[1]
                                Unc_ProcType = Unc_ProcType*ProcSF[1]
                                break
                
                    list_Hist_ProcType.append(Hist_ProcType)
                    list_Unc_ProcType.append(Unc_ProcType)    
                    
                    # Luminosity uncertainty 
                    Hist_ProcType_lumi_up = Hist_ProcType*(1 + LumiUnc)
                    Hist_ProcType_lumi_down = Hist_ProcType*(1 - LumiUnc)
                    list_Hist_ProcType_lumi_up.append(Hist_ProcType_lumi_up)
                    list_Hist_ProcType_lumi_down.append(Hist_ProcType_lumi_down)    
                    
                    # Statistical uncertainty 
                    Hist_ProcType_stat_up = Hist_ProcType + Unc_ProcType
                    Hist_ProcType_stat_down = Hist_ProcType - Unc_ProcType
                    Hist_ProcType_stat_down[Hist_ProcType_stat_down < 0] = 0.0000001  # avoid down syst with negative number of events
                    list_Hist_ProcType_stat_up.append(Hist_ProcType_stat_up)
                    list_Hist_ProcType_stat_down.append(Hist_ProcType_stat_down)

                    # Process SF uncertainty
                    if processes_scale_factors is not None:
                        Has_SF = False
                        for ProcSF in processes_scale_factors:
                            if iProcType == ProcSF[0]:
                                Hist_ProcType_ProcSF_up = Hist_ProcType*(1 + ProcSF[2]/ProcSF[1])
                                Hist_ProcType_ProcSF_down = Hist_ProcType*(1 - ProcSF[2]/ProcSF[1])
                                Has_SF = True
                                break
                        if Has_SF:
                            list_Hist_ProcType_ProcSF_up.append(Hist_ProcType_ProcSF_up)
                            list_Hist_ProcType_ProcSF_down.append(Hist_ProcType_ProcSF_down)
                        else:
                            list_Hist_ProcType_ProcSF_up.append(Hist_ProcType)
                            list_Hist_ProcType_ProcSF_down.append(Hist_ProcType)
                
                self.LumiUnc = LumiUnc
                self.LumiValuesUnc = LumiValuesUnc
                self.LumiTagsUnc = LumiTagsUnc
                
                self.hist_table3D[0][0] = list_Hist_ProcType
                self.unc_table3D[0][0] = list_Unc_ProcType
            
                if processes_scale_factors is None:
                    self.hist_table3D.append([list_Hist_ProcType_lumi_down, list_Hist_ProcType_lumi_up])
                    self.hist_table3D.append([list_Hist_ProcType_stat_down, list_Hist_ProcType_stat_up])
                    self.sys_IDs.append(ID_max+2)
                    self.sys_IDs.append(ID_max+1)
                    self.sys_labels.append("Stat")
                    self.sys_labels.append("Lumi")
                else:
                    self.hist_table3D.append([list_Hist_ProcType_ProcSF_down, list_Hist_ProcType_ProcSF_up])
                    self.hist_table3D.append([list_Hist_ProcType_lumi_down, list_Hist_ProcType_lumi_up])
                    self.hist_table3D.append([list_Hist_ProcType_stat_down, list_Hist_ProcType_stat_up])
                    self.sys_IDs.append(ID_max+3)
                    self.sys_IDs.append(ID_max+2)
                    self.sys_IDs.append(ID_max+1)
                    self.sys_labels.append("Stat")
                    self.sys_labels.append("Lumi")
                    self.sys_labels.append("ProcSF")
        
            #--------------------------------------------------------------------------------
            elif( "_" in iSource ):    
            
                for iUniverse in range(2):  # loop in the universes
                    list_Hist_ProcType = []
                    list_Unc_ProcType = []
                    for iProcType in range(len(datasets)):  # loop in the proc_type_lists
                
                        Hist_ProcType = np.zeros(len(bins)-1)
                        Unc_ProcType = np.zeros(len(bins)-1)  # Stat. Uncertainty of Hist
                        for iProcDic in range(len(datasets[iProcType])):  # loop in the proc_dictionaries inside the lists
                            #print(systematics[iSource][0], iUniverse, iProcType, iProcDic)
                            eff_iUniverse = iUniverse + 2*(systematics[iSource][0] - systematics[iSource][3])
                            proc_dic = datasets[iProcType][iProcDic][var+"_"+str(region)+"_"+str(systematics[iSource][3])+"_"+str(eff_iUniverse)]
                            Hist_raw = proc_dic["Hist"]
                            Unc_raw = proc_dic["Unc"]
                            Start_raw = proc_dic["Start"]
                            End_raw = proc_dic["End"]
                            Nbins_raw = proc_dic["Nbins"]
                            Delta_raw = (End_raw - Start_raw)/Nbins_raw
                    
                            Hist_new = [0]*(len(bins)-1)
                            Unc_new = [0]*(len(bins)-1)
                                    
                            for iRawBin in range(Nbins_raw):
                                inf_raw = Start_raw + iRawBin*Delta_raw
                                sup_raw = Start_raw + (iRawBin+1)*Delta_raw
                                for iNewBin in range(len(Hist_new)):
                                    if( (inf_raw >= bins[iNewBin]) and (sup_raw <= bins[iNewBin+1]) ):
                                        Hist_new[iNewBin] = Hist_new[iNewBin] + Hist_raw[iRawBin]
                                        Unc_new[iNewBin] = np.sqrt( Unc_new[iNewBin]**2 + Unc_raw[iRawBin]**2 )
                    
                            for iNewBin in range(len(Hist_new)):
                                if Hist_new[iNewBin] < 0:
                                    Hist_new[iNewBin] = 0
                                    Unc_new[iNewBin] = 0
                    
                            Hist_ProcType = Hist_ProcType + np.array(Hist_new)
                            Unc_ProcType = np.sqrt(Unc_ProcType**2 + np.array(Unc_new)**2)
                        #print("iSource", iSource)
                        #print("Hist_ProcType.sum", Hist_ProcType.sum())
                        if Hist_ProcType.sum() == 0:
                            Hist_ProcType[0] = 0.0001
                            Unc_ProcType[0] = 0.0001
                        if (signal_cross_section is not None) and (iProcType == 0):
                            Hist_ProcType = Hist_ProcType/signal_cross_section
                            Unc_ProcType = Unc_ProcType/signal_cross_section
                        if processes_scale_factors is not None:
                            for ProcSF in processes_scale_factors:
                                if iProcType == ProcSF[0]:
                                    Hist_ProcType = Hist_ProcType*ProcSF[1]
                                    Unc_ProcType = Unc_ProcType*ProcSF[1]
                                    break
                       
                        list_Hist_ProcType.append(Hist_ProcType)
                        list_Unc_ProcType.append(Unc_ProcType)
        
                    self.hist_table3D[systematics[iSource][0]][iUniverse] = list_Hist_ProcType
                    self.unc_table3D[systematics[iSource][0]][iUniverse] = list_Unc_ProcType

                self.sys_IDs.append(systematics[iSource][0])
                self.sys_labels.append(iSource)

                # JES->MET propagation shape normalized unc. plus JES->Jets propagation unc. when MET recoil is applied
                if( iSource == "JES_Total" and ('Recoil' in systematics) ):

                    for iUniverse in range(2):  # loop in the universes
                        iUniverseEff = iUniverse+4
                        list_Hist_ProcType = []
                        for iProcType in range(len(datasets)):  # loop in the proc_type_lists
                            hist_proc_name = var+"_"+str(region)+"_"+str(systematics['Recoil'][0])+"_"+str(iUniverseEff)
                            if hist_proc_name in datasets[iProcType][0]:
                                Hist_ProcType = np.zeros(len(bins)-1)
                                for iProcDic in range(len(datasets[iProcType])):  # loop in the proc_dictionaries inside the lists
                                    #print(systematics['Recoil'][0], iUniverseEff, iProcType, iProcDic)
                                    proc_dic = datasets[iProcType][iProcDic][var+"_"+str(region)+"_"+str(systematics['Recoil'][0])+"_"+str(iUniverseEff)]
                                    Hist_raw = proc_dic["Hist"]
                                    Start_raw = proc_dic["Start"]
                                    End_raw = proc_dic["End"]
                                    Nbins_raw = proc_dic["Nbins"]
                                    Delta_raw = (End_raw - Start_raw)/Nbins_raw

                                    Hist_new = [0]*(len(bins)-1)

                                    for iRawBin in range(Nbins_raw):
                                        inf_raw = Start_raw + iRawBin*Delta_raw
                                        sup_raw = Start_raw + (iRawBin+1)*Delta_raw
                                        for iNewBin in range(len(Hist_new)):
                                            if( (inf_raw >= bins[iNewBin]) and (sup_raw <= bins[iNewBin+1]) ):
                                                Hist_new[iNewBin] = Hist_new[iNewBin] + Hist_raw[iRawBin]

                                    for iNewBin in range(len(Hist_new)):
                                        if Hist_new[iNewBin] < 0:
                                            Hist_new[iNewBin] = 0

                                    Hist_ProcType = Hist_ProcType + np.array(Hist_new)

                                if Hist_ProcType.sum() == 0:
                                    Hist_ProcType[0] = 0.0001
                                if (signal_cross_section is not None) and (iProcType == 0):
                                    Hist_ProcType = Hist_ProcType/signal_cross_section
                                if processes_scale_factors is not None:
                                    for ProcSF in processes_scale_factors:
                                        if iProcType == ProcSF[0]:
                                            Hist_ProcType = Hist_ProcType*ProcSF[1]
                                            break

                                list_Hist_ProcType.append(Hist_ProcType)
                            else:
                                list_Hist_ProcType.append([])

                        if iUniverse == 0:
                            list_Hist_ProcType_down = list_Hist_ProcType
                        if iUniverse == 1:
                            list_Hist_ProcType_up = list_Hist_ProcType

                    for iProcType in range(len(datasets)):  # loop in the proc_type_lists
                        if len(list_Hist_ProcType_up[iProcType]) > 0:
                            sf_up = list_ProcType_sum[iProcType]/self.hist_table3D[systematics[iSource][0]][1][iProcType].sum()
                            sf_down = list_ProcType_sum[iProcType]/self.hist_table3D[systematics[iSource][0]][0][iProcType].sum()

                            hist_table3D_JES_METJets_shape_down = self.hist_table3D[systematics[iSource][0]][0][iProcType]*sf_down
                            hist_table3D_JES_METJets_shape_up = self.hist_table3D[systematics[iSource][0]][1][iProcType]*sf_up

                            hist_table3D_JES_Jets_down = list_Hist_ProcType_down[iProcType]
                            hist_table3D_JES_Jets_up = list_Hist_ProcType_up[iProcType]

                            hist_table3D_nominal = self.hist_table3D[0][0][iProcType]

                            variation_JES_METJets_shape_down = hist_table3D_JES_METJets_shape_down - hist_table3D_nominal
                            variation_JES_METJets_shape_up = hist_table3D_JES_METJets_shape_up - hist_table3D_nominal

                            variation_JES_Jets_down = hist_table3D_JES_Jets_down - hist_table3D_nominal
                            variation_JES_Jets_up = hist_table3D_JES_Jets_up - hist_table3D_nominal

                            for ibin in range(len(hist_table3D_JES_Jets_up)):
                                if np.abs(variation_JES_Jets_up[ibin]) >= np.abs(variation_JES_METJets_shape_up[ibin]):
                                    self.hist_table3D[systematics[iSource][0]][1][iProcType][ibin] = hist_table3D_JES_Jets_up[ibin]
                                else:
                                    self.hist_table3D[systematics[iSource][0]][1][iProcType][ibin] = hist_table3D_JES_METJets_shape_up[ibin]

                                if np.abs(variation_JES_Jets_down[ibin]) >= np.abs(variation_JES_METJets_shape_down[ibin]):
                                    self.hist_table3D[systematics[iSource][0]][0][iProcType][ibin] = hist_table3D_JES_Jets_down[ibin]
                                else:
                                    self.hist_table3D[systematics[iSource][0]][0][iProcType][ibin] = hist_table3D_JES_METJets_shape_down[ibin]

        
            #--------------------------------------------------------------------------------
            #elif( systematics[iSource][1] == 2 ):  
            elif( (iSource != "CV") and (iSource != "Scales") and (iSource != "PDF") and (iSource != "Recoil") and (iSource != "TopPt") and (iSource != "WPt") and ("_" not in iSource) ):
            
                for iUniverse in range(2):  # loop in the universes
                    list_Hist_ProcType = []
                    list_Unc_ProcType = []
                    for iProcType in range(len(datasets)):  # loop in the proc_type_lists
                
                        Hist_ProcType = np.zeros(len(bins)-1)
                        Unc_ProcType = np.zeros(len(bins)-1)  # Stat. Uncertainty of Hist
                        for iProcDic in range(len(datasets[iProcType])):  # loop in the proc_dictionaries inside the lists
                            #print(systematics[iSource][0], iUniverse, iProcType, iProcDic)
                            proc_dic = datasets[iProcType][iProcDic][var+"_"+str(region)+"_"+str(systematics[iSource][0])+"_"+str(iUniverse)]
                            Hist_raw = proc_dic["Hist"]
                            Unc_raw = proc_dic["Unc"]
                            Start_raw = proc_dic["Start"]
                            End_raw = proc_dic["End"]
                            Nbins_raw = proc_dic["Nbins"]
                            Delta_raw = (End_raw - Start_raw)/Nbins_raw
                    
                            Hist_new = [0]*(len(bins)-1)
                            Unc_new = [0]*(len(bins)-1)
                                    
                            for iRawBin in range(Nbins_raw):
                                inf_raw = Start_raw + iRawBin*Delta_raw
                                sup_raw = Start_raw + (iRawBin+1)*Delta_raw
                                for iNewBin in range(len(Hist_new)):
                                    if( (inf_raw >= bins[iNewBin]) and (sup_raw <= bins[iNewBin+1]) ):
                                        Hist_new[iNewBin] = Hist_new[iNewBin] + Hist_raw[iRawBin]
                                        Unc_new[iNewBin] = np.sqrt( Unc_new[iNewBin]**2 + Unc_raw[iRawBin]**2 )
                    
                            for iNewBin in range(len(Hist_new)):
                                if Hist_new[iNewBin] < 0:
                                    Hist_new[iNewBin] = 0
                                    Unc_new[iNewBin] = 0
                    
                            Hist_ProcType = Hist_ProcType + np.array(Hist_new)
                            Unc_ProcType = np.sqrt(Unc_ProcType**2 + np.array(Unc_new)**2)
                       
                        if Hist_ProcType.sum() == 0:
                            Hist_ProcType[0] = 0.0001
                            Unc_ProcType[0] = 0.0001
                        if (signal_cross_section is not None) and (iProcType == 0):
                            Hist_ProcType = Hist_ProcType/signal_cross_section
                            Unc_ProcType = Unc_ProcType/signal_cross_section
                        if processes_scale_factors is not None:
                            for ProcSF in processes_scale_factors:
                                if iProcType == ProcSF[0]:
                                    Hist_ProcType = Hist_ProcType*ProcSF[1]
                                    Unc_ProcType = Unc_ProcType*ProcSF[1]
                                    break
                       
                        list_Hist_ProcType.append(Hist_ProcType)
                        list_Unc_ProcType.append(Unc_ProcType)
        
                    self.hist_table3D[systematics[iSource][0]][iUniverse] = list_Hist_ProcType
                    self.unc_table3D[systematics[iSource][0]][iUniverse] = list_Unc_ProcType

                self.sys_IDs.append(systematics[iSource][0])
                self.sys_labels.append(iSource)

         
            #--------------------------------------------------------------------------------
            elif( (iSource == "Scales") or (iSource == "Recoil") ):  # Envelop
                
                #----------------------------------------------------------------------------------
                # Get the normalization factors from the "Scales" histograms
                if iSource == "Scales":
                    list_Hist_ProcType_norm_up = []
                    list_Hist_ProcType_norm_down = []
                    for iUniverse in range(systematics[iSource][1]):  # loop in the universes
                        list_Hist_ProcType = []
                        for iProcType in range(len(datasets)):  # loop in the proc_type_lists
                            Hist_ProcType = np.zeros(len(bins)-1)
                            for iProcDic in range(len(datasets[iProcType])):  # loop in the proc_dictionaries inside the lists
                                Hist_name = var+"_"+str(region)+"_"+str(systematics[iSource][0])+"_"+str(iUniverse)
                                if Hist_name in datasets[iProcType][iProcDic]:
                                    proc_dic = datasets[iProcType][iProcDic][Hist_name]
                                else:
                                    proc_dic = datasets[iProcType][iProcDic][var+"_"+str(region)+"_0_0"]
                                Hist_raw = proc_dic["Hist"]
                                Start_raw = proc_dic["Start"]
                                End_raw = proc_dic["End"]
                                Nbins_raw = proc_dic["Nbins"]
                                Delta_raw = (End_raw - Start_raw)/Nbins_raw

                                Hist_new = [0]*(len(bins)-1)

                                for iRawBin in range(Nbins_raw):
                                    inf_raw = Start_raw + iRawBin*Delta_raw
                                    sup_raw = Start_raw + (iRawBin+1)*Delta_raw
                                    for iNewBin in range(len(Hist_new)):
                                        if( (inf_raw >= bins[iNewBin]) and (sup_raw <= bins[iNewBin+1]) ):
                                            Hist_new[iNewBin] = Hist_new[iNewBin] + Hist_raw[iRawBin]

                                for iNewBin in range(len(Hist_new)):
                                    if Hist_new[iNewBin] < 0:
                                        Hist_new[iNewBin] = 0


                                Hist_ProcType = Hist_ProcType + np.array(Hist_new)

                            if Hist_ProcType.sum() == 0:
                                Hist_ProcType[0] = 0.0001
                            if (signal_cross_section is not None) and (iProcType == 0):
                                Hist_ProcType = Hist_ProcType/signal_cross_section
                            if processes_scale_factors is not None:
                                for ProcSF in processes_scale_factors:
                                    if iProcType == ProcSF[0]:
                                        Hist_ProcType = Hist_ProcType*ProcSF[1]
                                        break

                            list_Hist_ProcType.append(Hist_ProcType)

                        if iUniverse == 0:
                            list_Hist_ProcType_norm_up = copy.deepcopy(list_Hist_ProcType)
                            list_Hist_ProcType_norm_down = copy.deepcopy(list_Hist_ProcType)
                        else:
                            for ihist in range(len(list_Hist_ProcType)):
                                for ibin in range(len(list_Hist_ProcType[ihist])):
                                    if list_Hist_ProcType[ihist][ibin] > list_Hist_ProcType_norm_up[ihist][ibin]:
                                        list_Hist_ProcType_norm_up[ihist][ibin] = list_Hist_ProcType[ihist][ibin]
                                    if list_Hist_ProcType[ihist][ibin] < list_Hist_ProcType_norm_down[ihist][ibin]:
                                        list_Hist_ProcType_norm_down[ihist][ibin] = list_Hist_ProcType[ihist][ibin]

                    list_ProcType_norm_up_sum = []
                    list_ProcType_norm_down_sum = []
                    for iProcType in range(len(datasets)):  # loop in the proc_type_lists
                        list_ProcType_norm_up_sum.append(list_Hist_ProcType_norm_up[iProcType].sum())
                        list_ProcType_norm_down_sum.append(list_Hist_ProcType_norm_down[iProcType].sum())


                #----------------------------------------------------------------------------------
                list_Hist_ProcType_up = []
                list_Unc_ProcType_up = []
                list_Hist_ProcType_down = []
                list_Unc_ProcType_down = []
                #print(systematics[iSource][1])
                if iSource == "Recoil":
                    Nuniverses = systematics[iSource][1] - 2
                else:
                    Nuniverses = systematics[iSource][1]
                for iUniverse in range(Nuniverses):  # loop in the universes
                    
                    list_Hist_ProcType = []
                    list_Unc_ProcType = []
                    for iProcType in range(len(datasets)):  # loop in the proc_type_lists
                    
                        Hist_ProcType = np.zeros(len(bins)-1)
                        Unc_ProcType = np.zeros(len(bins)-1)  # Stat. Uncertainty of Hist
                        for iProcDic in range(len(datasets[iProcType])):  # loop in the proc_dictionaries inside the lists
                            Hist_name = var+"_"+str(region)+"_"+str(systematics[iSource][0])+"_"+str(iUniverse)
                            if Hist_name in datasets[iProcType][iProcDic]:
                                proc_dic = datasets[iProcType][iProcDic][Hist_name]
                            else:
                                proc_dic = datasets[iProcType][iProcDic][var+"_"+str(region)+"_0_0"]
                            Hist_raw = proc_dic["Hist"]
                            Unc_raw = proc_dic["Unc"]
                            Start_raw = proc_dic["Start"]
                            End_raw = proc_dic["End"]
                            Nbins_raw = proc_dic["Nbins"]
                            Delta_raw = (End_raw - Start_raw)/Nbins_raw
                    
                            Hist_new = [0]*(len(bins)-1)
                            Unc_new = [0]*(len(bins)-1)
                                    
                            for iRawBin in range(Nbins_raw):
                                inf_raw = Start_raw + iRawBin*Delta_raw
                                sup_raw = Start_raw + (iRawBin+1)*Delta_raw
                                for iNewBin in range(len(Hist_new)):
                                    if( (inf_raw >= bins[iNewBin]) and (sup_raw <= bins[iNewBin+1]) ):
                                        Hist_new[iNewBin] = Hist_new[iNewBin] + Hist_raw[iRawBin]
                                        Unc_new[iNewBin] = np.sqrt( Unc_new[iNewBin]**2 + Unc_raw[iRawBin]**2 )
                    
                            for iNewBin in range(len(Hist_new)):
                                if Hist_new[iNewBin] < 0:
                                    Hist_new[iNewBin] = 0
                                    Unc_new[iNewBin] = 0
                    
                            Hist_ProcType = Hist_ProcType + np.array(Hist_new)
                            Unc_ProcType = np.sqrt(Unc_ProcType**2 + np.array(Unc_new)**2)
                        
                        if Hist_ProcType.sum() == 0:
                            Hist_ProcType[0] = 0.0001
                            Unc_ProcType[0] = 0.0001
                        if (signal_cross_section is not None) and (iProcType == 0):
                            Hist_ProcType = Hist_ProcType/signal_cross_section
                            Unc_ProcType = Unc_ProcType/signal_cross_section
                        if processes_scale_factors is not None:
                            for ProcSF in processes_scale_factors:
                                if iProcType == ProcSF[0]:
                                    Hist_ProcType = Hist_ProcType*ProcSF[1]
                                    Unc_ProcType = Unc_ProcType*ProcSF[1]
                                    break
                        
                        list_Hist_ProcType.append(Hist_ProcType)
                        list_Unc_ProcType.append(Unc_ProcType)
                
                    if iUniverse == 0:
                        list_Hist_ProcType_up = copy.deepcopy(list_Hist_ProcType)
                        list_Unc_ProcType_up = copy.deepcopy(list_Unc_ProcType)
                        list_Hist_ProcType_down = copy.deepcopy(list_Hist_ProcType)
                        list_Unc_ProcType_down = copy.deepcopy(list_Unc_ProcType)
                    else:
                        for ihist in range(len(list_Hist_ProcType)):
                            for ibin in range(len(list_Hist_ProcType[ihist])):
                                if list_Hist_ProcType[ihist][ibin] > list_Hist_ProcType_up[ihist][ibin]:
                                    list_Hist_ProcType_up[ihist][ibin] = list_Hist_ProcType[ihist][ibin]
                                    list_Unc_ProcType_up[ihist][ibin] = list_Unc_ProcType[ihist][ibin]
                                if list_Hist_ProcType[ihist][ibin] < list_Hist_ProcType_down[ihist][ibin]:
                                    list_Hist_ProcType_down[ihist][ibin] = list_Hist_ProcType[ihist][ibin]
                                    list_Unc_ProcType_down[ihist][ibin] = list_Unc_ProcType[ihist][ibin]
                                    
                
                #----------------------------------------------------------------------------------
                # Identify which ProcTypes have XS uncertainties
                ProcType_has_XS_unc = [False]*len(datasets)
                for iProcType in range(len(datasets)):  # loop in the proc_type_lists
                    for jSource in self.XS_syst_list:
                        universe_hists = []
                        for iUniverse in range(2):
                            Hist_ProcType = np.zeros(len(bins)-1)
                            for iProcDic in range(len(datasets[iProcType])):  # loop in the proc_dictionaries inside the lists
                                Hist_name = var+"_"+str(region)+"_"+str(systematics[iSource][0])+"_"+str(iUniverse)
                                if Hist_name in datasets[iProcType][iProcDic]:
                                    proc_dic = datasets[iProcType][iProcDic][Hist_name]
                                else:
                                    proc_dic = datasets[iProcType][iProcDic][var+"_"+str(region)+"_0_0"]
                                Hist_raw = proc_dic["Hist"]
                                Unc_raw = proc_dic["Unc"]
                                Start_raw = proc_dic["Start"]
                                End_raw = proc_dic["End"]
                                Nbins_raw = proc_dic["Nbins"]
                                Delta_raw = (End_raw - Start_raw)/Nbins_raw
                        
                                Hist_new = [0]*(len(bins)-1)
                                Unc_new = [0]*(len(bins)-1)
                                        
                                for iRawBin in range(Nbins_raw):
                                    inf_raw = Start_raw + iRawBin*Delta_raw
                                    sup_raw = Start_raw + (iRawBin+1)*Delta_raw
                                    for iNewBin in range(len(Hist_new)):
                                        if( (inf_raw >= bins[iNewBin]) and (sup_raw <= bins[iNewBin+1]) ):
                                            Hist_new[iNewBin] = Hist_new[iNewBin] + Hist_raw[iRawBin]
                                            Unc_new[iNewBin] = np.sqrt( Unc_new[iNewBin]**2 + Unc_raw[iRawBin]**2 )
                        
                                for iNewBin in range(len(Hist_new)):
                                    if Hist_new[iNewBin] < 0:
                                        Hist_new[iNewBin] = 0
                                        Unc_new[iNewBin] = 0
                        
                                Hist_ProcType = Hist_ProcType + np.array(Hist_new)
                            
                            if Hist_ProcType.sum() == 0:
                                Hist_ProcType[0] = 0.0001
                            if (signal_cross_section is not None) and (iProcType == 0):
                                Hist_ProcType = Hist_ProcType/signal_cross_section
                            if processes_scale_factors is not None:
                                for ProcSF in processes_scale_factors:
                                    if iProcType == ProcSF[0]:
                                        Hist_ProcType = Hist_ProcType*ProcSF[1]
                                        break
                            
                            universe_hists.append(Hist_ProcType)
                            
                        universe_differences = np.array(universe_hists[1]) - np.array(universe_hists[0])  
                        if np.mean(universe_differences) != 0:
                            ProcType_has_XS_unc[iProcType] = True
                #print(ProcType_has_XS_unc)
            
        
                
                #----------------------------------------------------------------------------------
                list_Norm_ProcType_down = []
                list_Norm_ProcType_up = []
                for iProcType in range(len(datasets)):
                    if (ProcType_has_XS_unc[iProcType] or ProcType_has_norm_param[iProcType]) and (iSource == "Scales"):
                        sf_up = list_ProcType_sum[iProcType]/list_ProcType_norm_up_sum[iProcType]
                        sf_down = list_ProcType_sum[iProcType]/list_ProcType_norm_down_sum[iProcType]
                        #list_Hist_ProcType_down[iProcType] = list_Hist_ProcType_down[iProcType]*sf_down
                        #list_Hist_ProcType_up[iProcType] = list_Hist_ProcType_up[iProcType]*sf_up
                        list_Norm_ProcType_down.append(sf_down)
                        list_Norm_ProcType_up.append(sf_up)
                    else:
                        list_Norm_ProcType_down.append(1.)
                        list_Norm_ProcType_up.append(1.)
            
                self.hist_table3D[systematics[iSource][0]] = [list_Hist_ProcType_down, list_Hist_ProcType_up]
                self.unc_table3D[systematics[iSource][0]] = [list_Unc_ProcType_down, list_Unc_ProcType_up]
                self.norm_table3D[systematics[iSource][0]] = [list_Norm_ProcType_down, list_Norm_ProcType_up]

                self.sys_IDs.append(systematics[iSource][0])
                self.sys_labels.append(iSource)       
    

            #--------------------------------------------------------------------------------
            elif( iSource == "PDF" ):
                
                list_Hist_ProcType_down = []
                list_Unc_ProcType_down = []
                list_Hist_ProcType_up = []
                list_Unc_ProcType_up = []
                list_ProcType_norm_up_sum = []
                list_ProcType_norm_down_sum = []
                for iProcType in range(len(datasets)):  # loop in the proc_type_lists

                    Hist_ProcType_up = np.zeros(len(bins)-1)
                    Hist_ProcType_down = np.zeros(len(bins)-1)
                    Unc_ProcType_up = np.zeros(len(bins)-1)  # Stat. Uncertainty of Hist
                    Unc_ProcType_down = np.zeros(len(bins)-1)  # Stat. Uncertainty of Hist
                    for iProcDic in range(len(datasets[iProcType])):  # loop in the proc_dictionaries inside the lists

                        list_Hist_PDFuniverse = []

                        # Find number of pdf universes
                        keys_diclist = datasets[iProcType][iProcDic].keys()
                        num_dislist = [int(dicname.split("_")[-1]) for dicname in keys_diclist]
                        NPDFuniverses = max(num_dislist)+1

                        # Find the PDF type
                        hist_proc_dic_type = np.array(datasets[iProcType][iProcDic][var+"_"+str(region)+"_"+str(systematics[iSource][0])+"_"+str(NPDFuniverses-1)]["Hist"])
                        hist_proc_dic_cv = np.array(datasets[iProcType][iProcDic][var+"_"+str(region)+"_0_0"]["Hist"])
                        has_positive = False
                        for ipbin in range(len(hist_proc_dic_cv)):
                            if hist_proc_dic_cv[ipbin] > 0:
                                ID_PDF_TYPE = round(hist_proc_dic_type[ipbin]/hist_proc_dic_cv[ipbin])
                                has_positive = True
                                break
                        if has_positive:
                            if ID_PDF_TYPE == 2:
                                PDF_TYPE = "mc"
                            elif ID_PDF_TYPE == 1:
                                PDF_TYPE = "hessian"
                            else:
                                PDF_TYPE = "none"
                        else:
                            PDF_TYPE = "hessian" # dumb treatment of zeros

                        # Check if there is an alphaS uncertainty:
                        n_offset = 0
                        if( NPDFuniverses-1 == 33 or NPDFuniverses-1 == 103 ):
                               n_offset = 2

                        for iPDFuniverse in range(NPDFuniverses):  # loop in the PDF universes
                            proc_dic = datasets[iProcType][iProcDic][var+"_"+str(region)+"_"+str(systematics[iSource][0])+"_"+str(iPDFuniverse)]
                            Hist_raw = proc_dic["Hist"]
                            Unc_raw = proc_dic["Unc"]
                            Start_raw = proc_dic["Start"]
                            End_raw = proc_dic["End"]
                            Nbins_raw = proc_dic["Nbins"]
                            Delta_raw = (End_raw - Start_raw)/Nbins_raw

                            Hist_new = [0]*(len(bins)-1)
                            Unc_new = [0]*(len(bins)-1)

                            for iRawBin in range(Nbins_raw):
                                inf_raw = Start_raw + iRawBin*Delta_raw
                                sup_raw = Start_raw + (iRawBin+1)*Delta_raw
                                for iNewBin in range(len(Hist_new)):
                                    if( (inf_raw >= bins[iNewBin]) and (sup_raw <= bins[iNewBin+1]) ):
                                        Hist_new[iNewBin] = Hist_new[iNewBin] + Hist_raw[iRawBin]
                                        Unc_new[iNewBin] = np.sqrt( Unc_new[iNewBin]**2 + Unc_raw[iRawBin]**2 )

                            for iNewBin in range(len(Hist_new)):
                                if Hist_new[iNewBin] < 0:
                                    Hist_new[iNewBin] = 0
                                    Unc_new[iNewBin] = 0

                            if iPDFuniverse == 0:
                                Hist_0 = Hist_new
                            elif iPDFuniverse == NPDFuniverses-1:
                                Hist_cv = Hist_new
                            else:
                                list_Hist_PDFuniverse.append(Hist_new)

                        #print(iProcType, PDF_TYPE, ID_PDF_TYPE)
                        if PDF_TYPE == "hessian":
                            PDFunc = np.zeros_like(Hist_cv)
                            for iPDFuniverse in range(NPDFuniverses-2-n_offset):
                                PDFunc = PDFunc + (np.array(list_Hist_PDFuniverse[iPDFuniverse]) - np.array(Hist_0))**2
                            PDFunc = np.sqrt(PDFunc)
                        elif PDF_TYPE == "mc":
                            Hist_cv = np.array(Hist_cv)/2
                            PDFunc = np.zeros_like(Hist_cv)
                            for ibin in range(len(Hist_cv)):
                                list_values_ibin = []
                                for iPDFuniverse in range(NPDFuniverses-2-n_offset):
                                    list_values_ibin.append(list_Hist_PDFuniverse[iPDFuniverse][ibin])
                                list_values_ibin.sort()
                                NPDFvar = len(list_values_ibin)
                                PDFunc[ibin] = (list_values_ibin[int(round(0.841344746*NPDFvar))] - list_values_ibin[int(round(0.158655254*NPDFvar))])/2.
                        else:
                            Hist_cv = np.array(Hist_cv)/3
                            PDFunc = np.zeros_like(Hist_cv)

                        if n_offset > 0:
                            AlphaSunc = np.abs(np.array(list_Hist_PDFuniverse[-1])-np.array(list_Hist_PDFuniverse[-2]))/2.
                            PDFunc = np.sqrt(PDFunc**2 + AlphaSunc**2)

                        Hist_new_up = np.array(Hist_cv)+PDFunc
                        Hist_new_down = np.array(Hist_cv)-PDFunc

                        for iNewBin in range(len(Hist_cv)):
                            if Hist_new_down[iNewBin] < 0:
                                Hist_new_down[iNewBin] = 0

                        Hist_ProcType_up = Hist_ProcType_up + Hist_new_up
                        Hist_ProcType_down = Hist_ProcType_down + Hist_new_down
                        Unc_ProcType_up = np.sqrt(Unc_ProcType**2 + np.array(Unc_new)**2) # Dumb [not used]
                        Unc_ProcType_down = np.sqrt(Unc_ProcType**2 + np.array(Unc_new)**2) # Dumb [not used]

                    if Hist_ProcType_up.sum() == 0:
                        Hist_ProcType_up[0] = 0.0001
                        Unc_ProcType_up[0] = 0.0001
                    if Hist_ProcType_down.sum() == 0:
                        Hist_ProcType_down[0] = 0.0001
                        Unc_ProcType_down[0] = 0.0001

                    if (signal_cross_section is not None) and (iProcType == 0):
                        Hist_ProcType_up = Hist_ProcType_up/signal_cross_section
                        Unc_ProcType_up = Unc_ProcType_up/signal_cross_section
                        Hist_ProcType_down = Hist_ProcType_down/signal_cross_section
                        Unc_ProcType_down = Unc_ProcType_down/signal_cross_section
                    if processes_scale_factors is not None:
                        for ProcSF in processes_scale_factors:
                            if iProcType == ProcSF[0]:
                                Hist_ProcType_up = Hist_ProcType_up*ProcSF[1]
                                Unc_ProcType_up = Unc_ProcType_up*ProcSF[1]
                                Hist_ProcType_down = Hist_ProcType_down*ProcSF[1]
                                Unc_ProcType_down = Unc_ProcType_down*ProcSF[1]
                                break

                    list_Hist_ProcType_down.append(Hist_ProcType_down)
                    list_Unc_ProcType_down.append(Unc_ProcType_down)
                    list_Hist_ProcType_up.append(Hist_ProcType_up)
                    list_Unc_ProcType_up.append(Unc_ProcType_up)

                    # Get the normalization factors from the "SystVar" histograms
                    list_ProcType_norm_up_sum.append(Hist_ProcType_up.sum())
                    list_ProcType_norm_down_sum.append(Hist_ProcType_down.sum())


                #----------------------------------------------------------------------------------
                list_Norm_ProcType_down = []
                list_Norm_ProcType_up = []
                for iProcType in range(len(datasets)):
                    if ProcType_has_norm_param[iProcType]:
                        sf_up = list_ProcType_sum[iProcType]/list_ProcType_norm_up_sum[iProcType]
                        sf_down = list_ProcType_sum[iProcType]/list_ProcType_norm_down_sum[iProcType]
                        #list_Hist_ProcType_down[iProcType] = list_Hist_ProcType_down[iProcType]*sf_down
                        #list_Hist_ProcType_up[iProcType] = list_Hist_ProcType_up[iProcType]*sf_up
                        list_Norm_ProcType_down.append(sf_down)
                        list_Norm_ProcType_up.append(sf_up)
                    else:
                        list_Norm_ProcType_down.append(1.)
                        list_Norm_ProcType_up.append(1.)


                self.hist_table3D[systematics[iSource][0]] = [list_Hist_ProcType_down, list_Hist_ProcType_up]
                self.unc_table3D[systematics[iSource][0]] = [list_Unc_ProcType_down, list_Unc_ProcType_up]
                self.norm_table3D[systematics[iSource][0]] = [list_Norm_ProcType_down, list_Norm_ProcType_up]

                self.sys_IDs.append(systematics[iSource][0])
                self.sys_labels.append(iSource)



            #--------------------------------------------------------------------------------
            elif( (iSource == "TopPt") or (iSource == "WPt") ):

                list_Hist_ProcType_up = []
                list_Unc_ProcType_up = []
                list_Hist_ProcType_down = []
                list_Unc_ProcType_down = []
                #print(systematics[iSource][1])

                # Only one universe
                iUniverse = 0

                list_Hist_ProcType = []
                list_Unc_ProcType = []
                for iProcType in range(len(datasets)):  # loop in the proc_type_lists

                    Hist_ProcType = np.zeros(len(bins)-1)
                    Unc_ProcType = np.zeros(len(bins)-1)  # Stat. Uncertainty of Hist
                    for iProcDic in range(len(datasets[iProcType])):  # loop in the proc_dictionaries inside the lists
                        Hist_name = var+"_"+str(region)+"_"+str(systematics[iSource][0])+"_"+str(iUniverse)
                        if Hist_name in datasets[iProcType][iProcDic]:
                            proc_dic = datasets[iProcType][iProcDic][Hist_name]
                        else:
                            proc_dic = datasets[iProcType][iProcDic][var+"_"+str(region)+"_0_0"]
                        Hist_raw = proc_dic["Hist"]
                        Unc_raw = proc_dic["Unc"]
                        Start_raw = proc_dic["Start"]
                        End_raw = proc_dic["End"]
                        Nbins_raw = proc_dic["Nbins"]
                        Delta_raw = (End_raw - Start_raw)/Nbins_raw

                        Hist_new = [0]*(len(bins)-1)
                        Unc_new = [0]*(len(bins)-1)

                        for iRawBin in range(Nbins_raw):
                            inf_raw = Start_raw + iRawBin*Delta_raw
                            sup_raw = Start_raw + (iRawBin+1)*Delta_raw
                            for iNewBin in range(len(Hist_new)):
                                if( (inf_raw >= bins[iNewBin]) and (sup_raw <= bins[iNewBin+1]) ):
                                    Hist_new[iNewBin] = Hist_new[iNewBin] + Hist_raw[iRawBin]
                                    Unc_new[iNewBin] = np.sqrt( Unc_new[iNewBin]**2 + Unc_raw[iRawBin]**2 )

                        for iNewBin in range(len(Hist_new)):
                            if Hist_new[iNewBin] < 0:
                                Hist_new[iNewBin] = 0
                                Unc_new[iNewBin] = 0

                        Hist_ProcType = Hist_ProcType + np.array(Hist_new)
                        Unc_ProcType = np.sqrt(Unc_ProcType**2 + np.array(Unc_new)**2)

                    if Hist_ProcType.sum() == 0:
                        Hist_ProcType[0] = 0.0001
                        Unc_ProcType[0] = 0.0001
                    if (signal_cross_section is not None) and (iProcType == 0):
                        Hist_ProcType = Hist_ProcType/signal_cross_section
                        Unc_ProcType = Unc_ProcType/signal_cross_section
                    if processes_scale_factors is not None:
                        for ProcSF in processes_scale_factors:
                            if iProcType == ProcSF[0]:
                                Hist_ProcType = Hist_ProcType*ProcSF[1]
                                Unc_ProcType = Unc_ProcType*ProcSF[1]
                                break

                    list_Hist_ProcType.append(Hist_ProcType)
                    list_Unc_ProcType.append(Unc_ProcType)

                list_Hist_ProcType_up = copy.deepcopy(list_Hist_ProcType)

                list_Hist_ProcType_down = copy.deepcopy(list_Hist_ProcType)
                for iProcType in range(len(datasets)):  # loop in the proc_type_lists
                    for ibin in range(len(list_Hist_ProcType_down[iProcType])):
                        diff = list_Hist_ProcType_down[iProcType][ibin] - self.hist_table3D[0][0][iProcType][ibin]
                        list_Hist_ProcType_down[iProcType][ibin] = list_Hist_ProcType_down[iProcType][ibin] - 2*diff

                list_Unc_ProcType_up = copy.deepcopy(list_Unc_ProcType)
                list_Unc_ProcType_down = copy.deepcopy(list_Unc_ProcType)

                self.hist_table3D[systematics[iSource][0]] = [list_Hist_ProcType_down, list_Hist_ProcType_up]
                self.unc_table3D[systematics[iSource][0]] = [list_Unc_ProcType_down, list_Unc_ProcType_up]

                self.sys_IDs.append(systematics[iSource][0])
                self.sys_labels.append(iSource)


        #--------------------------------------------------------------------------------
        #print("WZ_nominal_sum", self.hist_table3D[0][0][2].sum())  
        #print("WZ_pdf_up_sum", self.hist_table3D[12][1][2].sum())  
        #print("WZ_pdf_down_sum", self.hist_table3D[12][0][2].sum())

        self.region = region
        self.var = var
        self.period = period
        self.bins = bins
        self.number_ds_groups = len(datasets)
        self.labels = labels
        self.colors = colors
        self.systematics = systematics
        self.ID_max = ID_max
        self.N_sources = (ID_max+1) + 2  # including lumi and stat sources
        if processes_scale_factors is not None:
            self.N_sources += 1  # including ProcSF
        self.symmetric_factor = symmetric_factor
        self.smooth_repeat = smooth_repeat
        self.set_smooth_factor(smooth_factor)
        self.has_data = False

        self.signal_name = signal_label 
        self.analysis_name = analysis_name
        #self.obs_features_list = []
        #self.obs_bins_list = []
        

    #==============================================================================================================
    def set_smooth_factor(self, smooth_factor):

        # Get smooth bins
        # Initialize bins (first index is source, and second index is process)
        
        self.smooth_toright_bins_list_down = []
        self.smooth_toright_bins_list_up = []
        self.smooth_toleft_bins_list_down = []
        self.smooth_toleft_bins_list_up = []
        for iSource in range(self.N_sources):
            self.smooth_toright_bins_list_down.append([self.bins]*self.number_ds_groups)
            self.smooth_toright_bins_list_up.append([self.bins]*self.number_ds_groups)
            self.smooth_toleft_bins_list_down.append([self.bins]*self.number_ds_groups)
            self.smooth_toleft_bins_list_up.append([self.bins]*self.number_ds_groups)
            
        #self.create_tables3D()
        #self.create_bkg_tables2D()
        #self.create_signal_tables2D()
        #self.frac_unc_tables3D()
        
        for iSource in range(self.N_sources):
            if self.hist_table3D[iSource][0] != 0:
                if( (iSource > 0) and (iSource < self.ID_max+1) ): # Systematics with smoothing
                    for iUniverse in range(2):
                        for iProcess in range(self.number_ds_groups):
                        
                            smooth_toright_bins = [self.bins[0]] # set first limit for the smooth bins (equal to bins)
                            smooth_toleft_bins = [self.bins[-1]] # set first limit for the smooth bins (equal to bins)
                        
                        
                            max_count = np.amax(self.hist_table3D[0][0][iProcess])
                            group_sys = self.hist_table3D[iSource][iUniverse][iProcess][0]
                            group_cv = self.hist_table3D[0][0][iProcess][0]
                            if group_cv > 0 and group_sys > 0:
                                frac_sys_unc = np.abs(group_sys - group_cv)/group_cv
                            else:
                                frac_sys_unc = 0
                            stat_unc = self.unc_table3D[0][0][iProcess][0]
                            if group_cv > 0:
                                frac_stat_unc = stat_unc/group_cv
                            else:
                                frac_stat_unc = 0
                            max_count_group = self.hist_table3D[0][0][iProcess][0]
                            frac_count = max_count_group/max_count
                            #smooth_cut = ((1-smooth_factor)*frac_count + smooth_factor)*frac_sys_unc
                            smooth_cut = smooth_factor*frac_sys_unc
                            for iBin in range(len(self.bins)-1):
                                if( (iBin < len(self.bins)-2) and (frac_stat_unc > smooth_cut) ):
                                    group_sys += self.hist_table3D[iSource][iUniverse][iProcess][iBin + 1]
                                    group_cv += self.hist_table3D[0][0][iProcess][iBin + 1]
                                    if group_cv > 0 and group_sys > 0:
                                        frac_sys_unc = np.abs(group_sys - group_cv)/group_cv
                                    else:
                                        frac_sys_unc = 0
                                    stat_unc = np.sqrt(stat_unc**2 + self.unc_table3D[0][0][iProcess][iBin + 1]**2)
                                    if group_cv > 0:
                                        frac_stat_unc = stat_unc/group_cv
                                    else:
                                        frac_stat_unc = 0
                                    max_count_group = max([max_count_group, self.hist_table3D[0][0][iProcess][iBin + 1]])
                                    frac_count = max_count_group/max_count
                                    #smooth_cut = ((1-smooth_factor)*frac_count + smooth_factor)*frac_sys_unc
                                    smooth_cut = smooth_factor*frac_sys_unc
                                elif( (iBin < len(self.bins)-2) and (frac_stat_unc <= smooth_cut) ):
                                    smooth_toright_bins.append(self.bins[iBin + 1])
                                    group_sys = self.hist_table3D[iSource][iUniverse][iProcess][iBin + 1]
                                    group_cv = self.hist_table3D[0][0][iProcess][iBin + 1]
                                    if group_cv > 0 and group_sys > 0:
                                        frac_sys_unc = np.abs(group_sys - group_cv)/group_cv
                                    else:
                                        frac_sys_unc = 0
                                    stat_unc = self.unc_table3D[0][0][iProcess][iBin + 1]
                                    if group_cv > 0:
                                        frac_stat_unc = stat_unc/group_cv
                                    else:
                                        frac_stat_unc = 0
                                    max_count_group = self.hist_table3D[0][0][iProcess][iBin + 1]
                                    frac_count = max_count_group/max_count
                                    #smooth_cut = ((1-smooth_factor)*frac_count + smooth_factor)*frac_sys_unc
                                    smooth_cut = smooth_factor*frac_sys_unc
                                elif( (iBin == len(self.bins)-2) and (frac_stat_unc > smooth_cut) and (len(smooth_toright_bins) > 1) ):
                                    smooth_toright_bins[-1] = self.bins[-1]
                                else:
                                    smooth_toright_bins.append(self.bins[-1])
                                    
                                    
                            
                            max_count = np.amax(self.hist_table3D[0][0][iProcess])
                            group_sys = self.hist_table3D[iSource][iUniverse][iProcess][len(self.bins)-2]
                            group_cv = self.hist_table3D[0][0][iProcess][len(self.bins)-2]
                            if group_cv > 0 and group_sys > 0:
                                frac_sys_unc = np.abs(group_sys - group_cv)/group_cv
                            else:
                                frac_sys_unc = 0
                            stat_unc = self.unc_table3D[0][0][iProcess][len(self.bins)-2]
                            if group_cv > 0:
                                frac_stat_unc = stat_unc/group_cv
                            else:
                                frac_stat_unc = 0
                            max_count_group = self.hist_table3D[0][0][iProcess][len(self.bins)-2]
                            frac_count = max_count_group/max_count
                            #smooth_cut = ((1-smooth_factor)*frac_count + smooth_factor)*frac_sys_unc
                            smooth_cut = smooth_factor*frac_sys_unc
                            #print("smooth_factor", smooth_factor)
                            #print("frac_sys_unc", frac_sys_unc)
                            #print("smooth_cut", smooth_cut)
                            #print("frac_stat_unc", frac_stat_unc)
                            for iBin in reversed(range(len(self.bins)-1)):
                                if( (iBin > 0) and (frac_stat_unc > smooth_cut) ):
                                    group_sys += self.hist_table3D[iSource][iUniverse][iProcess][iBin]
                                    group_cv += self.hist_table3D[0][0][iProcess][iBin]
                                    if group_cv > 0 and group_sys > 0:
                                        frac_sys_unc = np.abs(group_sys - group_cv)/group_cv
                                    else:
                                        frac_sys_unc = 0
                                    stat_unc = np.sqrt(stat_unc**2 + self.unc_table3D[0][0][iProcess][iBin]**2)
                                    if group_cv > 0:
                                        frac_stat_unc = stat_unc/group_cv
                                    else:
                                        frac_stat_unc = 0
                                    max_count_group = max([max_count_group, self.hist_table3D[0][0][iProcess][iBin]])
                                    frac_count = max_count_group/max_count
                                    #smooth_cut = ((1-smooth_factor)*frac_count + smooth_factor)*frac_sys_unc
                                    smooth_cut = smooth_factor*frac_sys_unc
                                elif( (iBin > 0) and (frac_stat_unc <= smooth_cut) ):
                                    smooth_toleft_bins.append(self.bins[iBin])
                                    group_sys = self.hist_table3D[iSource][iUniverse][iProcess][iBin]
                                    group_cv = self.hist_table3D[0][0][iProcess][iBin]
                                    if group_cv > 0 and group_sys > 0:
                                        frac_sys_unc = np.abs(group_sys - group_cv)/group_cv
                                    else:
                                        frac_sys_unc = 0
                                    stat_unc = self.unc_table3D[0][0][iProcess][iBin]
                                    if group_cv > 0:
                                        frac_stat_unc = stat_unc/group_cv
                                    else:
                                        frac_stat_unc = 0
                                    max_count_group = self.hist_table3D[0][0][iProcess][iBin]
                                    frac_count = max_count_group/max_count
                                    #smooth_cut = ((1-smooth_factor)*frac_count + smooth_factor)*frac_sys_unc
                                    smooth_cut = smooth_factor*frac_sys_unc
                                elif( (iBin == 0) and (frac_stat_unc > smooth_cut) and (len(smooth_toleft_bins) > 1) ):
                                    smooth_toleft_bins[-1] = self.bins[0]
                                else:
                                    smooth_toleft_bins.append(self.bins[0])
                        
                                        
                            smooth_toleft_bins.reverse()
                            if iUniverse == 0:
                                self.smooth_toright_bins_list_down[iSource][iProcess] = smooth_toright_bins 
                                self.smooth_toleft_bins_list_down[iSource][iProcess] = smooth_toleft_bins
                            elif iUniverse == 1:
                                self.smooth_toright_bins_list_up[iSource][iProcess] = smooth_toright_bins  
                                self.smooth_toleft_bins_list_up[iSource][iProcess] = smooth_toleft_bins
                            

                            # Apply the normalization factor to the histograms from Scales, PDF, etc
                            if self.norm_table3D[iSource][0] != 0:
                                self.hist_table3D[iSource][iUniverse][iProcess] = self.hist_table3D[iSource][iUniverse][iProcess]*self.norm_table3D[iSource][iUniverse][iProcess]


        
        self.create_tables3D()
        self.create_bkg_tables2D()
        self.create_signal_tables2D()
        self.frac_unc_tables3D()
        
    #==============================================================================================================
    def show_smooth_bins(self):
        ds_list = []
        for iSource in range(self.N_sources):
            if self.hist_table3D[iSource][0] != 0:
                if( (iSource > 0) and (iSource < self.ID_max+1) ): # Systematics with smoothing
                    for Source in self.systematics.keys():
                        if iSource == self.systematics[Source][0]:
                            for iProcess in range(self.number_ds_groups):
                                ds_list.append({ "Systematic": Source,  "Universe": "Down_toright",  "Datasets": self.processes[iProcess], "Smooth bins": self.smooth_toright_bins_list_down[iSource][iProcess] })
                                ds_list.append({ "Systematic": Source,  "Universe": "Down_toleft",  "Datasets": self.processes[iProcess], "Smooth bins": self.smooth_toleft_bins_list_down[iSource][iProcess] })
                                
                                ds_list.append({ "Systematic": Source,  "Universe": "Up_toright",  "Datasets": self.processes[iProcess], "Smooth bins": self.smooth_toright_bins_list_up[iSource][iProcess] })
                                ds_list.append({ "Systematic": Source,  "Universe": "Up_toleft",  "Datasets": self.processes[iProcess], "Smooth bins": self.smooth_toleft_bins_list_up[iSource][iProcess] })
        
        ds_list = pd.DataFrame(ds_list)
        print(ds_list)
        
                            
    #==============================================================================================================
    def create_tables3D(self):

        #Initialize systematic table3D (first index is source, second index is universe, and third index is process)
        self.sys_unc_table3D = []    # Use in Combine datacards
        for i in range(self.N_sources):
            self.sys_unc_table3D.append([[], []])
    
        for iSource in range(self.N_sources):
            if iSource > 0: 
                for iUniverse in range(2): # The name universe is used here but it only represents the up and down variations after universes combination
                    for iProcess in range(self.number_ds_groups):
                        self.sys_unc_table3D[iSource][iUniverse].append(np.zeros(len(self.bins)-1))
                
                
        for iSource in range(self.N_sources):
            if self.hist_table3D[iSource][0] != 0:
                if( (iSource > 0) and (iSource < self.ID_max+1) ): # Systematics with smoothing
                    for iUniverse in range(2):
                        for iProcess in range(self.number_ds_groups):  
                            
                            
                            if iUniverse == 0:
                                smooth_toright_bins = self.smooth_toright_bins_list_down[iSource][iProcess]
                            elif iUniverse == 1:
                                smooth_toright_bins = self.smooth_toright_bins_list_up[iSource][iProcess]
                                
                            sys_unc_table3D_toright = np.copy(self.sys_unc_table3D[iSource][iUniverse][iProcess])

                            for isBin in range(len(smooth_toright_bins)-1):
                                combined_sys = 0
                                combined_cv = 0
                                for iBin in range(len(self.bins)-1):
                                    if( (self.bins[iBin] >= smooth_toright_bins[isBin]) and (self.bins[iBin+1] <= smooth_toright_bins[isBin+1]) ):
                                        combined_sys += self.hist_table3D[iSource][iUniverse][iProcess][iBin]
                                        combined_cv += self.hist_table3D[0][0][iProcess][iBin]
                                        
                                if combined_cv > 0:
                                    frac_comb_variation = (combined_sys - combined_cv)/combined_cv
                                else:
                                    frac_comb_variation = 0
                                
                                for iBin in range(len(self.bins)-1):
                                    if( (self.bins[iBin] >= smooth_toright_bins[isBin]) and (self.bins[iBin+1] <= smooth_toright_bins[isBin+1]) ):
                                        sys_unc_table3D_toright[iBin] = self.hist_table3D[0][0][iProcess][iBin]*frac_comb_variation
                                        
                                        
                            if iUniverse == 0:
                                smooth_toleft_bins = self.smooth_toleft_bins_list_down[iSource][iProcess]
                            elif iUniverse == 1:
                                smooth_toleft_bins = self.smooth_toleft_bins_list_up[iSource][iProcess]
                                
                            sys_unc_table3D_toleft = np.copy(self.sys_unc_table3D[iSource][iUniverse][iProcess])
                            for isBin in range(len(smooth_toleft_bins)-1):
                                combined_sys = 0
                                combined_cv = 0
                                for iBin in range(len(self.bins)-1):
                                    if( (self.bins[iBin] >= smooth_toleft_bins[isBin]) and (self.bins[iBin+1] <= smooth_toleft_bins[isBin+1]) ):
                                        combined_sys += self.hist_table3D[iSource][iUniverse][iProcess][iBin]
                                        combined_cv += self.hist_table3D[0][0][iProcess][iBin]
                                        
                                if combined_cv > 0:
                                    frac_comb_variation = (combined_sys - combined_cv)/combined_cv
                                else:
                                    frac_comb_variation = 0
                                    
                                for iBin in range(len(self.bins)-1):
                                    if( (self.bins[iBin] >= smooth_toleft_bins[isBin]) and (self.bins[iBin+1] <= smooth_toleft_bins[isBin+1]) ):
                                        sys_unc_table3D_toleft[iBin] = self.hist_table3D[0][0][iProcess][iBin]*frac_comb_variation
                                        
                                        
                            for iBin in range(len(self.bins)-1):           
                                if sys_unc_table3D_toright[iBin]*sys_unc_table3D_toleft[iBin] > 0:
                                    if sys_unc_table3D_toright[iBin] > 0:
                                        self.sys_unc_table3D[iSource][iUniverse][iProcess][iBin] = np.maximum(sys_unc_table3D_toright[iBin], sys_unc_table3D_toleft[iBin]) 
                                    else:
                                        self.sys_unc_table3D[iSource][iUniverse][iProcess][iBin] = np.minimum(sys_unc_table3D_toright[iBin], sys_unc_table3D_toleft[iBin])
                                else:
                                    self.sys_unc_table3D[iSource][iUniverse][iProcess][iBin] = (sys_unc_table3D_toright[iBin] + sys_unc_table3D_toleft[iBin])/2
                                
                                
                            
                            # Smearing
                            """
                            sys_frac_unc_table3D_now = np.zeros_like(self.hist_table3D[0][0][iProcess])
                            sys_frac_unc_table3D_temp = np.zeros_like(self.hist_table3D[0][0][iProcess])
                            for iBin in range(len(self.bins)-1):
                                if self.hist_table3D[0][0][iProcess][iBin] > 0:
                                    sys_frac_unc_table3D_now[iBin] = self.sys_unc_table3D[iSource][iUniverse][iProcess][iBin]/self.hist_table3D[0][0][iProcess][iBin]
                                    sys_frac_unc_table3D_temp[iBin] = self.sys_unc_table3D[iSource][iUniverse][iProcess][iBin]/self.hist_table3D[0][0][iProcess][iBin]
                                
                            if self.smear_factor > 0 and len(self.bins)-1 >= 3:
                                for iBin in range(len(self.bins)-2):
                                    if iBin > 0:
                                        if self.hist_table3D[0][0][iProcess][iBin] > 0:
                                            #if self.unc_table3D[0][0][iProcess][iBin]/self.hist_table3D[0][0][iProcess][iBin] > 0.05:
                                            #if iBin == 0:
                                            #    sys_frac_unc_table3D_temp[iBin] = (sys_frac_unc_table3D_now[iBin] + self.smear_factor*sys_frac_unc_table3D_now[iBin+1])/(1 + self.smear_factor)
                                            #elif iBin == len(self.bins)-2:
                                            #    sys_frac_unc_table3D_temp[iBin] = (sys_frac_unc_table3D_now[iBin] + self.smear_factor*sys_frac_unc_table3D_now[iBin-1])/(1 + self.smear_factor)
                                            #else:
                                            sys_frac_unc_table3D_temp[iBin] = (sys_frac_unc_table3D_now[iBin] + self.smear_factor*(sys_frac_unc_table3D_now[iBin+1] + sys_frac_unc_table3D_now[iBin-1]))/(1 + 2*self.smear_factor)           
                                self.sys_unc_table3D[iSource][iUniverse][iProcess] = sys_frac_unc_table3D_temp*self.hist_table3D[0][0][iProcess]          
                            """
                            
                            #---------------------------------------------------------------------------------
                            # 353QH
                            # void  TH1::SmoothArray(Int_t nn, Double_t *xx, Int_t ntimes)
                            for ipass in range(self.smooth_repeat):
                                if len(self.bins)-1 >= 3:
                                    nn = len(self.bins)-1
                                    
                                    xx = np.zeros_like(self.hist_table3D[0][0][iProcess])
                                    for iBin in range(len(self.bins)-1):
                                        if self.hist_table3D[0][0][iProcess][iBin] > 0:
                                            xx[iBin] = self.sys_unc_table3D[iSource][iUniverse][iProcess][iBin]/self.hist_table3D[0][0][iProcess][iBin]
                                    
                                    # first copy original data into temp array -> copied xx to zz
                                    zz = xx.copy()
                                
                                
                                    for noent in range(2):  # run algorithm two times
                                
                                        #  do 353 i.e. running median 3, 5, and 3 in a single loop
                                        for kk in range(3):
                                            yy = zz.copy()
                                            
                                            if kk != 1:
                                                medianType = 3
                                                ifirst = 1
                                                ilast = nn-1
                                            else:
                                                medianType = 5
                                                ifirst = 2
                                                ilast = nn-2
                                    
                                            # do all elements beside the first and last point for median 3
                                            #  and first two and last 2 for median 5
                                            hh = np.zeros(medianType)
                                            for ii in range(ifirst,ilast,1):
                                                for jj in range(0,medianType,1):
                                                    hh[jj] = yy[ii - ifirst + jj]
                                                zz[ii] = np.median(hh)
                                            
                                
                                            if kk == 0:   # first median 3
                                                # first point
                                                hh[0] = zz[1]
                                                hh[1] = zz[0]
                                                hh[2] = 3*zz[1] - 2*zz[2]
                                                zz[0] = np.median(hh)
                                                # last point
                                                hh[0] = zz[nn - 2];
                                                hh[1] = zz[nn - 1];
                                                hh[2] = 3*zz[nn - 2] - 2*zz[nn - 3];
                                                zz[nn - 1] = np.median(hh)
                                
                                            if kk == 1:   #  median 5
                                                for ii in range(0,3,1):
                                                    hh[ii] = yy[ii]
                                                zz[1] = np.median(hh)
                                                # last two points
                                                for ii in range(0,3,1):
                                                    hh[ii] = yy[nn - 3 + ii]
                                                zz[nn - 2] = np.median(hh)
                                            
                                            
                                        
                                        yy = zz.copy()  # -> copied zz to yy
                                
                                        # quadratic interpolation for flat segments
                                        for ii in range(2,nn-2,1):
                                            if (zz[ii - 1] != zz[ii]) or (zz[ii] != zz[ii + 1]): 
                                                continue
                                            hh[0] = zz[ii - 2] - zz[ii]
                                            hh[1] = zz[ii + 2] - zz[ii]
                                            if hh[0]*hh[1] <= 0: 
                                                continue
                                            jk = 1
                                            if np.abs(hh[1]) > np.abs(hh[0]): 
                                                jk = -1
                                            yy[ii] = -0.5*zz[ii - 2*jk] + zz[ii]/0.75 + zz[ii + 2*jk]/6.
                                            yy[ii + jk] = 0.5*(zz[ii + 2*jk] - zz[ii - 2*jk]) + zz[ii]
                                
                                        # running means
                                        for ii in range(1,nn-1,1): 
                                            zz[ii] = 0.25*yy[ii - 1] + 0.5*yy[ii] + 0.25*yy[ii + 1]
                                        zz[0] = yy[0]
                                        zz[nn - 1] = yy[nn - 1]
                                
                                        if noent == 0:
                                            # save computed values
                                            rr = zz.copy()  # -> copied zz to rr
                                
                                            # COMPUTE  residuals
                                            for ii in range(0,nn,1):
                                                zz[ii] = xx[ii] - zz[ii]
                                        
                                    xmin = np.amin(xx)
                                    for ii in range(0,nn,1):
                                        if xmin < 0: 
                                            xx[ii] = rr[ii] + zz[ii]
                                        else: # make smoothing defined positive - not better using 0 ?
                                            xx[ii] = max(rr[ii] + zz[ii], 0.0)
                                    
                                    self.sys_unc_table3D[iSource][iUniverse][iProcess] = xx*self.hist_table3D[0][0][iProcess]
                            #---------------------------------------------------------------------------------
                            
                            """
                            for iBin in range(len(self.bins)-1):
                                syst_unc_down = self.sys_unc_table3D[iSource][0][iProcess][iBin]
                                syst_unc_up = self.sys_unc_table3D[iSource][1][iProcess][iBin]
                                syst_unc_max = max(abs(syst_unc_down), abs(syst_unc_up))
                                stat_unc = self.unc_table3D[0][0][iProcess][iBin]
                                if self.hist_table3D[0][0][iProcess][iBin] > 0 and stat_unc > self.symmetric_factor*syst_unc_max:
                                    if syst_unc_down != 0 and syst_unc_up != 0:
                                        self.sys_unc_table3D[iSource][0][iProcess][iBin] = (syst_unc_down/abs(syst_unc_down))*syst_unc_max                                    
                                        self.sys_unc_table3D[iSource][1][iProcess][iBin] = (syst_unc_up/abs(syst_unc_up))*syst_unc_max 
                                    elif syst_unc_down == 0 and syst_unc_up != 0:
                                        self.sys_unc_table3D[iSource][0][iProcess][iBin] = -(syst_unc_up/abs(syst_unc_up))*syst_unc_max                                    
                                        self.sys_unc_table3D[iSource][1][iProcess][iBin] = (syst_unc_up/abs(syst_unc_up))*syst_unc_max
                                    elif syst_unc_down != 0 and syst_unc_up == 0:
                                        self.sys_unc_table3D[iSource][0][iProcess][iBin] = (syst_unc_down/abs(syst_unc_down))*syst_unc_max                                    
                                        self.sys_unc_table3D[iSource][1][iProcess][iBin] = -(syst_unc_down/abs(syst_unc_down))*syst_unc_max
                            """
                            
                elif( iSource >= self.ID_max+1 ):   # Lumi and Stat systematics 
                    for iUniverse in range(2):
                        for iProcess in range(self.number_ds_groups):
                            for iBin in range(len(self.bins)-1):
                                self.sys_unc_table3D[iSource][iUniverse][iProcess][iBin] = self.hist_table3D[iSource][iUniverse][iProcess][iBin] - self.hist_table3D[0][0][iProcess][iBin]
        
        
    #==============================================================================================================
    def create_bkg_tables2D(self):                
                    
        #Initialize systematic table2D (first index is source, second index is universe)
        self.hist_table2D = []                   
        self.sys_unc_table2D = []    # Use in Fractional plot
        for i in range(self.N_sources):
            self.hist_table2D.append([0, 0])
            self.sys_unc_table2D.append([np.zeros(len(self.bins)-1), np.zeros(len(self.bins)-1)])

    
        # Filling hist_table2D
        for iSource in range(self.N_sources):
            if self.hist_table3D[iSource][0] != 0:
                if iSource == 0:
                    for iProcess in range(self.number_ds_groups):
                        if iProcess == 1 :
                            self.hist_table2D[0][0] = self.hist_table3D[0][0][iProcess]
                        elif iProcess > 1 :
                            self.hist_table2D[0][0] = self.hist_table2D[0][0] + self.hist_table3D[0][0][iProcess]
                if iSource > 0: 
                    for iUniverse in range(2):
                        for iProcess in range(self.number_ds_groups):
                            if iProcess == 1 :
                                self.hist_table2D[iSource][iUniverse] = self.hist_table3D[iSource][iUniverse][iProcess]
                            elif iProcess > 1 :
                                self.hist_table2D[iSource][iUniverse] = self.hist_table2D[iSource][iUniverse] + self.hist_table3D[iSource][iUniverse][iProcess]

        for iSource in range(self.N_sources):
            if self.hist_table3D[iSource][0] != 0:
                if iSource > 0: 
                    for iUniverse in range(2):
                        for iProcess in range(self.number_ds_groups):
                            if iProcess >= 1 :
                                self.sys_unc_table2D[iSource][iUniverse] = self.sys_unc_table2D[iSource][iUniverse] + self.sys_unc_table3D[iSource][iUniverse][iProcess]
    
        self.hist_bkg = self.hist_table2D[0][0]
    
        #--------------------------------------------------------------------------------
        # Transform uncertainties in fractional uncertainties
        for iSource in range(self.N_sources):
            if self.hist_table3D[iSource][0] != 0:
                if iSource > 0: 
                    for iUniverse in range(2):
                        for iBin in range(len(self.bins)-1):
                            if self.hist_table2D[0][0][iBin] > 0:
                                self.sys_unc_table2D[iSource][iUniverse][iBin] = self.sys_unc_table2D[iSource][iUniverse][iBin]/self.hist_table2D[0][0][iBin]
                            else: 
                                self.sys_unc_table2D[iSource][iUniverse][iBin] = 0
   
        #--------------------------------------------------------------------------------
        # Get total fractional uncertainty        
        self.sys_total_unc_up = np.zeros(len(self.bins)-1)
        self.sys_total_unc_down = np.zeros(len(self.bins)-1)
            
        for iSource in range(self.N_sources):
            if self.hist_table3D[iSource][0] != 0:
                if iSource > 0: 
                    for iBin in range(len(self.bins)-1):
                        if (self.sys_unc_table2D[iSource][0][iBin] >= 0) and (self.sys_unc_table2D[iSource][1][iBin] < 0):
                            self.sys_total_unc_up[iBin] += np.power(self.sys_unc_table2D[iSource][0][iBin],2)
                            self.sys_total_unc_down[iBin] += np.power(self.sys_unc_table2D[iSource][1][iBin],2)
                        elif (self.sys_unc_table2D[iSource][0][iBin] < 0) and (self.sys_unc_table2D[iSource][1][iBin] >= 0): 
                            self.sys_total_unc_up[iBin] += np.power(self.sys_unc_table2D[iSource][1][iBin],2)
                            self.sys_total_unc_down[iBin] += np.power(self.sys_unc_table2D[iSource][0][iBin],2)
                        elif (self.sys_unc_table2D[iSource][0][iBin] >= 0) and (self.sys_unc_table2D[iSource][1][iBin] >= 0):
                            if np.abs(self.sys_unc_table2D[iSource][0][iBin]) >= np.abs(self.sys_unc_table2D[iSource][1][iBin]):
                                self.sys_total_unc_up[iBin] += np.power(self.sys_unc_table2D[iSource][0][iBin],2)
                            else:
                                self.sys_total_unc_up[iBin] += np.power(self.sys_unc_table2D[iSource][1][iBin],2)
                        elif (self.sys_unc_table2D[iSource][0][iBin] < 0) and (self.sys_unc_table2D[iSource][1][iBin] < 0):
                            if np.abs(self.sys_unc_table2D[iSource][0][iBin]) >= np.abs(self.sys_unc_table2D[iSource][1][iBin]):
                                self.sys_total_unc_down[iBin] += np.power(self.sys_unc_table2D[iSource][0][iBin],2)
                            else:
                                self.sys_total_unc_down[iBin] += np.power(self.sys_unc_table2D[iSource][1][iBin],2)
                                
        self.sys_total_unc_up = np.sqrt(self.sys_total_unc_up)
        self.sys_total_unc_down = np.sqrt(self.sys_total_unc_down)*(-1)
    
        #--------------------------------------------------------------------------------
        # Get groups systematic fractional uncertainty
        if self.groups is not None:
            self.sys_groups_list = []
            self.sys_groups_name = []
            self.sys_groups_unc_up = []
            self.sys_groups_unc_down = []
            for ig in range(len(self.groups)):
                sys_group_list = []
                sys_group_unc_up = np.zeros(len(self.bins)-1)
                sys_group_unc_down = np.zeros(len(self.bins)-1)
                if isinstance(self.groups[ig], str):
                    self.sys_groups_name.append(self.groups[ig])
                    prefix = self.groups[ig]+"_"
                    for isys in range(len(self.sys_IDs)):
                        iSource = self.sys_IDs[isys]
                        if self.hist_table3D[iSource][0] != 0:
                            if iSource > 0 and self.sys_labels[isys][:len(prefix)] == prefix: 
                                sys_group_list.append(self.sys_labels[isys])
                                for iBin in range(len(self.bins)-1):
                                    if (self.sys_unc_table2D[iSource][0][iBin] >= 0) and (self.sys_unc_table2D[iSource][1][iBin] < 0):
                                        sys_group_unc_up[iBin] += np.power(self.sys_unc_table2D[iSource][0][iBin],2)
                                        sys_group_unc_down[iBin] += np.power(self.sys_unc_table2D[iSource][1][iBin],2)
                                    elif (self.sys_unc_table2D[iSource][0][iBin] < 0) and (self.sys_unc_table2D[iSource][1][iBin] >= 0): 
                                        sys_group_unc_up[iBin] += np.power(self.sys_unc_table2D[iSource][1][iBin],2)
                                        sys_group_unc_down[iBin] += np.power(self.sys_unc_table2D[iSource][0][iBin],2)
                                    elif (self.sys_unc_table2D[iSource][0][iBin] >= 0) and (self.sys_unc_table2D[iSource][1][iBin] >= 0):
                                        if np.abs(self.sys_unc_table2D[iSource][0][iBin]) >= np.abs(self.sys_unc_table2D[iSource][1][iBin]):
                                            sys_group_unc_up[iBin] += np.power(self.sys_unc_table2D[iSource][0][iBin],2)
                                        else:
                                            sys_group_unc_up[iBin] += np.power(self.sys_unc_table2D[iSource][1][iBin],2)
                                    elif (self.sys_unc_table2D[iSource][0][iBin] < 0) and (self.sys_unc_table2D[iSource][1][iBin] < 0):
                                        if np.abs(self.sys_unc_table2D[iSource][0][iBin]) >= np.abs(self.sys_unc_table2D[iSource][1][iBin]):
                                            sys_group_unc_down[iBin] += np.power(self.sys_unc_table2D[iSource][0][iBin],2)
                                        else:
                                            sys_group_unc_down[iBin] += np.power(self.sys_unc_table2D[iSource][1][iBin],2)
                    sys_group_unc_up = np.sqrt(sys_group_unc_up)
                    sys_group_unc_down = np.sqrt(sys_group_unc_down)*(-1)
                if isinstance(self.groups[ig], list):
                    self.sys_groups_name.append(self.groups[ig][0])
                    for isys in range(len(self.sys_IDs)):
                        iSource = self.sys_IDs[isys]
                        if self.hist_table3D[iSource][0] != 0:
                            if iSource > 0 and self.sys_labels[isys] in self.groups[ig][1:]: 
                                sys_group_list.append(self.sys_labels[isys])
                                for iBin in range(len(self.bins)-1):
                                    if (self.sys_unc_table2D[iSource][0][iBin] >= 0) and (self.sys_unc_table2D[iSource][1][iBin] < 0):
                                        sys_group_unc_up[iBin] += np.power(self.sys_unc_table2D[iSource][0][iBin],2)
                                        sys_group_unc_down[iBin] += np.power(self.sys_unc_table2D[iSource][1][iBin],2)
                                    elif (self.sys_unc_table2D[iSource][0][iBin] < 0) and (self.sys_unc_table2D[iSource][1][iBin] >= 0): 
                                        sys_group_unc_up[iBin] += np.power(self.sys_unc_table2D[iSource][1][iBin],2)
                                        sys_group_unc_down[iBin] += np.power(self.sys_unc_table2D[iSource][0][iBin],2)
                                    elif (self.sys_unc_table2D[iSource][0][iBin] >= 0) and (self.sys_unc_table2D[iSource][1][iBin] >= 0):
                                        if np.abs(self.sys_unc_table2D[iSource][0][iBin]) >= np.abs(self.sys_unc_table2D[iSource][1][iBin]):
                                            sys_group_unc_up[iBin] += np.power(self.sys_unc_table2D[iSource][0][iBin],2)
                                        else:
                                            sys_group_unc_up[iBin] += np.power(self.sys_unc_table2D[iSource][1][iBin],2)
                                    elif (self.sys_unc_table2D[iSource][0][iBin] < 0) and (self.sys_unc_table2D[iSource][1][iBin] < 0):
                                        if np.abs(self.sys_unc_table2D[iSource][0][iBin]) >= np.abs(self.sys_unc_table2D[iSource][1][iBin]):
                                            sys_group_unc_down[iBin] += np.power(self.sys_unc_table2D[iSource][0][iBin],2)
                                        else:
                                            sys_group_unc_down[iBin] += np.power(self.sys_unc_table2D[iSource][1][iBin],2)
                    sys_group_unc_up = np.sqrt(sys_group_unc_up)
                    sys_group_unc_down = np.sqrt(sys_group_unc_down)*(-1)
                self.sys_groups_list.append(sys_group_list)
                self.sys_groups_unc_up.append(sys_group_unc_up)
                self.sys_groups_unc_down.append(sys_group_unc_down)
            #print("sys_groups_list", self.sys_groups_list)
            #print("sys_groups_name", self.sys_groups_name)
            #print("sys_groups_unc_up", self.sys_groups_unc_up)
            #print("sys_groups_unc_down", self.sys_groups_unc_down)
            

    #==============================================================================================================
    def create_signal_tables2D(self):

        #Initialize systematic table2D (first index is source, second index is universe)
        self.signal_hist_table2D = []                   
        self.signal_sys_unc_table2D = []    # Use in Fractional plot
        for i in range(self.N_sources):
            self.signal_hist_table2D.append([0, 0])
            self.signal_sys_unc_table2D.append([np.zeros(len(self.bins)-1), np.zeros(len(self.bins)-1)])

    
        # Filling hist_table2D
        for iSource in range(self.N_sources):
            if self.hist_table3D[iSource][0] != 0:
                if iSource == 0:
                    self.signal_hist_table2D[0][0] = self.hist_table3D[0][0][0]
                if iSource > 0: 
                    for iUniverse in range(2):
                        self.signal_hist_table2D[iSource][iUniverse] = self.hist_table3D[iSource][iUniverse][0]

        for iSource in range(self.N_sources):
            if self.hist_table3D[iSource][0] != 0:
                if iSource > 0: 
                    for iUniverse in range(2):
                        self.signal_sys_unc_table2D[iSource][iUniverse] = self.signal_sys_unc_table2D[iSource][iUniverse] + self.sys_unc_table3D[iSource][iUniverse][0]
    
        self.hist_signal = self.signal_hist_table2D[0][0]
    
        #--------------------------------------------------------------------------------
        # Transform uncertainties in fractional uncertainties
        for iSource in range(self.N_sources):
            if self.hist_table3D[iSource][0] != 0:
                if iSource > 0: 
                    for iUniverse in range(2):
                        for iBin in range(len(self.bins)-1):
                            if self.signal_hist_table2D[0][0][iBin] > 0:
                                self.signal_sys_unc_table2D[iSource][iUniverse][iBin] = self.signal_sys_unc_table2D[iSource][iUniverse][iBin]/self.signal_hist_table2D[0][0][iBin]
                            else: 
                                self.signal_sys_unc_table2D[iSource][iUniverse][iBin] = 0
   
        #--------------------------------------------------------------------------------
        # Get total fractional uncertainty        
        self.signal_sys_total_unc_up = np.zeros(len(self.bins)-1)
        self.signal_sys_total_unc_down = np.zeros(len(self.bins)-1)
            
        for iSource in range(self.N_sources):
            if self.hist_table3D[iSource][0] != 0:
                if iSource > 0: 
                    for iBin in range(len(self.bins)-1):
                        if (self.signal_sys_unc_table2D[iSource][0][iBin] >= 0) and (self.signal_sys_unc_table2D[iSource][1][iBin] < 0):
                            self.signal_sys_total_unc_up[iBin] += np.power(self.signal_sys_unc_table2D[iSource][0][iBin],2)
                            self.signal_sys_total_unc_down[iBin] += np.power(self.signal_sys_unc_table2D[iSource][1][iBin],2)
                        elif (self.signal_sys_unc_table2D[iSource][0][iBin] < 0) and (self.signal_sys_unc_table2D[iSource][1][iBin] >= 0): 
                            self.signal_sys_total_unc_up[iBin] += np.power(self.signal_sys_unc_table2D[iSource][1][iBin],2)
                            self.signal_sys_total_unc_down[iBin] += np.power(self.signal_sys_unc_table2D[iSource][0][iBin],2)
                        elif (self.signal_sys_unc_table2D[iSource][0][iBin] >= 0) and (self.signal_sys_unc_table2D[iSource][1][iBin] >= 0):
                            if np.abs(self.signal_sys_unc_table2D[iSource][0][iBin]) >= np.abs(self.signal_sys_unc_table2D[iSource][1][iBin]):
                                self.signal_sys_total_unc_up[iBin] += np.power(self.signal_sys_unc_table2D[iSource][0][iBin],2)
                            else:
                                self.signal_sys_total_unc_up[iBin] += np.power(self.signal_sys_unc_table2D[iSource][1][iBin],2)
                        elif (self.signal_sys_unc_table2D[iSource][0][iBin] < 0) and (self.signal_sys_unc_table2D[iSource][1][iBin] < 0):
                            if np.abs(self.signal_sys_unc_table2D[iSource][0][iBin]) >= np.abs(self.signal_sys_unc_table2D[iSource][1][iBin]):
                                self.signal_sys_total_unc_down[iBin] += np.power(self.signal_sys_unc_table2D[iSource][0][iBin],2)
                            else:
                                self.signal_sys_total_unc_down[iBin] += np.power(self.signal_sys_unc_table2D[iSource][1][iBin],2)
                        
        self.signal_sys_total_unc_up = np.sqrt(self.signal_sys_total_unc_up)
        self.signal_sys_total_unc_down = np.sqrt(self.signal_sys_total_unc_down)*(-1)
        
        #--------------------------------------------------------------------------------
        # Get groups systematic fractional uncertainty
        if self.groups is not None:
            self.signal_sys_groups_list = []
            self.signal_sys_groups_name = []
            self.signal_sys_groups_unc_up = []
            self.signal_sys_groups_unc_down = []
            for ig in range(len(self.groups)):
                sys_group_list = []
                sys_group_unc_up = np.zeros(len(self.bins)-1)
                sys_group_unc_down = np.zeros(len(self.bins)-1)
                if isinstance(self.groups[ig], str):
                    self.signal_sys_groups_name.append(self.groups[ig])
                    prefix = self.groups[ig]+"_"
                    for isys in range(len(self.sys_IDs)):
                        iSource = self.sys_IDs[isys]
                        if self.hist_table3D[iSource][0] != 0:
                            if iSource > 0 and self.sys_labels[isys][:len(prefix)] == prefix: 
                                sys_group_list.append(self.sys_labels[isys])
                                for iBin in range(len(self.bins)-1):
                                    if (self.signal_sys_unc_table2D[iSource][0][iBin] >= 0) and (self.signal_sys_unc_table2D[iSource][1][iBin] < 0):
                                        sys_group_unc_up[iBin] += np.power(self.signal_sys_unc_table2D[iSource][0][iBin],2)
                                        sys_group_unc_down[iBin] += np.power(self.signal_sys_unc_table2D[iSource][1][iBin],2)
                                    elif (self.signal_sys_unc_table2D[iSource][0][iBin] < 0) and (self.signal_sys_unc_table2D[iSource][1][iBin] >= 0): 
                                        sys_group_unc_up[iBin] += np.power(self.signal_sys_unc_table2D[iSource][1][iBin],2)
                                        sys_group_unc_down[iBin] += np.power(self.signal_sys_unc_table2D[iSource][0][iBin],2)
                                    elif (self.signal_sys_unc_table2D[iSource][0][iBin] >= 0) and (self.signal_sys_unc_table2D[iSource][1][iBin] >= 0):
                                        if np.abs(self.signal_sys_unc_table2D[iSource][0][iBin]) >= np.abs(self.signal_sys_unc_table2D[iSource][1][iBin]):
                                            sys_group_unc_up[iBin] += np.power(self.signal_sys_unc_table2D[iSource][0][iBin],2)
                                        else:
                                            sys_group_unc_up[iBin] += np.power(self.signal_sys_unc_table2D[iSource][1][iBin],2)
                                    elif (self.signal_sys_unc_table2D[iSource][0][iBin] < 0) and (self.signal_sys_unc_table2D[iSource][1][iBin] < 0):
                                        if np.abs(self.signal_sys_unc_table2D[iSource][0][iBin]) >= np.abs(self.signal_sys_unc_table2D[iSource][1][iBin]):
                                            sys_group_unc_down[iBin] += np.power(self.signal_sys_unc_table2D[iSource][0][iBin],2)
                                        else:
                                            sys_group_unc_down[iBin] += np.power(self.signal_sys_unc_table2D[iSource][1][iBin],2)
                    sys_group_unc_up = np.sqrt(sys_group_unc_up)
                    sys_group_unc_down = np.sqrt(sys_group_unc_down)*(-1)
                if isinstance(self.groups[ig], list):
                    self.signal_sys_groups_name.append(self.groups[ig][0])
                    for isys in range(len(self.sys_IDs)):
                        iSource = self.sys_IDs[isys]
                        if self.hist_table3D[iSource][0] != 0:
                            if iSource > 0 and self.sys_labels[isys] in self.groups[ig][1:]: 
                                sys_group_list.append(self.sys_labels[isys])
                                for iBin in range(len(self.bins)-1):
                                    if (self.signal_sys_unc_table2D[iSource][0][iBin] >= 0) and (self.signal_sys_unc_table2D[iSource][1][iBin] < 0):
                                        sys_group_unc_up[iBin] += np.power(self.signal_sys_unc_table2D[iSource][0][iBin],2)
                                        sys_group_unc_down[iBin] += np.power(self.signal_sys_unc_table2D[iSource][1][iBin],2)
                                    elif (self.signal_sys_unc_table2D[iSource][0][iBin] < 0) and (self.signal_sys_unc_table2D[iSource][1][iBin] >= 0): 
                                        sys_group_unc_up[iBin] += np.power(self.signal_sys_unc_table2D[iSource][1][iBin],2)
                                        sys_group_unc_down[iBin] += np.power(self.signal_sys_unc_table2D[iSource][0][iBin],2)
                                    elif (self.signal_sys_unc_table2D[iSource][0][iBin] >= 0) and (self.signal_sys_unc_table2D[iSource][1][iBin] >= 0):
                                        if np.abs(self.signal_sys_unc_table2D[iSource][0][iBin]) >= np.abs(self.signal_sys_unc_table2D[iSource][1][iBin]):
                                            sys_group_unc_up[iBin] += np.power(self.signal_sys_unc_table2D[iSource][0][iBin],2)
                                        else:
                                            sys_group_unc_up[iBin] += np.power(self.signal_sys_unc_table2D[iSource][1][iBin],2)
                                    elif (self.signal_sys_unc_table2D[iSource][0][iBin] < 0) and (self.signal_sys_unc_table2D[iSource][1][iBin] < 0):
                                        if np.abs(self.signal_sys_unc_table2D[iSource][0][iBin]) >= np.abs(self.signal_sys_unc_table2D[iSource][1][iBin]):
                                            sys_group_unc_down[iBin] += np.power(self.signal_sys_unc_table2D[iSource][0][iBin],2)
                                        else:
                                            sys_group_unc_down[iBin] += np.power(self.signal_sys_unc_table2D[iSource][1][iBin],2)
                    sys_group_unc_up = np.sqrt(sys_group_unc_up)
                    sys_group_unc_down = np.sqrt(sys_group_unc_down)*(-1)
                self.signal_sys_groups_list.append(sys_group_list)
                self.signal_sys_groups_unc_up.append(sys_group_unc_up)
                self.signal_sys_groups_unc_down.append(sys_group_unc_down)
            #print("signal_sys_groups_list", self.signal_sys_groups_list)
            #print("signal_sys_groups_name", self.signal_sys_groups_name)
            #print("signal_sys_groups_unc_up", self.signal_sys_groups_unc_up)
            #print("signal_sys_groups_unc_down", self.signal_sys_groups_unc_down)
            
        
    #==============================================================================================================
    def frac_unc_tables3D(self):
        # Transform uncertainties in fractional uncertainties
        for iSource in range(self.N_sources):
            if self.hist_table3D[iSource][0] != 0:
                if iSource > 0: 
                    for iUniverse in range(2):
                        for iProcess in range(self.number_ds_groups):
                            for iBin in range(len(self.bins)-1):
                                if self.hist_table3D[0][0][iProcess][iBin] > 0.0001:
                                    self.sys_unc_table3D[iSource][iUniverse][iProcess][iBin] = self.sys_unc_table3D[iSource][iUniverse][iProcess][iBin]/self.hist_table3D[0][0][iProcess][iBin]
                                else: 
                                    self.sys_unc_table3D[iSource][iUniverse][iProcess][iBin] = 0
    
    #==============================================================================================================
    def set_data(self, data_group):
        
        item_type = type(data_group).__name__
        if item_type != "list": 
            data_group = [data_group]
        
        self.has_data = True
        
        #Initialize data histograms
        self.hist_data = np.zeros(len(self.bins)-1)                  
        self.unc_data = np.zeros(len(self.bins)-1)
            
        for iProcDic in range(len(data_group)):  # loop in the proc_dictionaries inside the lists
            proc_dic = data_group[iProcDic][self.var+"_"+str(self.region)+"_0_0"]
            Hist_raw = proc_dic["Hist"]
            Unc_raw = proc_dic["Unc"]
            Start_raw = proc_dic["Start"]
            End_raw = proc_dic["End"]
            Nbins_raw = proc_dic["Nbins"]
            Delta_raw = (End_raw - Start_raw)/Nbins_raw
                
            Hist_new = [0]*(len(self.bins)-1)
            Unc_new = [0]*(len(self.bins)-1)
                                    
            for iRawBin in range(Nbins_raw):
                inf_raw = Start_raw + iRawBin*Delta_raw
                sup_raw = Start_raw + (iRawBin+1)*Delta_raw
                for iNewBin in range(len(Hist_new)):
                    if( (inf_raw >= self.bins[iNewBin]) and (sup_raw <= self.bins[iNewBin+1]) ):
                        Hist_new[iNewBin] = Hist_new[iNewBin] + Hist_raw[iRawBin]
                        Unc_new[iNewBin] = np.sqrt( Unc_new[iNewBin]**2 + Unc_raw[iRawBin]**2 )
                    
            for iNewBin in range(len(Hist_new)):
                if Hist_new[iNewBin] < 0:
                    Hist_new[iNewBin] = 0
                    Unc_new[iNewBin] = 0
                    
            self.hist_data = self.hist_data + np.array(Hist_new)
            self.unc_data = np.sqrt(self.unc_data**2 + np.array(Unc_new)**2)
    
    #==============================================================================================================
    def frac_bkg_syst_plot(self, ax, version=1, width="physical"):
        
        if width == "physical":
            plot_bins = self.bins
        elif width == "same":
            plot_bins = range(len(self.bins))
            #print("Physical bins are:", self.bins)
            
        veto_sys_group_list = []
        if self.groups is not None:
            for i1 in range(len(self.sys_groups_list)):
                for i2 in range(len(self.sys_groups_list[i1])):
                    veto_sys_group_list.append(self.sys_groups_list[i1][i2])
                    
        if version == 2:
            plt.axhline(0, color='black', linewidth=1)
            hist_up = np.insert(self.sys_total_unc_up, 0, self.sys_total_unc_up[0], axis=0)
            hist_down = np.insert(self.sys_total_unc_down, 0, self.sys_total_unc_down[0], axis=0)
            plt.step(plot_bins, hist_up, label="Total", color="black", linewidth=1.5 )
            plt.step(plot_bins, hist_down, linestyle="--", color="black", linewidth=1.5 )
            
            i_colors = 0
            for i in range(len(self.sys_IDs)):
                if self.sys_labels[i] not in veto_sys_group_list:
                    hist_up = np.insert(self.sys_unc_table2D[self.sys_IDs[i]][1], 0, self.sys_unc_table2D[self.sys_IDs[i]][1][0], axis=0)
                    hist_down = np.insert(self.sys_unc_table2D[self.sys_IDs[i]][0], 0, self.sys_unc_table2D[self.sys_IDs[i]][0][0], axis=0)
                    plt.step(plot_bins, hist_up, label=self.sys_labels[i], color=self.sys_colors[i_colors], linewidth=1.5 )
                    plt.step(plot_bins, hist_down, linestyle="--", color=self.sys_colors[i_colors], linewidth=1.5 )
                    i_colors += 1
            
            if self.groups is not None:        
                for i in range(len(self.sys_groups_name)):        
                    hist_up = np.insert(self.sys_groups_unc_up[i], 0, self.sys_groups_unc_up[i][0], axis=0)
                    hist_down = np.insert(self.sys_groups_unc_down[i], 0, self.sys_groups_unc_down[i][0], axis=0)
                    plt.step(plot_bins, hist_up, label=self.sys_groups_name[i], color=self.sys_colors[i+i_colors], linewidth=1.5 )
                    plt.step(plot_bins, hist_down, linestyle="--", color=self.sys_colors[i+i_colors], linewidth=1.5 )

        if version == 1:
            x = np.array(plot_bins)
            dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
            x = x[:-1]
            hist_up = np.insert(self.sys_total_unc_up, 0, self.sys_total_unc_up[0], axis=0)
            hist_down = np.insert(self.sys_total_unc_down, 0, self.sys_total_unc_down[0], axis=0)
            plt.step(plot_bins, np.abs(hist_down), color="black", linestyle="dotted", linewidth=1.8)
            plt.step(plot_bins, np.abs(hist_up), label="Total", color="black", linewidth=1.8)
            for ix in range(len(x)):
                
                # DOWN
                if hist_down[1:][ix] >= 0:
                    plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_down[1:][ix]), linewidth=0, marker='^', color="black", markerfacecolor='white', markersize=7, markeredgewidth=0.7)
                else:
                    plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_down[1:][ix]), linewidth=0, marker='v', color="black", markerfacecolor='white', markersize=7, markeredgewidth=0.7)
                
                # UP
                if hist_up[1:][ix] >= 0:
                    plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_up[1:][ix]), linewidth=0, marker='^', color="black", markerfacecolor='black', markersize=7, markeredgewidth=0)
                else:
                    plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_up[1:][ix]), linewidth=0, marker='v', color="black", markerfacecolor='black', markersize=7, markeredgewidth=0)
                
            height = np.abs(np.abs(hist_up[1:])-np.abs(hist_down[1:]))
            bottom = np.minimum(np.abs(hist_down[1:]), np.abs(hist_up[1:]))
            plt.bar(x=plot_bins[:-1], height=height, bottom=bottom, width=np.diff(plot_bins), align='edge', linewidth=0, color='black', alpha=0.15, zorder=-1)
            
            i_colors = 0
            for i in range(len(self.sys_IDs)):
                if self.sys_labels[i] not in veto_sys_group_list:
                    hist_up = np.insert(self.sys_unc_table2D[self.sys_IDs[i]][1], 0, self.sys_unc_table2D[self.sys_IDs[i]][1][0], axis=0)
                    hist_down = np.insert(self.sys_unc_table2D[self.sys_IDs[i]][0], 0, self.sys_unc_table2D[self.sys_IDs[i]][0][0], axis=0)
                    sys_color=self.sys_colors[i_colors]

                    plt.step(plot_bins, np.abs(hist_up), label=self.sys_labels[i], color=sys_color, linewidth=1.8, linestyle="-" )
                    plt.step(plot_bins, np.abs(hist_down), color=sys_color, linewidth=1.8, linestyle="dotted" )
                    for ix in range(len(x)):
                        
                        # DOWN
                        if hist_down[1:][ix] >= 0:
                            plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_down[1:][ix]), linewidth=0, marker='^', color=sys_color, markerfacecolor='white', markersize=7, markeredgewidth=0.7, zorder=100+i)
                        else:
                            plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_down[1:][ix]), linewidth=0, marker='v', color=sys_color, markerfacecolor='white', markersize=7, markeredgewidth=0.7, zorder=100+i)
                        
                        # UP
                        if hist_up[1:][ix] >= 0:
                            plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_up[1:][ix]), linewidth=0, marker='^', color=sys_color, markerfacecolor=sys_color, markersize=7, markeredgewidth=0, zorder=100+i)
                        else:
                            plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_up[1:][ix]), linewidth=0, marker='v', color=sys_color, markerfacecolor=sys_color, markersize=7, markeredgewidth=0, zorder=100+i)
                    
                    height = np.abs(np.abs(hist_up[1:])-np.abs(hist_down[1:]))
                    bottom = np.minimum(np.abs(hist_down[1:]), np.abs(hist_up[1:]))
                    plt.bar(x=plot_bins[:-1], height=height, bottom=bottom, width=np.diff(plot_bins), align='edge', linewidth=0, color=sys_color, alpha=0.15, zorder=-2+i)
                    i_colors += 1
                    
            if self.groups is not None:        
                for i in range(len(self.sys_groups_name)):
                    hist_up = np.insert(self.sys_groups_unc_up[i], 0, self.sys_groups_unc_up[i][0], axis=0)
                    hist_down = np.insert(self.sys_groups_unc_down[i], 0, self.sys_groups_unc_down[i][0], axis=0)
                    hist_max = np.maximum(np.abs(hist_down), np.abs(hist_up))
                    sys_color=self.sys_colors[i+i_colors]
                    
                    plt.step(plot_bins, hist_max, label=self.sys_groups_name[i], color=sys_color, linewidth=1.8, linestyle="-" )
    
    #==============================================================================================================
    def frac_signal_syst_plot(self, ax, version=1, width="physical"):
        
        if width == "physical":
            plot_bins = self.bins
        elif width == "same":
            plot_bins = range(len(self.bins)) 
            #print("Physical bins are:", self.bins)
        
        veto_sys_group_list = []
        if self.groups is not None:
            for i1 in range(len(self.signal_sys_groups_list)):
                for i2 in range(len(self.signal_sys_groups_list[i1])):
                    veto_sys_group_list.append(self.signal_sys_groups_list[i1][i2])
        
        if version == 2:
            plt.axhline(0, color='black', linewidth=1)
            hist_up = np.insert(self.signal_sys_total_unc_up, 0, self.signal_sys_total_unc_up[0], axis=0)
            hist_down = np.insert(self.signal_sys_total_unc_down, 0, self.signal_sys_total_unc_down[0], axis=0)
            plt.step(plot_bins, hist_up, label="Total", color="black", linewidth=1.5 )
            plt.step(plot_bins, hist_down, linestyle="--", color="black", linewidth=1.5 )
        
            i_colors = 0
            for i in range(len(self.sys_IDs)):
                if self.sys_labels[i] not in veto_sys_group_list:
                    hist_up = np.insert(self.signal_sys_unc_table2D[self.sys_IDs[i]][1], 0, self.signal_sys_unc_table2D[self.sys_IDs[i]][1][0], axis=0)
                    hist_down = np.insert(self.signal_sys_unc_table2D[self.sys_IDs[i]][0], 0, self.signal_sys_unc_table2D[self.sys_IDs[i]][0][0], axis=0)
                    plt.step(plot_bins, hist_up, label=self.sys_labels[i], color=self.sys_colors[i_colors], linewidth=1.5 )
                    plt.step(plot_bins, hist_down, linestyle="--", color=self.sys_colors[i_colors], linewidth=1.5 )
                    i_colors += 1
                    
            if self.groups is not None:        
                for i in range(len(self.signal_sys_groups_name)):        
                    hist_up = np.insert(self.signal_sys_groups_unc_up[i], 0, self.signal_sys_groups_unc_up[i][0], axis=0)
                    hist_down = np.insert(self.signal_sys_groups_unc_down[i], 0, self.signal_sys_groups_unc_down[i][0], axis=0)
                    plt.step(plot_bins, hist_up, label=self.signal_sys_groups_name[i], color=self.sys_colors[i+i_colors], linewidth=1.5 )
                    plt.step(plot_bins, hist_down, linestyle="--", color=self.sys_colors[i+i_colors], linewidth=1.5 )

        if version == 1:
            x = np.array(plot_bins)
            dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
            x = x[:-1]
            hist_up = np.insert(self.signal_sys_total_unc_up, 0, self.signal_sys_total_unc_up[0], axis=0)
            hist_down = np.insert(self.signal_sys_total_unc_down, 0, self.signal_sys_total_unc_down[0], axis=0)
            plt.step(plot_bins, np.abs(hist_down), color="black", linestyle="dotted", linewidth=1.8)
            plt.step(plot_bins, np.abs(hist_up), label="Total", color="black", linewidth=1.8)
            for ix in range(len(x)):
                
                # DOWN
                if hist_down[1:][ix] >= 0:
                    plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_down[1:][ix]), linewidth=0, marker='^', color="black", markerfacecolor='white', markersize=7, markeredgewidth=0.7)
                else:
                    plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_down[1:][ix]), linewidth=0, marker='v', color="black", markerfacecolor='white', markersize=7, markeredgewidth=0.7)
                
                # UP
                if hist_up[1:][ix] >= 0:
                    plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_up[1:][ix]), linewidth=0, marker='^', color="black", markerfacecolor='black', markersize=7, markeredgewidth=0)
                else:
                    plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_up[1:][ix]), linewidth=0, marker='v', color="black", markerfacecolor='black', markersize=7, markeredgewidth=0)
                
            height = np.abs(np.abs(hist_up[1:])-np.abs(hist_down[1:]))
            bottom = np.minimum(np.abs(hist_down[1:]), np.abs(hist_up[1:]))
            plt.bar(x=plot_bins[:-1], height=height, bottom=bottom, width=np.diff(plot_bins), align='edge', linewidth=0, color='black', alpha=0.15, zorder=-1)
            
            i_colors = 0
            for i in range(len(self.sys_IDs)):
                if self.sys_labels[i] not in veto_sys_group_list:
                    hist_up = np.insert(self.signal_sys_unc_table2D[self.sys_IDs[i]][1], 0, self.signal_sys_unc_table2D[self.sys_IDs[i]][1][0], axis=0)
                    hist_down = np.insert(self.signal_sys_unc_table2D[self.sys_IDs[i]][0], 0, self.signal_sys_unc_table2D[self.sys_IDs[i]][0][0], axis=0)
                    sys_color=self.sys_colors[i_colors]

                    plt.step(plot_bins, np.abs(hist_up), label=self.sys_labels[i], color=sys_color, linewidth=1.8, linestyle="-" )
                    plt.step(plot_bins, np.abs(hist_down), color=sys_color, linewidth=1.8, linestyle="dotted" )
                    for ix in range(len(x)):
                        
                        # DOWN
                        if hist_down[1:][ix] >= 0:
                            plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_down[1:][ix]), linewidth=0, marker='^', color=sys_color, markerfacecolor='white', markersize=7, markeredgewidth=0.7, zorder=100+i)
                        else:
                            plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_down[1:][ix]), linewidth=0, marker='v', color=sys_color, markerfacecolor='white', markersize=7, markeredgewidth=0.7, zorder=100+i)
                        
                        # UP
                        if hist_up[1:][ix] >= 0:
                            plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_up[1:][ix]), linewidth=0, marker='^', color=sys_color, markerfacecolor=sys_color, markersize=7, markeredgewidth=0, zorder=100+i)
                        else:
                            plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_up[1:][ix]), linewidth=0, marker='v', color=sys_color, markerfacecolor=sys_color, markersize=7, markeredgewidth=0, zorder=100+i)
                    
                    height = np.abs(np.abs(hist_up[1:])-np.abs(hist_down[1:]))
                    bottom = np.minimum(np.abs(hist_down[1:]), np.abs(hist_up[1:]))
                    plt.bar(x=plot_bins[:-1], height=height, bottom=bottom, width=np.diff(plot_bins), align='edge', linewidth=0, color=sys_color, alpha=0.15, zorder=-2+i)
                    i_colors += 1
                    
            if self.groups is not None:        
                for i in range(len(self.signal_sys_groups_name)):
                    hist_up = np.insert(self.signal_sys_groups_unc_up[i], 0, self.signal_sys_groups_unc_up[i][0], axis=0)
                    hist_down = np.insert(self.signal_sys_groups_unc_down[i], 0, self.signal_sys_groups_unc_down[i][0], axis=0)
                    hist_max = np.maximum(np.abs(hist_down), np.abs(hist_up))
                    sys_color=self.sys_colors[i+i_colors]
                    
                    plt.step(plot_bins, hist_max, label=self.signal_sys_groups_name[i], color=sys_color, linewidth=1.8, linestyle="-" )
                    
                    
    #==============================================================================================================
    def frac_bkg_group_syst_plot(self, ax, group_name, version=1, width="physical"):
        
        if self.groups is not None:
        
            if width == "physical":
                plot_bins = self.bins
            elif width == "same":
                plot_bins = range(len(self.bins))
                #print("Physical bins are:", self.bins)
                
            sys_group_list = []
            i_group = 0
            for i in range(len(self.sys_groups_list)):
                if self.sys_groups_name[i] == group_name:
                    sys_group_list = self.sys_groups_list[i]
                    i_group = i
                        
            if version == 2:
                plt.axhline(0, color='black', linewidth=1)
                hist_up = np.insert(self.sys_groups_unc_up[i_group], 0, self.sys_groups_unc_up[i_group][0], axis=0)
                hist_down = np.insert(self.sys_groups_unc_down[i_group], 0, self.sys_groups_unc_down[i_group][0], axis=0)
                plt.step(plot_bins, hist_up, label=group_name, color="black", linewidth=1.5 )
                plt.step(plot_bins, hist_down, linestyle="--", color="black", linewidth=1.5 )
                
                i_colors = 1
                for i in range(len(self.sys_IDs)):
                    if self.sys_labels[i] in sys_group_list:
                        hist_up = np.insert(self.sys_unc_table2D[self.sys_IDs[i]][1], 0, self.sys_unc_table2D[self.sys_IDs[i]][1][0], axis=0)
                        hist_down = np.insert(self.sys_unc_table2D[self.sys_IDs[i]][0], 0, self.sys_unc_table2D[self.sys_IDs[i]][0][0], axis=0)
                        plot_label = self.sys_labels[i]
                        prefix = group_name + "_"
                        if prefix == plot_label[:len(prefix)]:
                            plot_label = plot_label[len(prefix):]
                        plt.step(plot_bins, hist_up, label=plot_label, color=self.sys_colors[i_colors], linewidth=1.5 )
                        plt.step(plot_bins, hist_down, linestyle="--", color=self.sys_colors[i_colors], linewidth=1.5 )
                        i_colors += 1

            if version == 1:
                x = np.array(plot_bins)
                dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
                x = x[:-1]
                hist_up = np.insert(self.sys_groups_unc_up[i_group], 0, self.sys_groups_unc_up[i_group][0], axis=0)
                hist_down = np.insert(self.sys_groups_unc_down[i_group], 0, self.sys_groups_unc_down[i_group][0], axis=0)
                hist_max = np.maximum(np.abs(hist_down), np.abs(hist_up))
                plt.step(plot_bins, hist_max, label=group_name, color="black", linewidth=1.8)
                
                i_colors = 1
                for i in range(len(self.sys_IDs)):
                    if self.sys_labels[i] in sys_group_list:
                        hist_up = np.insert(self.sys_unc_table2D[self.sys_IDs[i]][1], 0, self.sys_unc_table2D[self.sys_IDs[i]][1][0], axis=0)
                        hist_down = np.insert(self.sys_unc_table2D[self.sys_IDs[i]][0], 0, self.sys_unc_table2D[self.sys_IDs[i]][0][0], axis=0)
                        
                        plot_label = self.sys_labels[i]
                        prefix = group_name + "_"
                        if prefix == plot_label[:len(prefix)]:
                            plot_label = plot_label[len(prefix):]
                        sys_color=self.sys_colors[i_colors]

                        plt.step(plot_bins, np.abs(hist_up), label=plot_label, color=sys_color, linewidth=1.8, linestyle="-" )
                        plt.step(plot_bins, np.abs(hist_down), color=sys_color, linewidth=1.8, linestyle="dotted" )
                        for ix in range(len(x)):
                            
                            # DOWN
                            if hist_down[1:][ix] >= 0:
                                plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_down[1:][ix]), linewidth=0, marker='^', color=sys_color, markerfacecolor='white', markersize=7, markeredgewidth=0.7, zorder=100+i)
                            else:
                                plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_down[1:][ix]), linewidth=0, marker='v', color=sys_color, markerfacecolor='white', markersize=7, markeredgewidth=0.7, zorder=100+i)
                            
                            # UP
                            if hist_up[1:][ix] >= 0:
                                plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_up[1:][ix]), linewidth=0, marker='^', color=sys_color, markerfacecolor=sys_color, markersize=7, markeredgewidth=0, zorder=100+i)
                            else:
                                plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_up[1:][ix]), linewidth=0, marker='v', color=sys_color, markerfacecolor=sys_color, markersize=7, markeredgewidth=0, zorder=100+i)
                        
                        height = np.abs(np.abs(hist_up[1:])-np.abs(hist_down[1:]))
                        bottom = np.minimum(np.abs(hist_down[1:]), np.abs(hist_up[1:]))
                        plt.bar(x=plot_bins[:-1], height=height, bottom=bottom, width=np.diff(plot_bins), align='edge', linewidth=0, color=sys_color, alpha=0.15, zorder=-2+i)
                        i_colors += 1
                        
            
    #==============================================================================================================
    def frac_signal_group_syst_plot(self, ax, group_name, version=1, width="physical"):
        
        if self.groups is not None:
        
            if width == "physical":
                plot_bins = self.bins
            elif width == "same":
                plot_bins = range(len(self.bins))
                #print("Physical bins are:", self.bins)
                
            sys_group_list = []
            i_group = 0
            for i in range(len(self.signal_sys_groups_list)):
                if self.signal_sys_groups_name[i] == group_name:
                    sys_group_list = self.signal_sys_groups_list[i]
                    i_group = i
                        
            if version == 2:
                plt.axhline(0, color='black', linewidth=1)
                hist_up = np.insert(self.signal_sys_groups_unc_up[i_group], 0, self.signal_sys_groups_unc_up[i_group][0], axis=0)
                hist_down = np.insert(self.signal_sys_groups_unc_down[i_group], 0, self.signal_sys_groups_unc_down[i_group][0], axis=0)
                plt.step(plot_bins, hist_up, label=group_name, color="black", linewidth=1.5 )
                plt.step(plot_bins, hist_down, linestyle="--", color="black", linewidth=1.5 )
                
                i_colors = 1
                for i in range(len(self.sys_IDs)):
                    if self.sys_labels[i] in sys_group_list:
                        hist_up = np.insert(self.signal_sys_unc_table2D[self.sys_IDs[i]][1], 0, self.signal_sys_unc_table2D[self.sys_IDs[i]][1][0], axis=0)
                        hist_down = np.insert(self.signal_sys_unc_table2D[self.sys_IDs[i]][0], 0, self.signal_sys_unc_table2D[self.sys_IDs[i]][0][0], axis=0)
                        plot_label = self.sys_labels[i]
                        prefix = group_name + "_"
                        if prefix == plot_label[:len(prefix)]:
                            plot_label = plot_label[len(prefix):]
                        plt.step(plot_bins, hist_up, label=plot_label, color=self.sys_colors[i_colors], linewidth=1.5 )
                        plt.step(plot_bins, hist_down, linestyle="--", color=self.sys_colors[i_colors], linewidth=1.5 )
                        i_colors += 1

            if version == 1:
                x = np.array(plot_bins)
                dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
                x = x[:-1]
                hist_up = np.insert(self.signal_sys_groups_unc_up[i_group], 0, self.signal_sys_groups_unc_up[i_group][0], axis=0)
                hist_down = np.insert(self.signal_sys_groups_unc_down[i_group], 0, self.signal_sys_groups_unc_down[i_group][0], axis=0)
                hist_max = np.maximum(np.abs(hist_down), np.abs(hist_up))
                plt.step(plot_bins, hist_max, label=group_name, color="black", linewidth=1.8)
                
                i_colors = 1
                for i in range(len(self.sys_IDs)):
                    if self.sys_labels[i] in sys_group_list:
                        hist_up = np.insert(self.signal_sys_unc_table2D[self.sys_IDs[i]][1], 0, self.signal_sys_unc_table2D[self.sys_IDs[i]][1][0], axis=0)
                        hist_down = np.insert(self.signal_sys_unc_table2D[self.sys_IDs[i]][0], 0, self.signal_sys_unc_table2D[self.sys_IDs[i]][0][0], axis=0)
                        
                        plot_label = self.sys_labels[i]
                        prefix = group_name + "_"
                        if prefix == plot_label[:len(prefix)]:
                            plot_label = plot_label[len(prefix):]
                        sys_color=self.sys_colors[i_colors]

                        plt.step(plot_bins, np.abs(hist_up), label=plot_label, color=sys_color, linewidth=1.8, linestyle="-" )
                        plt.step(plot_bins, np.abs(hist_down), color=sys_color, linewidth=1.8, linestyle="dotted" )
                        for ix in range(len(x)):
                            
                            # DOWN
                            if hist_down[1:][ix] >= 0:
                                plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_down[1:][ix]), linewidth=0, marker='^', color=sys_color, markerfacecolor='white', markersize=7, markeredgewidth=0.7, zorder=100+i)
                            else:
                                plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_down[1:][ix]), linewidth=0, marker='v', color=sys_color, markerfacecolor='white', markersize=7, markeredgewidth=0.7, zorder=100+i)
                            
                            # UP
                            if hist_up[1:][ix] >= 0:
                                plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_up[1:][ix]), linewidth=0, marker='^', color=sys_color, markerfacecolor=sys_color, markersize=7, markeredgewidth=0, zorder=100+i)
                            else:
                                plt.plot(x[ix]+0.5*dx[ix], np.abs(hist_up[1:][ix]), linewidth=0, marker='v', color=sys_color, markerfacecolor=sys_color, markersize=7, markeredgewidth=0, zorder=100+i)
                        
                        height = np.abs(np.abs(hist_up[1:])-np.abs(hist_down[1:]))
                        bottom = np.minimum(np.abs(hist_down[1:]), np.abs(hist_up[1:]))
                        plt.bar(x=plot_bins[:-1], height=height, bottom=bottom, width=np.diff(plot_bins), align='edge', linewidth=0, color=sys_color, alpha=0.15, zorder=-2+i)
                        i_colors += 1
        
        
    #==============================================================================================================
    def stacked_plot(self, ax, width="physical"):
        
        if width == "physical":
            plot_bins = self.bins
        elif width == "same":
            plot_bins = range(len(self.bins)) 
            print("Physical bins are:", self.bins)
        
        hist = np.zeros(len(plot_bins))
        for i in range(self.number_ds_groups):
            if i >= 1 :
                hist += np.insert(self.hist_table3D[0][0][i], 0, self.hist_table3D[0][0][i][0], axis=0)
                #plt.step(plot_bins, hist, label=self.labels[i], color=self.colors[i] )
                plt.fill_between(plot_bins, hist, step="pre", label=self.labels[i], color=self.colors[i], linewidth=0, zorder=-i*5) 
        
        yl = self.hist_bkg*(1 + self.sys_total_unc_down)
        yh = self.hist_bkg*(1 + self.sys_total_unc_up)
        x = np.array(plot_bins)
        dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
        x = x[:-1]
        dy = yh - yl
        pats = [ pat.Rectangle( (x[i], yl[i]), dx[i], dy[i], hatch='/////', fill=False, linewidth=0, edgecolor='grey', zorder=(self.number_ds_groups+1)*5 ) for i in range(len(x)-1) ]
        pats.append(pat.Rectangle( (x[len(x)-1], yl[len(x)-1]), dx[len(x)-1], dy[len(x)-1], hatch='/////', fill=False, linewidth=0, edgecolor='grey', label="Syst. Unc."))
        for p in pats:
            ax.add_patch(p) 
        
        return self.hist_bkg, self.sys_total_unc_up, self.sys_total_unc_down
    
    #==============================================================================================================
    def data_plot(self, ax, width="physical"):
        
        if width == "physical":
            plot_bins = self.bins
        elif width == "same":
            plot_bins = range(len(self.bins))   
            #print("Physical bins are:", self.bins)
        
        if self.has_data:
            x = np.array(plot_bins)
            dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
            x = x[:-1]
            ax.errorbar(
                x+0.5*dx, 
                self.hist_data, 
                yerr=[self.unc_data, self.unc_data], 
                xerr=0.5*dx, 
                fmt='.', 
                ecolor="black", 
                label="Data", 
                color="black", 
                elinewidth=0.7, 
                capsize=0
            )  
            
            return self.hist_data, self.unc_data
        else:
            print("Error: data is not set!")
    
    #==============================================================================================================
    def signal_plot(self, ax, width="physical"):
        
        if width == "physical":
            plot_bins = self.bins
        elif width == "same":
            plot_bins = range(len(self.bins)) 
            print("Physical bins are:", self.bins)
        
        x = np.array(plot_bins)
        dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
        x = x[:-1]
        
        ext_hist_signal = np.append([self.hist_signal[0]], self.hist_signal)
        
        plt.step(plot_bins, ext_hist_signal, color=self.colors[0], label=self.labels[0], linewidth=1.5)
        
        ax.errorbar(
            x+0.5*dx, 
            self.hist_signal, 
            yerr=[-1*self.hist_signal*self.signal_sys_total_unc_down, self.hist_signal*self.signal_sys_total_unc_up], 
            fmt=',', 
            color="blue",
            elinewidth=1
        )  
        
        return self.hist_signal, self.signal_sys_total_unc_up, self.signal_sys_total_unc_down
       
    #==============================================================================================================
    def ratio_plot(self, ax, width="physical"):
        
        if width == "physical":
            plot_bins = self.bins
        elif width == "same":
            plot_bins = range(len(self.bins)) 
            #print("Physical bins are:", self.bins)
        
        if self.has_data:
            x = np.array(plot_bins)
            dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
            x = x[:-1]
            yratio = np.zeros(self.hist_data.size)
            yeratio = np.zeros(self.hist_data.size)
            y2ratio = np.zeros(self.hist_data.size)
            for i in range(self.hist_data.size):
                if self.hist_bkg[i] == 0:
                    yratio[i] = 99999
                    yeratio[i] = 0
                else:
                    yratio[i] = self.hist_data[i]/self.hist_bkg[i]
                    yeratio[i] = self.unc_data[i]/self.hist_bkg[i]
                    y2ratio[i] = self.hist_bkg[i]/self.hist_bkg[i]
            
            yl = 1 + self.sys_total_unc_down
            yh = 1 + self.sys_total_unc_up
            dy = yh - yl
            pats = [ pat.Rectangle( (x[i], yl[i]), dx[i], dy[i], hatch='/////', fill=False, linewidth=0, edgecolor='grey' ) for i in range(len(x)-1) ]
            pats.append(pat.Rectangle( (x[len(x)-1], yl[len(x)-1]), dx[len(x)-1], dy[len(x)-1], hatch='/////', fill=False, linewidth=0, edgecolor='grey' ))
            for p in pats:
                ax.add_patch(p) 
    
            ax.axhline(1, color='red', linestyle='-', linewidth=0.5)
    
            ax.errorbar(x+0.5*dx, yratio, yerr=[yeratio, yeratio], xerr=0.5*dx, fmt='.', ecolor='black', color='black', elinewidth=0.7, capsize=0)
            
            return yratio
        else:
            print("Error: data is not set!")

    #==============================================================================================================
    """
    def set_regions(self, regions_feature, regions_list):
        
        self.regions_feature = regions_feature
        self.regions_list = regions_list
        self.regions_code_list = ["R"+str(i) for i in range(len(self.regions_list))]
    """
    #==============================================================================================================
    """
    def set_channels(self, channels_feature, channels_list):
        
        self.channels_feature = channels_feature
        self.channels_list = channels_list  
        self.channels_code_list = ["C"+str(i) for i in range(len(self.channels_list))]
    """
    #==============================================================================================================
    """
    def add_obs_bins(self, feature, bins):
        
        self.obs_features_list.append(feature)
        self.obs_bins_list.append(bins)
        
        self.obs_bins_code_list = []
        self.imax = 0
        for i in range(len(self.obs_features_list)):
            self.obs_bins_code_list.append(["F"+str(i)+"B"+str(j) for j in range(len(self.obs_bins_list[i]))])
            self.imax += len(self.obs_bins_list[i])
            
        self.obs_bins_code_short_string = ""
        self.obs_bins_code_long_string = ""
        #for bin_code in self.obs_bins_code_list:
        #    self.obs_bins_code_short_string += 
    """
    #==============================================================================================================
    def get_combine_datacard(self, outdir, tag, mode="shape"):
        
        outdir = os.path.join(outdir, tag)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            
            
        if mode == "counting":
            bins_code = [self.regions_labels[self.region]+"B"+str(i) for i in range(len(self.bins)-1)]                   
        elif mode == "shape":
            bins_code = [self.regions_labels[self.region]]
        
        
        self.imax = len(self.bins)-1
        self.jmax = self.number_ds_groups-1
        self.kmax = len(self.sys_IDs)
        
        bin_short_string = '{:<40s}'.format("bin")
        for icode in bins_code:
            bin_short_string = bin_short_string + '{0:<30s}'.format(icode)
        bin_short_string = bin_short_string + "\n"
        
        if mode == "counting":
            obs_string = '{:<40s}'.format("observation")
            for iobs in self.hist_data:
                obs_string = obs_string + '{0:<30s}'.format(str('{:.8f}'.format(iobs)))
            obs_string = obs_string + "\n"                   
        elif mode == "shape":
            obs_string = '{:<40s}'.format("observation")
            obs_string = obs_string + '{0:<30s}'.format(str(-1))
            obs_string = obs_string + "\n"
        
        
        bin_long_string = '{:<40s}'.format("bin")
        for icode in bins_code:
            for j in range(self.jmax+1):
                bin_long_string = bin_long_string + '{0:<30s}'.format(icode)
        bin_long_string = bin_long_string + "\n"
        
        process_string = '{:<40s}'.format("process")
        for i in range(len(bins_code)):
            for iproc in self.processes:
                process_string = process_string + '{0:<30s}'.format(re.sub('[^A-Za-z0-9]+', '', iproc))
        process_string = process_string + "\n"
        
        processID_string = '{:<40s}'.format("process")
        for i in range(len(bins_code)):
            for j in range(len(self.processes)):
                processID_string = processID_string + '{0:<30s}'.format(str(j))
        processID_string = processID_string + "\n"
        
        if mode == "counting":
            rate_string = '{:<40s}'.format("rate")
            for i in range(len(bins_code)):
                for j in range(len(self.processes)):
                    rate_string = rate_string + '{0:<30s}'.format(str('{:.8f}'.format(round(self.hist_table3D[0][0][j][i],8))))
            rate_string = rate_string + "\n"                   
        elif mode == "shape":
            rate_string = '{:<40s}'.format("rate")
            for j in range(len(self.processes)):
                rate_string = rate_string + '{0:<30s}'.format(str(-1))
            rate_string = rate_string + "\n"
        
        #syst_dist_type = "lnN"
        syst_string_list = []
        templates_pred_syst_hist_list = []
        templates_pred_syst_name_list = []
        for k in range(len(self.sys_IDs)):
            
            syst_name = self.sys_labels[k]
            if( syst_name in self.full_year_correlated or syst_name[-3:] == "_fc" ): 
                syst_string = '{:<32s}'.format("CMS_"+self.sys_labels[k])
            else:
                syst_string = '{:<32s}'.format("CMS_"+self.sys_labels[k]+"_"+self.period)
            
            if mode == "counting":
                syst_string = syst_string + '{:<8s}'.format("lnN")
                for i in range(len(bins_code)):
                    for j in range(len(self.processes)):
                        if np.abs(self.sys_unc_table3D[self.sys_IDs[k]][0][j][i]) < 0.000001:
                            self.sys_unc_table3D[self.sys_IDs[k]][0][j][i] = -0.000001 # avoid zeros in syst variations
                        if np.abs(self.sys_unc_table3D[self.sys_IDs[k]][1][j][i]) < 0.000001:
                            self.sys_unc_table3D[self.sys_IDs[k]][1][j][i] = 0.000001 # avoid zeros in syst variations
                        syst_string = syst_string + '{0:<30s}'.format(str('{:.6f}'.format(round(self.sys_unc_table3D[self.sys_IDs[k]][0][j][i]+1,6)))+"/"+str('{:.6f}'.format(round(self.sys_unc_table3D[self.sys_IDs[k]][1][j][i]+1,6))))
                syst_string = syst_string + "\n"
                syst_string_list.append(syst_string)
            
            if mode == "shape":
                if self.sys_labels[k] == "Lumi":

                    syst_string_lumi = []
                    for tagUnc in self.LumiTagsUnc:
                        if tagUnc == "uncorr":
                            syst_string_lumi.append('{:<32s}'.format("CMS_"+self.sys_labels[k]+"_"+self.period) + '{:<8s}'.format("lnN"))
                        else:
                            syst_string_lumi.append('{:<32s}'.format("CMS_"+self.sys_labels[k]+"_"+tagUnc) + '{:<8s}'.format("lnN"))

                    for ilumi in range(len(self.LumiValuesUnc)):
                        for j in range(len(self.processes)):
                            syst_string_lumi[ilumi] = syst_string_lumi[ilumi] + '{0:<30s}'.format(str('{:.6f}'.format(round(1-self.LumiValuesUnc[ilumi],6)))+"/"+str('{:.6f}'.format(round(1+self.LumiValuesUnc[ilumi],6))))
                        syst_string_lumi[ilumi] = syst_string_lumi[ilumi] + "\n"
                        syst_string_list.append(syst_string_lumi[ilumi])
                
                if self.sys_labels[k] != "Lumi" and self.sys_labels[k] != "Stat":
                    
                    if len(self.bins)-1 == 1 and not self.allow_single_bin_template:
                        syst_string = syst_string + '{:<8s}'.format("lnN")
                        for j in range(len(self.processes)):
                            if np.abs(self.sys_unc_table3D[self.sys_IDs[k]][0][j][i]) < 0.000001:
                                self.sys_unc_table3D[self.sys_IDs[k]][0][j][i] = -0.000001 # avoid zeros in syst variations
                            if np.abs(self.sys_unc_table3D[self.sys_IDs[k]][1][j][i]) < 0.000001:
                                self.sys_unc_table3D[self.sys_IDs[k]][1][j][i] = 0.000001 # avoid zeros in syst variations
                            syst_string = syst_string + '{0:<30s}'.format(str('{:.6f}'.format(round(self.sys_unc_table3D[self.sys_IDs[k]][0][j][0]+1,6)))+"/"+str('{:.6f}'.format(round(self.sys_unc_table3D[self.sys_IDs[k]][1][j][0]+1,6))))   
                        syst_string = syst_string + "\n"
                        syst_string_list.append(syst_string)
                    else:
                        syst_string = syst_string + '{:<8s}'.format("shape")
                        for i in range(len(bins_code)):
                            for j in range(len(self.processes)):
                                syst_string = syst_string + '{0:<30s}'.format(str(1.0))
                        syst_string = syst_string + "\n"
                        syst_string_list.append(syst_string)
                    
                        for j in range(len(self.processes)):
                            syst_name = self.sys_labels[k]
                            
                            templates_pred_syst_hist_list.append((self.sys_unc_table3D[self.sys_IDs[k]][0][j]+1)*self.hist_table3D[0][0][j]) #down
                            if( syst_name in self.full_year_correlated or syst_name[-3:] == "_fc" ):
                                templates_pred_syst_name_list.append(self.processes[j] + "_" + "CMS_"+self.sys_labels[k] + "Down")
                            else:
                                templates_pred_syst_name_list.append(self.processes[j] + "_" + "CMS_"+self.sys_labels[k] + "_"+self.period + "Down")
                            
                            templates_pred_syst_hist_list.append((self.sys_unc_table3D[self.sys_IDs[k]][1][j]+1)*self.hist_table3D[0][0][j]) #up
                            if( syst_name in self.full_year_correlated or syst_name[-3:] == "_fc" ):
                                templates_pred_syst_name_list.append(self.processes[j] + "_" + "CMS_"+self.sys_labels[k] + "Up")
                            else:
                                templates_pred_syst_name_list.append(self.processes[j] + "_" + "CMS_"+self.sys_labels[k] + "_"+self.period + "Up")
                        
        
        file_name = os.path.join(outdir, "datacard_combine_" + self.regions_labels[self.region] + "_" + mode + "_" + re.sub('[^A-Za-z0-9]+', '', self.signal_name) + ".txt")
        datacard = open(file_name, "w") 
                
        if self.analysis_name is None:
            datacard.write("# Datacard for analysis\n")
        else:
            datacard.write("# Datacard for " + self.analysis_name + " analysis\n")
        datacard.write("# Signal sample: " + re.sub('[^A-Za-z0-9]+', '', self.signal_name) + "\n")
        if mode == "counting":
            datacard.write("# Mode: simple counting\n")
        elif mode == "shape":
            datacard.write("# Mode: binned shape\n")
        #datacard.write("# Regions: " + str(self.regions_list) + "  <-->  " + str(self.regions_code_list))
        #datacard.write("# Channels: " + str(self.channels_list) + "  <-->  " + str(self.channels_code_list))
        #datacard.write("# Features & bins:")
        #for i in range(len(self.obs_features_list)):
        #    datacard.write("# " + self.obs_features_list[i] + " = " + str(self.obs_bins_list) + "  <-->  " + str(self.obs_bins_code_list))
        datacard.write("#=====================================================================================\n")
        if mode == "counting":
            datacard.write("imax " + str(self.imax) + "\tnumber of channels\n")                   
        elif mode == "shape":    
            datacard.write("imax 1\tnumber of channels\n")                   #Modify
            
        datacard.write("jmax " + str(self.jmax) + "\tnumber of backgrounds\n")
        datacard.write("kmax *" + "\tnumber of nuisance parameters\n")  # str(self.kmax)
        datacard.write("--------------------------------------------------------------------------------------\n")
        if mode == "shape":    
            shapes_file_name = os.path.join("datacard_combine_" + self.regions_labels[self.region] + "_" + mode + "_" + re.sub('[^A-Za-z0-9]+', '', self.signal_name) + ".root")
            datacard.write("shapes * " + self.regions_labels[self.region] + " " + shapes_file_name + " $PROCESS $PROCESS_$SYSTEMATIC\n")
            
        datacard.write("--------------------------------------------------------------------------------------\n")
        datacard.write(bin_short_string)
        datacard.write(obs_string)                                                  #Future - Include regions and channes in the names
        datacard.write("--------------------------------------------------------------------------------------\n")
        datacard.write(bin_long_string)
        datacard.write(process_string)
        datacard.write(processID_string)
        datacard.write(rate_string)
        datacard.write("--------------------------------------------------------------------------------------\n")
        for syst_string in syst_string_list:
            datacard.write(syst_string)
            
        if mode == "shape":
            for iregion in range(len(self.regions_labels)):
                if iregion > 0:
                    #datacard.write(self.regions_labels[iregion][:-2] + "_norm_" + self.period + " rateParam * " + self.regions_labels[iregion][:-2] + " 1 [0.1,10]\n")
                    datacard.write(self.regions_labels[iregion][:-2] + "_norm rateParam * " + self.regions_labels[iregion][:-2] + " 1 [0.1,10]\n")
            datacard.write("* autoMCStats 100 0 1\n")
                
        datacard.close()
        
        if mode == "shape":
            return file_name, templates_pred_syst_hist_list, templates_pred_syst_name_list, self.hist_table3D[0][0], self.unc_table3D[0][0], self.processes, self.hist_data, np.array(self.bins)*1.
        elif mode == "counting":
            return file_name
