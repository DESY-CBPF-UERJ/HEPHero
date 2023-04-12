nano_version = 'v9'
pathAPV16 = 'Datasets/Files/signal_16/UL_APV'+nano_version+'/'
path16 = 'Datasets/Files/signal_16/UL_'+nano_version+'/'
path17 = 'Datasets/Files/signal_17/UL_'+nano_version+'/'
path18 = 'Datasets/Files/signal_18/UL_'+nano_version+'/'

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
        
    Signal = [
        ["Signal_400_100_"+period]           + [year+'0100'+APVID, path+"Signal_400_100.txt"],
        ["Signal_400_200_"+period]           + [year+'0101'+APVID, path+"Signal_400_200.txt"],
        ["Signal_500_100_"+period]           + [year+'0102'+APVID, path+"Signal_500_100.txt"],
        ["Signal_500_200_"+period]           + [year+'0103'+APVID, path+"Signal_500_200.txt"],
        ["Signal_500_300_"+period]           + [year+'0104'+APVID, path+"Signal_500_300.txt"],
        ["Signal_600_100_"+period]           + [year+'0105'+APVID, path+"Signal_600_100.txt"],
        ["Signal_600_200_"+period]           + [year+'0106'+APVID, path+"Signal_600_200.txt"],
        ["Signal_600_300_"+period]           + [year+'0107'+APVID, path+"Signal_600_300.txt"],
        ["Signal_600_400_"+period]           + [year+'0108'+APVID, path+"Signal_600_400.txt"],
        ["Signal_800_100_"+period]           + [year+'0109'+APVID, path+"Signal_800_100.txt"],
        ["Signal_800_200_"+period]           + [year+'0110'+APVID, path+"Signal_800_200.txt"],
        ["Signal_800_300_"+period]           + [year+'0111'+APVID, path+"Signal_800_300.txt"],
        ["Signal_800_400_"+period]           + [year+'0112'+APVID, path+"Signal_800_400.txt"],
        ["Signal_800_600_"+period]           + [year+'0113'+APVID, path+"Signal_800_600.txt"],
        ["Signal_1000_100_"+period]          + [year+'0114'+APVID, path+"Signal_1000_100.txt"],
        ["Signal_1000_200_"+period]          + [year+'0115'+APVID, path+"Signal_1000_200.txt"],
        ["Signal_1000_300_"+period]          + [year+'0116'+APVID, path+"Signal_1000_300.txt"],
        ["Signal_1000_400_"+period]          + [year+'0117'+APVID, path+"Signal_1000_400.txt"],
        ["Signal_1000_600_"+period]          + [year+'0118'+APVID, path+"Signal_1000_600.txt"],
        ["Signal_1000_800_"+period]          + [year+'0119'+APVID, path+"Signal_1000_800.txt"],
    ]    
        
        
    if period == "APV_16":
        Signal_preVFP_16 = Signal
    elif period == "16":
        Signal_postVFP_16 = Signal
    elif period == "17":
        Signal_17 = Signal
    elif period == "18":
        Signal_18 = Signal
