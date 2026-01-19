import os

from src.grouper import combine_jobs

def test_checker():
    basedir = "/mnt/CERNBox/Projects/Mestrado/HHDM/BtaggerComparision"
    os_basedir = os.listdir(basedir)
    TreeName = 'selection'
    period = '16'
    samples = {
        'Signal_400_100':  [i for i in os_basedir if 'Signal_400_100' in i], 
        'Signal_600_150':  [i for i in os_basedir if 'Signal_600_150' in i],
        'Signal_1000_800':  [i for i in os_basedir if 'Signal_1000_800' in i],
        'Signal_1000_100':  [i for i in os_basedir if 'Signal_1000_100' in i],
        'DYJetsToLL':  [i for i in os_basedir if 'DYJetsToLL' in i], 
        'DYJetsToTauTau':  [i for i in os_basedir if 'DYJetsToTauTau' in i],
        'ZZ':  [i for i in os_basedir if 'ZZ' == i.split("_")[0]],
        'WW':  [i for i in os_basedir if 'WW' == i.split("_")[0]],
        'WZ':  [i for i in os_basedir if 'WZ' == i.split("_")[0]], 
        'TTTo2L2Nu':  [i for i in os_basedir if 'TTTo2L2Nu' in i],
        'TTToSemiLeptonic':  [i for i in os_basedir if 'TTToSemiLeptonic' in i],
        'ST_tW_top':  [i for i in os_basedir if 'ST_tW_top' in i],
        'ST_tW_antitop':  [i for i in os_basedir if 'ST_tW_antitop' in i], 
        'ZZZ':  [i for i in os_basedir if 'ZZZ' in i],
        'WWZ':  [i for i in os_basedir if 'WWZ' in i],
        'ZGToLLG':  [i for i in os_basedir if 'ZGToLLG' in i],
        'WGToLNuG':  [i for i in os_basedir if 'WGToLNuG' in i],
        'TTZToQQ':  [i for i in os_basedir if 'TTZToQQ' in i],
        'TWZToLL_thad_Wlept':  [i for i in os_basedir if 'TWZToLL_thad_Wlept' in i], 
        'TWZToLL_tlept_Whad':  [i for i in os_basedir if 'TWZToLL_tlept_Whad' in i],
        'TWZToLL_tlept_Wlept':  [i for i in os_basedir if 'TWZToLL_tlept_Wlept' in i],
        #'WJetsToLNu':  [i for i in os.listdir("./") if 'WJetsToLNu' in i],
    }

    combine_jobs(basedir, period, TreeName, samples)
    files = os.listdir(os.path.join(basedir, "datasets/16"))
    for dataset in files:
        assert dataset.split(".")[0] in samples.keys()