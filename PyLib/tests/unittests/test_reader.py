import os

from src.reader import read_datasets

def test_reader():
    basedir = "/mnt/CERNBox/Projects/Mestrado/HHDM/BtaggerComparision"
    period = '16'
    datasets_name = [
        'Signal_400_100',
        'Signal_600_150',
        'Signal_1000_800',
        'Signal_1000_100',
        'DYJetsToLL',
        'DYJetsToTauTau',
        'ZZ',
        'WW',
        'WZ',
        'TTTo2L2Nu',
        'TTToSemiLeptonic',
        'ST_tW_top',
        'ST_tW_antitop',
        'ZZZ',
        'WWZ',
        'ZGToLLG',
        'WGToLNuG',
        'TTZToQQ',
        'TWZToLL_thad_Wlept',
        'TWZToLL_tlept_Whad',
        'TWZToLL_tlept_Wlept',
    ]

    datasets = read_datasets(basedir, period)
    assert set(list(datasets.keys())) == set(datasets_name)