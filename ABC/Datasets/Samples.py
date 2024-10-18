import os


def get_samples( basedir, period ):

    list_basedir = os.listdir(basedir)
    periodTag = period + "_files"

    samples = {
        'Signal_400_100':           [i for i in list_basedir if 'Signal_400_100' == i.split("_"+periodTag)[0]],
        'Signal_400_200':           [i for i in list_basedir if 'Signal_400_200' == i.split("_"+periodTag)[0]],
        'Signal_500_100':           [i for i in list_basedir if 'Signal_500_100' == i.split("_"+periodTag)[0]],
        'Signal_500_200':           [i for i in list_basedir if 'Signal_500_200' == i.split("_"+periodTag)[0]],
        'DYJetsToLL_PtZ-0To50':     [i for i in list_basedir if 'DYJetsToLL_PtZ-0To50' == i.split("_"+periodTag)[0]],
        'DYJetsToLL_PtZ-3To50':     [i for i in list_basedir if 'DYJetsToLL_PtZ-3To50' == i.split("_"+periodTag)[0]],
        'DYJetsToLL_PtZ-50To100':   [i for i in list_basedir if 'DYJetsToLL_PtZ-50To100' == i.split("_"+periodTag)[0]],
        'DYJetsToLL_PtZ-100To250':  [i for i in list_basedir if 'DYJetsToLL_PtZ-100To250' == i.split("_"+periodTag)[0]],
        'DYJetsToLL_PtZ-250To400':  [i for i in list_basedir if 'DYJetsToLL_PtZ-250To400' == i.split("_"+periodTag)[0]],
        'DYJetsToLL_PtZ-400To650':  [i for i in list_basedir if 'DYJetsToLL_PtZ-400To650' == i.split("_"+periodTag)[0]],
        'DYJetsToLL_PtZ-650ToInf':  [i for i in list_basedir if 'DYJetsToLL_PtZ-650ToInf' == i.split("_"+periodTag)[0]],
        'TTTo2L2Nu':                [i for i in list_basedir if 'TTTo2L2Nu' == i.split("_"+periodTag)[0]],
        'Data_A':                   [i for i in list_basedir if 'Data' in i and '_A_' in i and periodTag in i],
        'Data_B':                   [i for i in list_basedir if 'Data' in i and '_B_' in i and periodTag in i],
        'Data_C':                   [i for i in list_basedir if 'Data' in i and '_C_' in i and periodTag in i],
        'Data_D':                   [i for i in list_basedir if 'Data' in i and '_D_' in i and periodTag in i],
        'Data_E':                   [i for i in list_basedir if 'Data' in i and '_E_' in i and periodTag in i],
        'Data_F':                   [i for i in list_basedir if 'Data' in i and '_F_' in i and periodTag in i],
        'Data_G':                   [i for i in list_basedir if 'Data' in i and '_G_' in i and periodTag in i],
        'Data_H':                   [i for i in list_basedir if 'Data' in i and '_H_' in i and periodTag in i],
    }
    
    empty_samples = []
    for sample in samples:
        if len(samples[sample]) == 0:
            empty_samples.append(sample)

    for sample in empty_samples:
        del samples[sample]

    return samples


