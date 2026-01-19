import os
import pandas as pd
import numpy as np

def join_datasets(ds, new_name, input_list, mode="normal", delete_inputs=True):
    
    datasets_list = []
    for input_name in input_list:
        datasets_list.append(ds[input_name])

    good_list = False
    if mode == "normal":
        ds[new_name] = pd.concat(datasets_list).reset_index(drop=True)
        good_list = True
    elif mode == "syst":
        ds[new_name] = datasets_list
        good_list = True
    elif mode == "scalars" or mode == "vectors":
        ds[new_name] = {}
        first = True
        for dataset in datasets_list:
            if first:
                for variable in dataset.keys():
                    ds[new_name][variable] = dataset[variable].copy()
            else:
                for variable in dataset.keys():
                    if mode == "vectors":
                        out_size = len(ds[new_name][variable][0])
                        dataset_size = len(dataset[variable][0])
                        diff_size = abs(out_size-dataset_size)
                        if out_size > dataset_size:
                            number_of_events = len(dataset[variable])
                            for i in range(diff_size):
                                dataset[variable] = np.c_[ dataset[variable], np.zeros(number_of_events) ]
                        elif dataset_size > out_size:
                            number_of_events = len(ds[new_name][variable])
                            for i in range(diff_size):
                                ds[new_name][variable] = np.c_[ ds[new_name][variable], np.zeros(number_of_events) ]
                    ds[new_name][variable] = np.concatenate((ds[new_name][variable],dataset[variable]))
            first = False
        good_list = True
        
        
    
        
        
    else:
        print("Type of the items is not supported!")
    
    if delete_inputs:
        if good_list:
            for input_name in input_list:
                del ds[input_name]
    
    del datasets_list
        
      
        
        
