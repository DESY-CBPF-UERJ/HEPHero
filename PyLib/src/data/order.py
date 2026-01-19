import os
import pandas as pd
import numpy as np

def order_datasets(dataframes, labels, colors):
    
    """
    colors = [] 
    labels = [] 
    dataframes = []
    for i in range(len(dataframes_in)):
        if len( dataframes_in[i] ) > 0 :
            colors.append( colors_in[i] )
            labels.append( labels_in[i] )
            dataframes.append( dataframes_in[i] )
        else:
            print(labels_in[i] + " is empty!!!")
    """
    

    color_0 = colors[0]
    label_0 = labels[0]
    df_0 = dataframes[0]
    size_0 = dataframes[0].evtWeight.sum()

    colors.pop(0)
    labels.pop(0)
    dataframes.pop(0)
    dataframes = [ df_i.reset_index() for df_i in dataframes ]

    sizes_0 = [ df_i.evtWeight.sum() for df_i in dataframes ]
    sizes_neg = [-1.0 if x==0.0 else x for x in sizes_0]
    
    sizes_neg, labels, colors, dataframes = (list(t) for t in zip(*sorted(zip(sizes_neg, labels, colors, dataframes))))
    
    
    
    colors.insert(0, color_0)
    labels.insert(0, label_0)
    dataframes.insert(0, df_0)
    sizes_neg.insert(0, size_0)
    sizes = [0.0 if x==-1.0 else x for x in sizes_neg]

    ds_lists = []
    total_size = 0
    for i in range(len(dataframes)):
        ds_lists.append({ "Datasets": labels[i], "Number of events": sizes[i] })
        total_size += sizes[i]
    ds_lists = pd.DataFrame(ds_lists)
    print(ds_lists)
    print("Purity:", sizes[-1]/total_size)

    return dataframes, labels, colors, sizes
        
