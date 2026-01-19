import numpy as np
import matplotlib.pyplot as plt
from .statistic import correlation
from sklearn.metrics import confusion_matrix

#======================================================================================================================
def correlation_matrix_plot(ax, df, variables, title="Correlation Matrix", title_size=22, text_size=18, var_names=None, weight=None):
    # Produce the covariance matrix between the list of variables "variables" of the dataframe "df"
        
    corr_matrix = []
    for i in range(len(variables)):
        cov_line = []
        for j in range(len(variables)):
            if weight is None:
                corr_xy = correlation( df[variables[i]], df[variables[j]] )
            else:
                corr_xy = correlation( df[variables[i]], df[variables[j]], weight=df[weight] )
            cov_line.append(corr_xy)
        corr_matrix.append(cov_line)   
    corr_matrix = np.around(corr_matrix,2)
    
    im = ax.imshow(corr_matrix, cmap="RdBu", vmin=-1.2, vmax=1.2, interpolation='none')
    plt.title(title, size=title_size)
    ax.set_xticks(np.arange(len(variables)))
    ax.set_yticks(np.arange(len(variables)))
    if var_names is None:
        var_names = variables
    ax.set_yticklabels(var_names, size=text_size)
    ax.set_xticklabels(var_names, size=text_size)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor" )
    plt.minorticks_off()
    for i in range(len(variables)):
        for j in range(len(variables)):
            text = ax.text(j, i, corr_matrix[i,j], ha="center", va="center", color="black", size=text_size)
    return corr_matrix


def confusion_matrix_plot(ax, y_true, y_pred, weights, classes, normalize='row', cmap=plt.cm.Blues):
    
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    
    if normalize == 'row':
        title = 'Confusion matrix - row normalized'
    if normalize == 'column':
        title = 'Confusion matrix - column normalized'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, sample_weight=weights)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    
    if normalize == 'row':
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if normalize == 'column':
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    
    # Plot normalized confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    #ax.figure.colorbar(im, ax=ax, pad=0)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, title=title, ylabel='True label', xlabel='Predicted label')
    plt.minorticks_off()

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.xlim(-0.5, len(np.unique(y_true))-0.5)
    plt.ylim(len(np.unique(y_true))-0.5, -0.5)

    np.set_printoptions(precision=2)

    return cm
