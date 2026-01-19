import concurrent.futures
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as pat
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm 
import statsmodels.stats.proportion as prop


from .statistic import (
    #pdf_efficiency,
    get_interval
)


#======================================================================================================================
def stacked_plot( ax, var, dataframes, labels, colors, weight=None, bins=np.linspace(0,100,5), plot=True ):
    """
    Produce stacked plot

    Args:
        var (str): Variable column name to be accessed in pd.DataFrame object.
        dataframes (list): List of pd.DataFrame objects to retrieve var data.
        labels (list): List of labels to be used for each pd.DataFrame var data in dataframes argument.
        colors (list): List of colors to be used for each pd.DataFrame var data in dataframes argument.
        weight (str, optional): Weight column name to be accessed in pd.DataFrame object. Defaults to None.
        bins (numpy.ndarray, optional): Array of bins to be used in plot. Defaults to [0, 25, 50, 75, 100].
    """
    y = []
    w = []

    for df in dataframes:
        y.append(df[var])
        if weight is not None:
            w.append(df[weight])

    if len(w) == 0:
        w = None

    if plot:
        out_hists = plt.hist(
            y, 
            bins=bins, 
            histtype='stepfilled', 
            stacked=True, 
            color=colors, 
            label=labels, 
            linewidth=0, 
            weights=w
        )
      
    
    counts = np.zeros((len(y),len(bins)-1))
    ybkg = np.zeros(len(bins)-1)
    counts2 = np.zeros((len(y),len(bins)-1))
    yerror = np.zeros(len(bins)-1)
    for i in range(len(y)):
        counts[i], bins = np.histogram(y[i], bins, weights=w[i])
        counts2[i], bins = np.histogram(y[i], bins, weights=np.array(w[i])*np.array(w[i]))
    for b in range(len(bins)-1):
        for i in range(len(y)):
            if counts[i,b] > 0:
                ybkg[b] += counts[i,b]
                yerror[b] += counts2[i,b]
    yerror = np.sqrt(yerror)


    yl = ybkg - yerror
    yh = ybkg + yerror
    x = np.array(bins)
    dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
    x = x[:-1]
    dy = yh - yl
    pats = [ pat.Rectangle( (x[i], yl[i]), dx[i], dy[i], hatch='/////', fill=False, linewidth=0, edgecolor='grey' ) for i in range(len(x)-1) ]
    pats.append(pat.Rectangle( (x[len(x)-1], yl[len(x)-1]), dx[len(x)-1], dy[len(x)-1], hatch='/////', fill=False, linewidth=0, edgecolor='grey', label="Stat. Unc." ))
    if plot:
        for p in pats:
            ax.add_patch(p) 

    return ybkg, yerror
    
    
    
#======================================================================================================================    
def step_plot( ax, var, dataframe, label, color='black', weight=None, error=False, normalize=False, bins=np.linspace(0,100,5), linestyle='solid', overflow=False, underflow=False, plot=True, linewidth=1.5 ):
    """
    Produce signal plot

    Args:
        var (str): Variable column name to be accessed in pd.DataFrame object.
        dataframes (pd.DataFrame): DataFrame to retrieve var's data and weights.
        param (list, optional): Heavy Higgs and a scalar boson mass parameters to be used in label. Defaults to [1000,100].
        color (str, optional): Color line. Defaults to 'black'.
        weight (str, optional): Weight column name to be accessed in pd.DataFrame object. Defaults to None.
        bins (numpy.ndarray, optional): Array of bins to be used in plot. Defaults to [0, 25, 50, 75, 100].
    """

    if weight is None:
        W = None
        W2 = None
    else:
        W = dataframe[weight]
        W2 = dataframe[weight]*dataframe[weight]

    eff_bins = bins[:]
    if overflow:
        eff_bins[-1] = np.inf
    if underflow:
        eff_bins[0] = -np.inf
    
    counts, binsW = np.histogram(
        dataframe[var], 
        bins=eff_bins, 
        weights=W
    )
    yMC = np.array(counts)
    
    countsW2, binsW2 = np.histogram(
        dataframe[var], 
        bins=eff_bins, 
        weights=W2
    )
    errMC = np.sqrt(np.array(countsW2))
    
    if normalize:
        if weight is None:
            norm_factor = len(dataframe[var])
        else:
            norm_factor = dataframe[weight].sum()
        yMC = yMC/norm_factor
        errMC = errMC/norm_factor
    
    ext_yMC = np.append([yMC[0]], yMC)
    
    if plot:
        plt.step(bins, ext_yMC, color=color, label=label, linewidth=linewidth, linestyle=linestyle)
        
        if error:
            x = np.array(bins)
            dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
            x = x[:-1]
            
            ax.errorbar(
                x+0.5*dx, 
                yMC, 
                yerr=[errMC, errMC], 
                fmt=',',
                color=color, 
                elinewidth=1
            ) 
    
    return yMC, errMC
    
    
#======================================================================================================================    
def data_plot( ax, var, dataframe, bins=np.linspace(0,100,5), label="Data", color="black", normalize=False, overflow=False, underflow=False, plot=True ):
    x = np.array(bins)
    dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
    x = x[:-1]
    
    eff_bins = bins[:]
    if overflow:
        eff_bins[-1] = np.inf
    if underflow:
        eff_bins[0] = -np.inf
    
    counts, binsW = np.histogram(dataframe[var], eff_bins)
    ydata = np.array(counts)
    errdata = np.sqrt(ydata)
    
    if normalize:
        norm_factor = len(dataframe[var])
        ydata = ydata/norm_factor
        errdata = errdata/norm_factor
    
    if plot:
        ax.errorbar(
            x+0.5*dx, 
            ydata, 
            yerr=[errdata, errdata], 
            xerr=0.5*dx, 
            fmt='.', 
            ecolor=color, 
            label=label, 
            color=color, 
            elinewidth=0.7, 
            capsize=0
        )   
    
    return ydata, errdata


#======================================================================================================================
def ratio_plot( ax, ynum, errnum, yden, errden, bins=np.linspace(0,100,5), color='black', numerator="data" ):
    x = np.array(bins)
    dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
    x = x[:-1]
    yratio = np.zeros(ynum.size)
    yeratio = np.zeros(ynum.size)
    y2ratio = np.zeros(ynum.size)
    ye2ratio = np.zeros(ynum.size)
    for i in range(ynum.size):
        if yden[i] == 0:
            yratio[i] = 99999
            yeratio[i] = 0
            y2ratio[i] = 1
            ye2ratio[i] = 0
        else:
            yratio[i] = ynum[i]/yden[i]
            yeratio[i] = errnum[i]/yden[i]
            y2ratio[i] = yden[i]/yden[i]
            ye2ratio[i] = errden[i]/yden[i]
            
    if numerator == "data":
        yl = (yden - errden)/yden
        yh = (yden + errden)/yden
        dy = yh - yl
        pats = [ pat.Rectangle( (x[i], yl[i]), dx[i], dy[i], hatch='/////', fill=False, linewidth=0, edgecolor='grey' ) for i in range(len(x)-1) ]
        pats.append(pat.Rectangle( (x[len(x)-1], yl[len(x)-1]), dx[len(x)-1], dy[len(x)-1], hatch='/////', fill=False, linewidth=0, edgecolor='grey' ))
        for p in pats:
            ax.add_patch(p) 
    
        ax.axhline(1, color='grey', linestyle='-', linewidth=0.5)
    
        ax.errorbar(x+0.5*dx, yratio, yerr=[yeratio, yeratio], xerr=0.5*dx, fmt='.', ecolor='black', color='black', elinewidth=0.7, capsize=0)
    elif numerator == "mc":
        ax.errorbar(x+0.5*dx, y2ratio, yerr=[ye2ratio, ye2ratio], xerr=0.5*dx, fmt=',', ecolor="grey", color="grey", elinewidth=1.2, capsize=0)
    
        ax.errorbar(x+0.5*dx, yratio, yerr=[yeratio, yeratio], xerr=0.5*dx, fmt=',', ecolor=color, color=color, elinewidth=1.2, capsize=0)
    
    return yratio
  


    
#======================================================================================================================
"""
def __bayesian(y_before, y_after, yratio):

    if y_before == 0:
        return 0, 0
    
    if y_before > 5000:
        x_grid = np.linspace(0,1,50001)
    elif y_before > 1000:
        x_grid = np.linspace(0,1,20001)
    elif y_before > 500:
        x_grid = np.linspace(0,1,5001)
    elif y_before > 100:
        x_grid = np.linspace(0,1,2001)
    elif y_before <= 100:
        x_grid = np.linspace(0,1,501)
    
    y_below, y_above = get_interval( x_grid, pdf_efficiency( x_grid, y_after, y_before ))
    ye_below = yratio - y_below
    ye_above = y_above - yratio

    return ye_below, ye_above
"""

def efficiency_plot( ax, var, dataframe, bit, label, color='black', bins=np.linspace(0,100,5), histograms=False, y2label="Events", uncertainty="clopper", multiprocess=True, overflow=False, underflow=False, weight=None, plot=True ):
    
    ax.set_ylim([0,1.05])
    plt.axhline(1, color='grey', linewidth=1, linestyle="dotted")
    
    if histograms:
        ax2 = ax.twinx()
        ax2.set_ylabel(y2label, color='royalblue', size=14, horizontalalignment='right', y=1.0)
        ax2.tick_params('y', colors='royalblue')
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.tick_params(which='major', length=8)
        ax2.tick_params(which='minor', length=4)
        ax2.margins(x=0)
        ax.set_zorder(10)
        ax.patch.set_visible(False)
       
    
    dataframe_selected = dataframe[dataframe[bit] == 1]
    dataframe_not_selected = dataframe[dataframe[bit] == 0]
    
    # overflow and underflow are not working with histograms=True
    eff_bins = bins[:]
    if overflow:
        eff_bins[-1] = np.inf
    if underflow:
        eff_bins[0] = -np.inf
    
    if weight is None:
        if histograms:
            y_before, bins_not_used, nothing = ax2.hist( dataframe[var], bins=eff_bins, histtype='step', color='royalblue', linewidth=1 )
            y_after, bins_not_use, nothing = ax2.hist( dataframe_selected[var], bins=eff_bins, histtype='stepfilled', color='aqua', linewidth=0 )
        else:
            y_before, bins_not_use = np.histogram( dataframe[var], eff_bins )
            y_after, bins_not_use = np.histogram( dataframe_selected[var], eff_bins )    
    else:
        if histograms:
            y_before, bins_not_use, nothing = ax2.hist( dataframe[var], bins=eff_bins, histtype='step', color='royalblue', linewidth=1, weights=dataframe[weight] )
            y_after, bins_not_use, nothing = ax2.hist( dataframe_selected[var], bins=eff_bins, histtype='stepfilled', color='aqua', linewidth=0, weights=dataframe_selected[weight] )
        else:
            y_before, bins_not_use = np.histogram( dataframe[var], eff_bins, weights=dataframe[weight] )
            y_after, bins_not_use = np.histogram( dataframe_selected[var], eff_bins, weights=dataframe_selected[weight] ) 
        y_not_after, bins_not_use = np.histogram( dataframe_not_selected[var], eff_bins, weights=dataframe_not_selected[weight] ) 
        y2_not_after, bins_not_use = np.histogram( dataframe_not_selected[var], eff_bins, weights=dataframe_not_selected[weight]*dataframe_not_selected[weight] )
        y2_before, bins_not_use = np.histogram( dataframe[var], eff_bins, weights=dataframe[weight]*dataframe[weight] )
        y2_after, bins_not_use = np.histogram( dataframe_selected[var], eff_bins, weights=dataframe_selected[weight]*dataframe_selected[weight] )
            
    
    yratio = np.zeros(y_after.size)
    yeratio_binomial = np.zeros(y_after.size)
    for i in range(y_after.size):
        if y_before[i] == 0:
            yratio[i] = 99999
        else:
            yratio[i] = y_after[i]/y_before[i]
            if(yratio[i] > 1):
                yratio[i] = 1
            yeratio_binomial[i] = np.sqrt((y_after[i]/y_before[i])*(1-y_after[i]/y_before[i])*(1/y_before[i])) # binomial uncertainty
    
    if weight is None:
        if uncertainty == "binomial":
            ye_below = yeratio_binomial
            ye_above = yeratio_binomial
        
        # https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html
        elif uncertainty == "clopper":
            y_below, y_above = prop.proportion_confint(y_after, y_before, alpha=0.32, method='beta')
            ye_below = yratio - y_below
            ye_above = y_above - yratio
            
        elif uncertainty == "wilson":
            y_below, y_above = prop.proportion_confint(y_after, y_before, alpha=0.32, method='wilson') 
            ye_below = yratio - y_below
            ye_above = y_above - yratio
            
        elif uncertainty == "jeffreys":
            y_below, y_above = prop.proportion_confint(y_after, y_before, alpha=0.32, method='jeffreys')
            ye_below = yratio - y_below
            ye_above = y_above - yratio
            for i in range(yratio.size):
                if y_before[i] == 0:
                    ye_below[i] = 0
                    ye_above[i] = 0
        
        """
        elif uncertainty == "bayesian":

            if multiprocess is True:

                cpu_count = multiprocessing.cpu_count()
                if cpu_count <= 2:
                    cpu_count = 1
                else:
                    cpu_count -= 2

                with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
                    jobs = list(tqdm(executor.map(__bayesian, y_before, y_after, yratio), total=len(y_before)))
                    jobs_result = [j for j in jobs]

                ye_below = np.array([job[0] for job in jobs_result])
                ye_above = np.array([job[1] for job in jobs_result])

            else:
            
                y_below = np.zeros(len(y_before))
                y_above = np.zeros(len(y_before))
                ye_below = np.zeros(len(y_before))
                ye_above = np.zeros(len(y_before))
                        
                for i in tqdm(range(len(y_before))):
                    if y_before[i] == 0:
                        ye_below[i] = 0
                        ye_above[i] = 0
                        continue
                    
                    if y_before[i] > 5000:
                        x_grid = np.linspace(0,1,50001)
                    elif y_before[i] > 1000:
                        x_grid = np.linspace(0,1,20001)
                    elif y_before[i] > 500:
                        x_grid = np.linspace(0,1,5001)
                    elif y_before[i] > 100:
                        x_grid = np.linspace(0,1,2001)
                    elif y_before[i] <= 100:
                        x_grid = np.linspace(0,1,501)
                    
                    y_below[i], y_above[i] = get_interval(x_grid, pdf_efficiency( x_grid, y_after[i], y_before[i] ))
                
                    ye_below[i] = yratio[i] - y_below[i]
                    ye_above[i] = y_above[i] - yratio[i]
        """        
    
    else:
        print("You are using weighted events, the uncertainties are calculated using the gaussian approximation!")
        err_after = np.sqrt(y2_after)
        err_not_after = np.sqrt(y2_not_after)
        
        r = y_not_after/y_after
        err_r = r*np.sqrt( (err_after/y_after)**2 + (err_not_after/y_not_after)**2 )
        err_ratio = err_r/(1+r)**2
        
        N_eff_after = y_after**2/y2_after
        N_eff_not_after = y_not_after**2/y2_not_after
        ye_below = err_ratio.copy() # 1 sigma
        ye_above = err_ratio.copy() # 1 sigma
        good_approximation = True
        for i in range(err_ratio.size):
            if yratio[i]+ye_above[i] > 1:
                ye_above[i] = 1 - yratio[i]
            if yratio[i]-ye_below[i] < 0:
                ye_below[i] = yratio[i]
            if N_eff_after[i] <= 25 or N_eff_not_after[i] <= 25:
                good_approximation = False
        if good_approximation:
            print("GAUSSIAN APPROXIMATION IS VALID!")
        else:
            print("GAUSSIAN APPROXIMATION IS NOT VALID!")
            print("Check if N_eff > 25 for selected events:")
            print(N_eff_after)
            print("Check if N_eff > 25 for not selected events:")
            print(N_eff_not_after)
        
        
    if plot:
        x = np.array(bins)
        dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
        x = x[:-1]

        ax.errorbar(x+0.5*dx, yratio, yerr=[ye_below, ye_above], xerr=0.5*dx, fmt='.', ecolor=color, color=color, elinewidth=0.7, capsize=0, label=label)
    
    return yratio, ye_below, ye_above    
    
def limits_plot( ax, parameter, expected_limits, observed_limits=None ):

    if observed_limits is not None:
        plt.plot(parameter, observed_limits, color='black', label=r'$\mathrm{Observed}$', linewidth=1.5, marker='.')

    ax.fill_between(parameter, expected_limits[0], expected_limits[4], facecolor='gold', linewidth=0, label='$\pm 2\ \mathrm{std.\ deviation}$')
    ax.fill_between(parameter, expected_limits[1], expected_limits[3], facecolor='limegreen', linewidth=0, label='$\pm 1\ \mathrm{std.\ deviation}$')
    plt.plot(parameter, expected_limits[2], color='blue', label=r'$\mathrm{Asymptotic\ CL}_\mathrm{s}\ \mathrm{expected}$', linewidth=1.5, linestyle='dashdot')
    
    
   
    
    
    
    
    
    
    
    
