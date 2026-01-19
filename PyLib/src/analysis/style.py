import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.ticker import AutoMinorLocator
#from matplotlib import rc
#rc('text', usetex=True)

#======================================================================================================================
def start():
    """
    Define initial setup to be used in the plots and terminal outputs
    """
    pd.set_option('display.expand_frame_repr', False)
    #https://cms-analysis.docs.cern.ch/guidelines/plotting/
    #plt.style.use(hep.style.CMS)  # Define CMS style
    hep.style.use("CMS")
    matplotlib.rcParams['xtick.labelsize'] = 14
    matplotlib.rcParams['ytick.labelsize'] = 14



    #matplotlib.rcParams['font.family'] = 'sans-serif'
    #matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams["text.latex.preamble"]  = r"\usepackage{cmbright}"

#======================================================================================================================
def position(gs1, grid, main, sub):
    """
    Auxiliar function to compute the position of the subplot

    Args:
        gs1 (matplotlib.pyplot.GridSpec): GridSpec object from matplotlib
        grid (list): List representation of subplot grid [rows, cols]
        main (int): Subplot column number
        sub (int): Subplot row number

    Returns:
        [type]: [description]
    """
    
    N = main - 1 + (sub-1)*grid[1]
    
    return gs1[N]    

#======================================================================================================================
def labels( ax, xlabel=None, ylabel=None):
    """
    Set up the label names if required

    Args:
        xlabel (str, optional): X-axis label. Defaults to None.
        ylabel (str, optional): Y-axis label. Defaults to None.
    """

    if xlabel is not None:
        ax.set_xlabel(xlabel, size=14, horizontalalignment='right', x=1.0)
    if ylabel is not None:
        ax.set_ylabel(ylabel, size=14, horizontalalignment='right', y=1.0)



#======================================================================================================================
def style(
    ax, 
    xlog=False, 
    ylog=False, 
    xticks=[], 
    yticks=[], 
    xticklabels=True, 
    yticklabels=True, 
    xgrid=False, 
    ygrid=False, 
    xlim=[], 
    ylim=[], 
    legend_ncol=1, 
    legend_loc='upper right',
    legend_latex=False,
    legend_size=10.5,
    energy_cm=None,
    lumi=None,
    year=None,  #deprecated
    cms_label=None,
):
    """
    Set up the plot style, legends and information on top. Many parameters available to be used if necessary

    Args:
        ax (matplotlib.pyplot.axis): Subplot axis object.
        lumi (int, optional): Dataset's luminosity. Defaults to None.
        year (int, optional): Dataset's year. Defaults to None.
        xlog (bool, optional): Boolean flag to apply log scale to x-axis. Defaults to False.
        ylog (bool, optional): Boolean flag to apply log scale to y-axis. Defaults to False.
        xticks (list, optional): Custom ticks position to x-axis. Defaults to [].
        yticks (list, optional): Custom ticks position to y-axis. Defaults to [].
        xticklabels (bool, optional): Boolean flag to show/hide ticker labels in x-axis. Defaults to True.
        yticklabels (bool, optional): Boolean flag to show/hide ticker labels in y-axis. Defaults to True.
        xgrid (bool, optional): Boolean flag to show/hide grid lines in x-axis. Defaults to False.
        ygrid (bool, optional): Boolean flag to show/hide grid lines in y-axis. Defaults to False.
        xlim (list, optional): 2-sized list with minimum and maximum limits for x-axis. Defaults to [].
        ylim (list, optional): 2-sized list with minimum and maximum limits for x-axis. Defaults to [].
        legend_ncol (int, optional): Number of columns in legend. Defaults to 1.
        legend_loc (str, optional): Legend's position for each subplot in column. Defaults to 'upper right'.
    """
    ax.tick_params(which='major', length=8)
    ax.tick_params(which='minor', length=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.margins(x=0)
    
    #x1,x2,y1,y2 = plt.axis()
    #plt.axis((x1,x2,y1 - 0.00001*y1,y2 + 0.05*y2))

    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    if xgrid:
        ax.grid(which='major', axis='x', linewidth=0.2, linestyle='-', color='0.75')
    if ygrid:
        ax.grid(which='major', axis='y', linewidth=0.2, linestyle='-', color='0.75')
    if len(xlim) > 0:
        ax.set_xlim(xlim)
    if len(ylim) > 0:
        ax.set_ylim(ylim)
    if len(xticks) > 0:
        plt.xticks(xticks)
    if len(yticks) > 0:
        plt.yticks(yticks)   
    if not(xticklabels):
        plt.setp(ax.get_xticklabels(), color="none")
    if not(yticklabels):
        plt.setp(ax.get_yticklabels(), color="none")

    if cms_label is not None:
        hep.cms.text(cms_label, ax=ax, fontsize=13)

    if energy_cm is not None and lumi is None:
        hep.cms.lumitext('$('+str(energy_cm)+'\ \mathrm{TeV})$', ax=ax, fontsize=13)
    elif energy_cm is None and lumi is not None:
        hep.cms.lumitext(str(lumi)+'$\ \mathrm{fb}^{-1}$', ax=ax, fontsize=13)
    elif energy_cm is not None and lumi is not None:
        hep.cms.lumitext(str(lumi)+'$\ \mathrm{fb}^{-1}\ ('+str(energy_cm)+'\ \mathrm{TeV})$', ax=ax, fontsize=13)


    if legend_latex:
        matplotlib.rcParams['text.usetex'] = True
    ax.legend(numpoints=1, ncol=legend_ncol, prop={'size': legend_size}, frameon=False, loc=legend_loc)
    if legend_latex:
        matplotlib.rcParams['text.usetex'] = False
