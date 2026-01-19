import warnings

import numpy as np
import matplotlib.pyplot as plt 
from sklearn import metrics 

class control:
    """
    Produce control information to assist in the defition of cuts
    """
    #==============================================================================================================
    def __init__(self, var, signal_list, others_list, weight=None, bins=np.linspace(0,100,5), above=True):
        self.bins = bins
        self.var = var
        self.signal_list = signal_list
        self.others_list = others_list
        self.weight = weight         
        
        use_bins = [np.array([-np.inf]), np.array(bins), np.array([np.inf])]
        use_bins = np.concatenate(use_bins)
    
        hist_signal_list = []
        for signal in signal_list:
            if weight is not None:
                hist, hbins = np.histogram( signal[var], weights=signal[weight], bins=use_bins )
            else:
                hist, hbins = np.histogram( signal[var], bins=use_bins )
            if not above:
                hist = np.cumsum(hist)
                hist = hist[:-1]
            else:
                hist = np.cumsum(hist[::-1])[::-1]
                hist = hist[1:]
            hist_signal_list.append(hist)
        hist_signal = hist_signal_list[0]
        for i in range(len(signal_list)-1):
            hist_signal = hist_signal + hist_signal_list[i+1]
        self.hist_signal = hist_signal
    
        hist_others_list = []
        for others in others_list:
            if weight is not None:
                hist, hbins = np.histogram( others[var], weights=others[weight], bins=use_bins )
            else:
                hist, hbins = np.histogram( others[var], bins=use_bins )
            if not above:
                hist = np.cumsum(hist)
                hist = hist[:-1]
            else:
                hist = np.cumsum(hist[::-1])[::-1]
                hist = hist[1:]
            hist_others_list.append(hist)
        hist_others = hist_others_list[0]
        for i in range(len(others_list)-1):
            hist_others = hist_others + hist_others_list[i+1]
        self.hist_others = hist_others    
    
        signal_sum_list = []
        for signal in signal_list:
            if weight is not None:
                signal_sum = signal[weight].sum()
            else:
                signal_sum = len(signal[var])
            signal_sum_list.append(signal_sum)
        full_signal = signal_sum_list[0]
        for i in range(len(signal_list)-1):
            full_signal = full_signal + signal_sum_list[i+1]
        self.full_signal = full_signal
        
        others_sum_list = []
        for others in others_list:
            if weight is not None:
                others_sum = others[weight].sum()
            else:
                others_sum = len(others[var])
            others_sum_list.append(others_sum)
        full_others = others_sum_list[0]
        for i in range(len(others_list)-1):
            full_others = full_others + others_sum_list[i+1]
        self.full_others = full_others
        
        self.purity = self.hist_signal/(self.hist_signal + self.hist_others)
        self.eff_signal = self.hist_signal/self.full_signal
        self.eff_others = self.hist_others/self.full_others
        self.rej_others = 1 - self.eff_others
    
    #==============================================================================================================
    def purity_plot(self, label='Signal purity', color='blue', cuts=None):
        plt.plot(self.bins, self.purity, color=color, label=label)
        if cuts is None:
            return None
        else:
            pur_values = []
            for cut in cuts:
                test = min(enumerate(self.bins), key=lambda x: abs(x[1]-cut))
                pur = self.purity[test[0]]
                pur_values.append(pur)
            return pur_values

    #==============================================================================================================
    def signal_eff_plot(self, label='Signal eff.', color='green', cuts=None):
        plt.plot(self.bins, self.eff_signal, color=color, label=label)
        if cuts is None:
            return None
        else:
            eff_sig_values = []
            for cut in cuts:
                test = min(enumerate(self.bins), key=lambda x: abs(x[1]-cut))
                eff_sig = self.eff_signal[test[0]]
                eff_sig_values.append(eff_sig)
            return eff_sig_values
     
    #============================================================================================================== 
    def bkg_eff_plot(self, label='Bkg. eff.', color='red', cuts=None):
        plt.plot(self.bins, self.eff_others, color=color, label=label)    
        if cuts is None:
            return None
        else:
            eff_others_values = []
            for cut in cuts:
                test = min(enumerate(self.bins), key=lambda x: abs(x[1]-cut))
                eff_others = self.eff_others[test[0]]
                eff_others_values.append(eff_others)
            return eff_others_values
        
    #==============================================================================================================    
    def bin_purity_plot(self, label='Signal purity per bin', color='blue', bins=None):
        
        if bins is None:
            bins=self.bins
    
        hist_signal_list = []
        for signal in self.signal_list:
            if self.weight is not None:
                hist, bins = np.histogram( signal[self.var], weights=signal[self.weight], bins=bins )
            else:
                hist, bins = np.histogram( signal[self.var], bins=bins )
            hist_signal_list.append(hist)
        hist_signal = hist_signal_list[0]
        for i in range(len(self.signal_list)-1):
            hist_signal = hist_signal + hist_signal_list[i+1]
    
        hist_others_list = []
        for others in self.others_list:
            if self.weight is not None:
                hist, bins = np.histogram( others[self.var], weights=others[self.weight], bins=bins )
            else:
                hist, bins = np.histogram( others[self.var], bins=bins )
            hist_others_list.append(hist)
        hist_others = hist_others_list[0]
        for i in range(len(self.others_list)-1):
            hist_others = hist_others + hist_others_list[i+1]
        
        hist_signal_purity = hist_signal/(hist_signal + hist_others)
        bincentres = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
        hist_signal_purity = hist_signal_purity.tolist()
    
        left_bins = [ bins[0], bincentres[0] ]
        right_bins = [ bincentres[-1], bins[-1] ]
    
        plt.plot(left_bins, [hist_signal_purity[0], hist_signal_purity[0]], color=color)
        plt.plot(right_bins, [hist_signal_purity[-1], hist_signal_purity[-1]], color=color)
        plt.step(bincentres, hist_signal_purity, where='mid', color=color, label=label)    
        
        
    #==============================================================================================================
    def roc_plot(self, label='Signal-bkg ROC', color='blue', linestyle="-", version=1):
        if version == 1:
            plt.plot(self.rej_others, self.eff_signal, color=color, label=label, linestyle=linestyle)
        elif version == 2:
            plt.plot(self.eff_signal, self.eff_others, color=color, label=label, linestyle=linestyle)
    
    #==============================================================================================================
    def auc(self, method="trapezoidal"):
        if method == "sklearn":
            area = metrics.auc(self.rej_others, self.eff_signal)
        if method == "trapezoidal":
            area = 0
            for i in range(len(self.bins)-1):
                area += 0.5*(self.eff_signal[i+1] + self.eff_signal[i])*abs(self.rej_others[i+1] - self.rej_others[i])
        return area    
    
    #==============================================================================================================
    def bkg_eff(self, cut, apx=False):
        eff_bkg = -999
        if apx is True:
            test = min(enumerate(self.bins), key=lambda x: abs(x[1]-cut))
            eff_bkg = self.eff_others[test[0]]
            if eff_bkg == -999:
                print("Enter a value that exists in bins")
        else:
            for i in range(len(self.bins)-1):
                if cut == self.bins[i]:
                    eff_bkg = self.eff_others[i]
            if eff_bkg == -999:
                print("Enter a value that exists in bins")
        return eff_bkg
    
    #==============================================================================================================
    def effpur_plot(self, label='Eff*Pur', color='orchid', cuts=None, normalize=True):
        effpur_vec = self.eff_signal*self.purity
        if normalize:
            not_nan_array = ~ np.isnan(effpur_vec)
            effpur_vec_good = effpur_vec[not_nan_array]
            effpur_max = np.amax(effpur_vec_good)
            effpur_vec = effpur_vec/effpur_max
            label = label + r" ($\div$ " + "{:.2e}".format(effpur_max) + ")" 
        plt.plot(self.bins, effpur_vec, color=color, label=label)
        if cuts is None:
            return None
        else:
            pur_values = []
            for cut in cuts:
                test = min(enumerate(self.bins), key=lambda x: abs(x[1]-cut))
                effpur = effpur_vec[test[0]]
                effpur_values.append(effpur)
            return effpur_values
    
    #==============================================================================================================
    def prc_plot(self, label='Signal-bkg PRC', color='blue', linestyle="-", normalize=True, cut_eff_signal=None):
        effpur_vec = self.eff_signal*self.purity

        # Sanity check
        if np.where(effpur_vec > 1)[0].size > 0:
            warnings.warn('Encountered effpur values above 1.')

        if normalize:
            not_nan_array = ~ np.isnan(effpur_vec)
            effpur_vec_good = effpur_vec[not_nan_array]
            effpur_max = np.amax(effpur_vec_good)
            effpur_vec = effpur_vec/effpur_max
            label = label + r" ($\div$ " + "{:.2e}".format(effpur_max) + ")" 
        plt.plot(self.eff_signal, effpur_vec, color=color, label=label, linestyle=linestyle)
        if cut_eff_signal:
            plt.axvline(x=cut_eff_signal, color='grey', linestyle='--')
        
    #==============================================================================================================
    def best_cut(self, cut_eff_signal=None):
        effpur_vec = self.eff_signal*self.purity

        # Transform effpur into -1 where eff_signal <= cut_eff_signal
        if cut_eff_signal:
            effpur_vec[self.eff_signal <= cut_eff_signal] = -1

        not_nan_array = ~ np.isnan(effpur_vec)
        effpur_vec_good = effpur_vec[not_nan_array]
        effpur_max = np.amax(effpur_vec_good)
        cut_idx = np.where(effpur_vec == effpur_max)
        best_cut = self.bins[cut_idx]
        return best_cut[0], effpur_max
        
        
