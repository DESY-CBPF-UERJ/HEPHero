import numpy as np
from iminuit.cost import poisson_chi2, chi2
#import mpmath as mp

#======================================================================================================================
"""
def pdf_efficiency( e, k, n ):
    # Enter a float (or a list) of efficiencie(s) and return the pdf value associated to it, considering the parameters k (number of events selected), and n (total number of events).
    # The gamma function returns reasonable values for n < 10000.
    # For MC with weights different of 1, the k and n values will be approximated to the nearest integer
    n = int(n)
    k = int(k)
    
    if n > 10000:
        print("Warning: n greater than 10000!")
    
    if isinstance(e, float) or isinstance(e, int):
        e = np.array([e])
        number_entries = 1
    else:
        number_entries = len(e)
    
    P = np.zeros(number_entries)
    
    if k > n:
        k = n
    if k < 0:
        k = 0
    if n < 0:
        n = 0
        
    for i in range(number_entries):
        if (e[i] >= 0) and (e[i] <= 1):
            P[i] = (mp.gamma(n+2)/(mp.gamma(k+1)*mp.gamma(n-k+1)))*np.power(e[i],k)*np.power(1-e[i],n-k)
    if len(P) == 1:
        P = P[0]
    
    return P
"""

#======================================================================================================================    
def get_interval(x, pdf, nsigma=1):
    # Enter two arrays with the x values and the pdf values associated to them
    # Returns inferios and superior limits associated to the confidence level of 1 sigma
    # The pdf must have a single maximum
    
    if nsigma != 1 and nsigma != 2:
        print("Enter a nsigma equal to 1 or 2 !")
        return 0, 1
    
    if nsigma == 1:
        area_nsigma = 0.682689492137086
    elif nsigma == 2:
        area_nsigma = 0.954499736103642
    
    max_idx = np.where(pdf == pdf.max())[0][0]
    
    area_right = []
    for i in range(max_idx, len(x)-1):
        delta_area = 0.5*(pdf[i+1] + pdf[i])*abs(x[i+1] - x[i])
        area_right.append(delta_area)
    area_right = np.cumsum(area_right)
    
    area_left = []
    for i in range(0, max_idx-1):
        delta_area = 0.5*(pdf[i+1] + pdf[i])*abs(x[i+1] - x[i])
        area_left.append(delta_area)
    area_left.reverse()
    area_left = np.cumsum(area_left)
    
    if max_idx == len(x)-1:
        exceeded = False
        for i in range(len(area_left)):
            area_i = area_left[i]
            if area_i > area_nsigma:
                alpha_idx = max_idx - i - 1
                beta_idx = len(x)-1
                exceeded = True
                break
    elif max_idx == 0:
        exceeded = False
        for j in range(len(area_right)):
            area_i = area_right[j]
            if area_i > area_nsigma:
                alpha_idx = 0
                beta_idx = max_idx + j + 1
                exceeded = True
                break
    else:
        exceeded = False
        for i in range(len(area_left)):
            for j in range(len(area_right)):
                area_i = area_left[i] + area_right[j]
                if area_i > area_nsigma:
                    alpha1_idx = max_idx - i - 1
                    beta1_idx = max_idx + j + 1
                    exceeded = True
                    break
            if exceeded:
                break
        interval1 = x[beta1_idx] - x[alpha1_idx]
    
        exceeded = False
        for j in range(len(area_right)):
            for i in range(len(area_left)):
                area_i = area_left[i] + area_right[j]
                if area_i > area_nsigma:
                    alpha2_idx = max_idx - i - 1
                    beta2_idx = max_idx + j + 1
                    exceeded = True
                    break
            if exceeded:
                break
        interval2 = x[beta2_idx] - x[alpha2_idx]
        
        alpha_idx = alpha1_idx
        beta_idx = beta1_idx
        if interval2 < interval1:
            alpha_idx = alpha2_idx
            beta_idx = beta2_idx
    
    alpha = x[alpha_idx]
    beta = x[beta_idx]
    
    return alpha, beta


#======================================================================================================================
def correlation(x, y, weight=None):
    # Returns the linear correlation between the variables x (1D array) and y (1D array).
    # w (1D array) are the event weights. Events with negative weights are not considered
    w = weight
    if w is None:
        w = np.ones(len(x))
    boolean = [w >= 0][0]
    w = np.array(w).ravel()
    w = w[boolean]
    x = np.array(x).ravel()
    x = x[boolean]
    y = np.array(y).ravel()
    y = y[boolean]
    cov_xy = np.sum(w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w))) / np.sum(w)
    cov_xx = np.sqrt(np.sum(w * (x - np.average(x, weights=w)) * (x - np.average(x, weights=w))) / np.sum(w))
    cov_yy = np.sqrt(np.sum(w * (y - np.average(y, weights=w)) * (y - np.average(y, weights=w))) / np.sum(w))
    return cov_xy/(cov_xx*cov_yy)


def smooth_array(x, repeat=1):
    # 353QH
    # void  TH1::SmoothArray(Int_t nn, Double_t *xx, Int_t ntimes)
    nbins = len(x)
    for ipass in range(repeat):
        if nbins >= 3:
            nn = nbins

            xx = np.zeros_like(x)
            for iBin in range(nbins):
                if x[iBin] > 0:
                    xx[iBin] = x[iBin]

            # first copy original data into temp array -> copied xx to zz
            zz = xx.copy()


            for noent in range(2):  # run algorithm two times

                #  do 353 i.e. running median 3, 5, and 3 in a single loop
                for kk in range(3):
                    yy = zz.copy()

                    if kk != 1:
                        medianType = 3
                        ifirst = 1
                        ilast = nn-1
                    else:
                        medianType = 5
                        ifirst = 2
                        ilast = nn-2

                    # do all elements beside the first and last point for median 3
                    #  and first two and last 2 for median 5
                    hh = np.zeros(medianType)
                    for ii in range(ifirst,ilast,1):
                        for jj in range(0,medianType,1):
                            hh[jj] = yy[ii - ifirst + jj]
                        zz[ii] = np.median(hh)


                    if kk == 0:   # first median 3
                        # first point
                        hh[0] = zz[1]
                        hh[1] = zz[0]
                        hh[2] = 3*zz[1] - 2*zz[2]
                        zz[0] = np.median(hh)
                        # last point
                        hh[0] = zz[nn - 2];
                        hh[1] = zz[nn - 1];
                        hh[2] = 3*zz[nn - 2] - 2*zz[nn - 3];
                        zz[nn - 1] = np.median(hh)

                    if kk == 1:   #  median 5
                        for ii in range(0,3,1):
                            hh[ii] = yy[ii]
                        zz[1] = np.median(hh)
                        # last two points
                        for ii in range(0,3,1):
                            hh[ii] = yy[nn - 3 + ii]
                        zz[nn - 2] = np.median(hh)



                yy = zz.copy()  # -> copied zz to yy

                # quadratic interpolation for flat segments
                for ii in range(2,nn-2,1):
                    if (zz[ii - 1] != zz[ii]) or (zz[ii] != zz[ii + 1]):
                        continue
                    hh[0] = zz[ii - 2] - zz[ii]
                    hh[1] = zz[ii + 2] - zz[ii]
                    if hh[0]*hh[1] <= 0:
                        continue
                    jk = 1
                    if np.abs(hh[1]) > np.abs(hh[0]):
                        jk = -1
                    yy[ii] = -0.5*zz[ii - 2*jk] + zz[ii]/0.75 + zz[ii + 2*jk]/6.
                    yy[ii + jk] = 0.5*(zz[ii + 2*jk] - zz[ii - 2*jk]) + zz[ii]

                # running means
                for ii in range(1,nn-1,1):
                    zz[ii] = 0.25*yy[ii - 1] + 0.5*yy[ii] + 0.25*yy[ii + 1]
                zz[0] = yy[0]
                zz[nn - 1] = yy[nn - 1]

                if noent == 0:
                    # save computed values
                    rr = zz.copy()  # -> copied zz to rr

                    # COMPUTE  residuals
                    for ii in range(0,nn,1):
                        zz[ii] = xx[ii] - zz[ii]

            xmin = np.amin(xx)
            for ii in range(0,nn,1):
                if xmin < 0:
                    xx[ii] = rr[ii] + zz[ii]
                else: # make smoothing defined positive - not better using 0 ?
                    xx[ii] = max(rr[ii] + zz[ii], 0.0)

    return xx

#======================================================================================================================
class analysis_model:
    def __init__(self, xe_regions, n_regions, t_regions, tsys_regions, N_rateParam=1):
        self.xe_regions = xe_regions
        self.data_regions = n_regions, t_regions, tsys_regions
        self.N_rateParam = N_rateParam

    def _pred(self, par):
        n_regions, t_regions, tsys_regions = self.data_regions

        N_nuisances_stat_param = 0
        for ir in range(len(n_regions)):
            if len(n_regions[ir]) > 1:
                N_nuisances_stat_param += len(t_regions[ir])*len(n_regions[ir])

        self.N_nuisances_syst_param = int(len(tsys_regions[0])/(2*len(t_regions[0])))

        rate = par[:self.N_rateParam]

        if N_nuisances_stat_param > 0:
            nuisances_stat = np.array(par[self.N_rateParam:N_nuisances_stat_param+self.N_rateParam])
            gamma = np.empty(len(n_regions)).tolist()
            bins_control = 0
            for ir in range(len(n_regions)):
                gamma[ir] = []
                if len(n_regions[ir]) > 1:
                    bins = len(self.xe_regions[ir]) - 1
                    for p in range(len(t_regions[ir])):
                        gamma[ir].append(nuisances_stat[bins_control+p*bins:bins_control+(p+1)*bins])
                    bins_control += len(t_regions[ir])*bins

        alpha = np.array(par[N_nuisances_stat_param+self.N_rateParam:])

        #print("rate", rate)
        #print("alpha", alpha)

        E_n_regions = []
        E_t_regions = []
        for ir in range(len(self.xe_regions)):
            bins = len(self.xe_regions[ir]) - 1
            #n_regions, t_regions, tsys_regions = self.data_regions

            delta = np.empty((len(alpha), len(t_regions[ir]), 2)).tolist()
            ii = 0
            for i in range(len(alpha)):
                for p in range(len(t_regions[ir])):
                    for u in range(2):
                        delta[i][p][u] = tsys_regions[ir][ii] - t_regions[ir][p]
                        ii += 1

            talpha = t_regions[ir].copy()
            for p in range(len(t_regions[ir])):
                for i in range(len(alpha)):
                    if np.abs(alpha[i]) <= 1:
                        talpha[p] = talpha[p] + 0.5*((delta[i][p][1]-delta[i][p][0])*alpha[i] + (1./8.)*(delta[i][p][1]+delta[i][p][0])*(3*alpha[i]**6 - 10*alpha[i]**4 + 15*alpha[i]**2))
                    elif alpha[i] > 1:
                        talpha[p] = talpha[p] + delta[i][p][1]*np.abs(alpha[i])
                    elif alpha[i] < -1:
                        talpha[p] = talpha[p] + delta[i][p][0]*np.abs(alpha[i])

            E_n = 0
            E_t = []
            for p in range(len(t_regions[ir])):
                if len(n_regions[ir]) > 1:
                    E_t.append(gamma[ir][p]*talpha[p])
                    if p < self.N_rateParam:
                        E_n += rate[p]*gamma[ir][p]*talpha[p] # change it
                    else:
                        E_n += gamma[ir][p]*talpha[p]
                else:
                    E_t.append(talpha[p])
                    if p < self.N_rateParam:
                        E_n += rate[p]*talpha[p]
                    else:
                        E_n += talpha[p]

            E_n_regions.append(E_n)
            E_t_regions.append(E_t)

        return E_n_regions, E_t_regions, alpha

    def __call__(self, par):
        n_regions, t_regions, tsys_regions = self.data_regions

        E_n_regions, E_t_regions, alpha = self._pred(par)
        for ir in range(len(E_n_regions)):
            if ir == 0:
                r = poisson_chi2(n_regions[ir], E_n_regions[ir])
            else:
                r += poisson_chi2(n_regions[ir], E_n_regions[ir])

        for ir in range(len(n_regions)):
            if len(n_regions[ir]) > 1:
                for i in range(len(t_regions[ir])):
                    r += poisson_chi2(t_regions[ir][i], E_t_regions[ir][i])

        for i in range(self.N_nuisances_syst_param):
            r += chi2(alpha[i], 1, 0)
        return r
