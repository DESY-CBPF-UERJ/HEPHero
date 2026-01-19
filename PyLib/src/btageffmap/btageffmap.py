import json
import pkgutil

import numpy as np
import statsmodels.stats.proportion as prop

btagging_file = pkgutil.get_data(__package__, "data/btagging.json").decode("utf-8")
btagging = json.loads(btagging_file)
del btagging_file

hadron_flavour_file = pkgutil.get_data(__package__, "data/hadron_flavour.json").decode(
    "utf-8"
)
hadron_flavour = json.loads(hadron_flavour_file)
del hadron_flavour_file


class BTaggingEfficiencyMap:
    """
    Class for computing b-tagging efficiency map for a given dataset.
    """

    def __init__(self, df, eta_bins):
        self.df = df
        self.eta_bins = eta_bins

    def calib(self, year, apv, algo, working_point):
        """
        Calibrate dataset for b-tagging algorithm working point.

        Args:
            year (str): Monte Carlo campaign year - 2016, 2017, 2018
            apv (bool): Flag that indicates if it is an APV dataset
            algo (str): b-tagging algorithm - DeepCSV, DeepJet
            working_point (str): b-tagging algorithm working point - loose, medium, tight
        """
        if apv:
            year = f"APV_{year}"
        btag_thr = btagging.get(year).get(algo).get(working_point)
        if algo.lower() == "deepcsv":
            df_tagged = self.df[self.df.Jet_btagDeepB > btag_thr]
        elif algo.lower() == "deepjet":
            df_tagged = self.df[self.df.Jet_btagDeepFlavB > btag_thr]
        else:
            raise ValueError("b-tagging algorithm must be deepcsv or deepjet.")

        self.df_tagged = df_tagged.reset_index(drop=True)

    def compute_pt_bins(self, pt_min, pt_max, step_size, max_unc):
        """
        Compute pt bins sweeping pt in range [pt_min, pt_max] by given step size,
        a given bin is accepted if the statistical uncertainty in the number of b-tagged events
        is positive and lesser then `unc` for all eta_bins.

        Args:
            pt_min (list, np.array): Minimum pt for sweeping.
            pt_max (list, np.array): Maximum pt for sweeping.
            step_size (float, int): Sweeping step size in pt array.
            max_unc (float, int): Accepted statistical uncertainty in the number of b-tagged events.
        """
        pt_bins = [pt_min]
        current_bin = (pt_min, pt_min + step_size)

        while current_bin[1] < pt_max:
            multiplier = 0
            hist_tag, _, _ = np.histogram2d(
                self.df_hf_tagged.Jet_pt,
                np.abs(self.df_hf_tagged.Jet_eta),
                bins=[current_bin, self.eta_bins],
                weights=self.df_hf_tagged.evtWeight,
            )
            hist_tag2, _, _ = np.histogram2d(
                self.df_hf_tagged.Jet_pt,
                np.abs(self.df_hf_tagged.Jet_eta),
                bins=[current_bin, self.eta_bins],
                weights=self.df_hf_tagged.evtWeight**2,
            )
            unc = np.sqrt(hist_tag2) / hist_tag
            if (unc < max_unc).all() and (unc > 0).all():
                multiplier = 1
                pt_bins.append(current_bin[1])
                current_bin = (current_bin[1], current_bin[1] + step_size)
            else:
                multiplier += 1
                current_bin = (current_bin[0], current_bin[1] + multiplier * step_size)
                if current_bin[1] + multiplier * step_size >= pt_max:
                    if len(pt_bins) > 1:
                        del pt_bins[-1]
                    pt_bins.append(current_bin[1] + multiplier * step_size)
                    break

        return pt_bins

    def make(
        self,
        pt_min=None,
        pt_max=None,
        step_size=None,
        max_unc=None,
        find_best_unc=True,
        unc_stop=0.5,
        unc_increase=0.001,
    ):
        """
        Make b-tagging efficiency map for given dataset
        """
        efficiency_map = {hf: [] for hf in hadron_flavour.keys()}
        uncertainty_map = {hf: [] for hf in hadron_flavour.keys()}

        if max_unc is None:
            raise ValueError("max_argument must be an dict or a float.")

        for hf_value, hf_id in hadron_flavour.items():

            # Filter by hadron flavour
            df_flav = self.df[self.df.Jet_hadronFlavour == hf_id]
            df_flav = df_flav.reset_index(drop=True)
            df_btag = self.df_tagged[self.df_tagged.Jet_hadronFlavour == hf_id]
            df_btag = df_btag.reset_index(drop=True)

            # Compute pt bins
            if isinstance(max_unc, int):
                munc = max_unc
            else:
                munc = max_unc.get(hf_value)

            self.df_hf_tagged = df_btag.copy()

            if find_best_unc:
                pt_bins = []
                while len(pt_bins) <= 2:
                    pt_bins = self.compute_pt_bins(pt_min, pt_max, step_size, munc)
                    if munc >= unc_stop:
                        break
                    munc += unc_increase
            else:
                pt_bins = self.compute_pt_bins(pt_min, pt_max, step_size, munc)

            # Compute efficiency histogram
            hist_flav, xedges_flav, yedges_flav = np.histogram2d(
                df_flav.Jet_pt,
                np.abs(df_flav.Jet_eta),
                bins=[pt_bins, self.eta_bins],
                weights=df_flav.evtWeight,
            )
            hist_flav_pow2, _, _ = np.histogram2d(
                df_flav.Jet_pt,
                np.abs(df_flav.Jet_eta),
                bins=[pt_bins, self.eta_bins],
                weights=df_flav.evtWeight**2,
            )
            hist_flav_noweight, _, _ = np.histogram2d(
                df_flav.Jet_pt,
                np.abs(df_flav.Jet_eta),
                bins=[pt_bins, self.eta_bins],
            )

            hist_btag, _, _ = np.histogram2d(
                df_btag.Jet_pt,
                np.abs(df_btag.Jet_eta),
                bins=[pt_bins, self.eta_bins],
                weights=df_btag.evtWeight,
            )
            hist_btag_pow2, _, _ = np.histogram2d(
                df_btag.Jet_pt,
                np.abs(df_btag.Jet_eta),
                bins=[pt_bins, self.eta_bins],
                weights=df_btag.evtWeight**2,
            )
            hist_btag_noweight, _, _ = np.histogram2d(
                df_btag.Jet_pt,
                np.abs(df_btag.Jet_eta),
                bins=[pt_bins, self.eta_bins],
            )

            hist_flav_err = np.sqrt(hist_flav_pow2)
            hist_btag_err = np.sqrt(hist_btag_pow2)

            effm = []
            for i in range(len(xedges_flav) - 1):
                for j in range(len(yedges_flav) - 1):
                    eff = hist_btag[i, j] / hist_flav[i, j]
                    eff_noweight = hist_btag_noweight[i, j] / hist_flav_noweight[i, j]
                    eff_err_prop = eff * np.sqrt(
                        (hist_btag_err[i, j] / hist_btag[i, j]) ** 2
                        + (hist_flav_err[i, j] / hist_flav[i, j]) ** 2
                    )
                    y_below, y_above = prop.proportion_confint(
                        hist_btag_noweight[i, j],
                        hist_flav_noweight[i, j],
                        alpha=0.32,
                        method="beta",
                    )
                    eff_err_clopper = [eff_noweight - y_below, y_above - eff_noweight]

                    if np.isnan(eff):
                        eff = None
                        eff_err_prop = None
                        eff_err_clopper = [None, None]

                    effm.append(
                        {
                            "eta_min": yedges_flav[j],
                            "eta_max": yedges_flav[j + 1],
                            "pt_min": xedges_flav[i],
                            "pt_max": xedges_flav[i + 1],
                            "eff": eff,
                            "eff_err_prop": eff_err_prop,
                            "eff_err_clopper": eff_err_clopper,
                        }
                    )
            efficiency_map[hf_value] = effm
            uncertainty_map[hf_value] = munc

        return efficiency_map, uncertainty_map
