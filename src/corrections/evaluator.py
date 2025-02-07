import numpy as np
import uproot
import json
import coffea
from coffea.lookup_tools import dense_lookup
import awkward as ak
import dask_awkward as dak
from omegaconf import OmegaConf
import correctionlib

# PU SF --------------------------------------------------------------------
# def pu_lookups(parameters, mode="nom", auto=[]):
#     lookups = {}
#     branch = {"nom": "pileup", "up": "pileup_plus", "down": "pileup_minus"}
#     for mode in ["nom", "up", "down"]:
#         if len(auto) == 0:
#             pu_hist = uproot.open(parameters["pu_file"])["pileup"].values() # returns a np array
#             nbins = len(pu_hist)
#             edges = [[i for i in range(nbins)]]
#             lookup = dense_lookup.dense_lookup(pu_hist, edges)
#         else:
#             pu_hist_data = uproot.open(parameters["pu_file_data"])[branch[mode]].values()
    
#             nbins = len(pu_hist_data)
#             # print(f"pu_reweight nbins: {nbins}")
#             edges = [[i for i in range(nbins)]]
#             pu_hist_mc = np.histogram(auto, bins=range(nbins + 1))[0]
#             print(f"pu_hist_mc: {pu_hist_mc}")
#             # print(f"pu_lookups type(pu_hist_data): {type(pu_hist_data)}")
#             # ----------------------------------------
#             # if len(auto) == 0:
#             #     pu_hist_mc = uproot.open(parameters["pu_file_mc"])["pu_mc"].values()
#             # else:
#             #     pu_hist_mc = np.histogram(auto, bins=range(nbins + 1))[0]
#             #----------------------------------------------------
    
#             lookup = dense_lookup.dense_lookup(pu_reweight(pu_hist_data, pu_hist_mc), edges)
#         lookups[mode] = lookup
#     return lookups

def pu_lookups(parameters, mode="nom", auto=[]):
    lookups = {}
    branch = {"nom": "pileup", "up": "pileup_plus", "down": "pileup_minus"}
    for mode in ["nom", "up", "down"]:
        pu_hist_data = uproot.open(parameters["pu_file_data"])[branch[mode]].values()

        nbins = len(pu_hist_data)
        # print(f"pu_reweight nbins: {nbins}")
        edges = [[i for i in range(nbins)]]
        # print(f"pu_lookups type(pu_hist_data): {type(pu_hist_data)}")
        # ----------------------------------------
        if len(auto) == 0:
            # pu_hist_mc = uproot.open(parameters["pu_file_mc"])["pu_mc"].values()
            with open(parameters["pu_file_mc"]) as file:
                # config = json.loads(file.read())
                # print(f"pu file: {file}")
                config = OmegaConf.load(file)
                # print(f"config: {config}")
            pu_hist_mc = np.array(config["pu_mc"])
        else:
            pu_hist_mc = np.histogram(auto, bins=range(nbins + 1))[0]
        #----------------------------------------------------

        lookup = dense_lookup.dense_lookup(pu_reweight(pu_hist_data, pu_hist_mc), edges)
        lookups[mode] = lookup
    return lookups

def pu_reweight(pu_hist_data, pu_hist_mc):
    #print(pu_hist_mc)
    # print(f"pu_reweight len(pu_hist_mc): {len(pu_hist_mc)}")
    pu_arr_mc_ = np.zeros(len(pu_hist_mc))
    # for ibin, value in enumerate(pu_hist_mc):
    #     pu_arr_mc_[ibin] = max(value, 0)

    # pu_arr_data = np.zeros(len(pu_hist_data))
    # for ibin, value in enumerate(pu_hist_data):
    #     pu_arr_data[ibin] = max(value, 0)
    pu_arr_mc_ = np.where(pu_hist_mc<0, 0, pu_hist_mc) # min cut of zero
    pu_arr_data = np.where(pu_hist_data<0, 0, pu_hist_data) # min cut of zero
    # print(f"pu_reweight pu_arr_mc_: {pu_arr_mc_}")
    # print(f"pu_reweight pu_arr_data: {pu_arr_data}")
    pu_arr_mc_ref = pu_arr_mc_
    pu_arr_mc = pu_arr_mc_ / np.sum(pu_arr_mc_)
    pu_arr_data = pu_arr_data / np.sum(pu_arr_data)
    #print(pu_arr_mc)
    weights = np.ones(len(pu_hist_mc))
    # print(f"len(pu_hist_mc): {len(pu_hist_mc)}")
    # print(f"len(pu_arr_data): {len(pu_arr_data)}")
    weights[pu_arr_mc != 0] = pu_arr_data[pu_arr_mc != 0] / pu_arr_mc[pu_arr_mc != 0]
    maxw = min(weights.max(), 5.0)
    cropped = []
    while maxw > 3:
        cropped = []
        for i in range(len(weights)):
            cropped.append(min(maxw, weights[i]))
        shift = checkIntegral(cropped, weights, pu_arr_mc_ref)
        if abs(shift) > 0.0025:
            break
        maxw *= 0.95

    maxw /= 0.95
    if len(cropped) > 0:
        for i in range(len(weights)):
            cropped[i] = min(maxw, weights[i])
        normshift = checkIntegral(cropped, weights, pu_arr_mc_ref)
        for i in range(len(weights)):
            weights[i] = cropped[i] * (1 - normshift)
    return weights


def checkIntegral(wgt1, wgt2, ref):
    myint = 0
    refint = 0
    for i in range(len(wgt1)):
        myint += wgt1[i] * ref[i]
        refint += wgt2[i] * ref[i]
    return (myint - refint) / refint


# def pu_evaluator(parameters, numevents, ntrueint):
#     """
#     params:
#     numevents = integer value of 
#     ntrueint = np array for making dense lookup
#     """
#     lookups = pu_lookups(parameters, auto=ntrueint)
#     #print("Hello")
#     pu_weights = {}
#     for var, lookup in lookups.items():
#         pu_weights[var] = np.ones(numevents)
#         pu_weights[var] = lookup(ntrueint)
#         pu_weights[var] = np.array(pu_weights[var])
#         pu_weights[var][ntrueint > 100] = 1
#         pu_weights[var][ntrueint < 1] = 1
#     return pu_weights

def pu_evaluator(parameters, ntrueint, onTheSpot=False, Run=2):
    """
    params:
    numevents = integer value of 
    ntrueint = np array for making dense lookup
    distinction for run2 and run3 is not the most elegant method, but it should
    be good enough for the time being
    """
    if Run ==2:
        if onTheSpot:
            lookups = pu_lookups(parameters, auto=ak.to_numpy(ntrueint.compute()))
        else:
            lookups = pu_lookups(parameters, auto=[])
        #print("Hello")
        pu_weights = {}
        for var, lookup in lookups.items():
            # print(f"lookup(ntrueint): {lookup(ntrueint)}")
            pu_weights[var] = lookup(ntrueint)
            pu_weights[var] = ak.where((ntrueint > 100), 1, pu_weights[var])
            pu_weights[var] = ak.where((ntrueint < 1), 1, pu_weights[var])
            # print(f"pu_weights[{var}]: {pu_weights[var].compute()}")
    elif Run==3:
        jsonGz_path = parameters["pu_file_mc"]
        print(f"jsonGz_path: {jsonGz_path}")
        ceval = correctionlib.CorrectionSet.from_file(jsonGz_path)
        key = list(ceval.keys())[0]
        pu_lookup = ceval[key]
        pu_weights = {}
        pu_weights["nom"] = pu_lookup.evaluate(ntrueint,"nominal")
        pu_weights["up"] = pu_lookup.evaluate(ntrueint,"up")
        pu_weights["down"] = pu_lookup.evaluate(ntrueint,"down")
    else:
        print("ERROR: unacceptable Run value is given!")
        raise ValueError
    return pu_weights



# NNLOPS SF-------------------------------------------------------------------------

class DelayedInterp:
    """
    this is a np.interp wrapper for dask awkward as suggested by 
    Lindsey, from github issue https://github.com/dask-contrib/dask-awkward/issues/493
    """
    def __init__(self, x_knots, y_knots):
        self.x_knots = x_knots
        self.y_knots = y_knots
    
    def __call__(self, vals):
        result = ak.Array(np.interp(
           ak.typetracer.length_zero_if_typetracer(vals), # this will either be a concrete array with data or a type tracer
           self.x_knots,
           self.y_knots,
        ))
        # print(f"result: {result}")
        if ak.backend(vals) == "typetracer":
           return ak.Array(result.layout.to_typetracer(forget_length=True))
        return result


class NNLOPS_Evaluator(object):
    def __init__(self, input_path):
        with uproot.open(input_path) as f:
            self.ratio_0jet = {
                "mcatnlo": f["gr_NNLOPSratio_pt_mcatnlo_0jet"],
                "powheg": f["gr_NNLOPSratio_pt_powheg_0jet"],
            }
            self.ratio_1jet = {
                "mcatnlo": f["gr_NNLOPSratio_pt_mcatnlo_1jet"],
                "powheg": f["gr_NNLOPSratio_pt_powheg_1jet"],
            }
            self.ratio_2jet = {
                "mcatnlo": f["gr_NNLOPSratio_pt_mcatnlo_2jet"],
                "powheg": f["gr_NNLOPSratio_pt_powheg_2jet"],
            }
            self.ratio_3jet = {
                "mcatnlo": f["gr_NNLOPSratio_pt_mcatnlo_3jet"],
                "powheg": f["gr_NNLOPSratio_pt_powheg_3jet"],
            }

    def evaluate(self, hig_pt, njets, mode):
        # result = np.ones(len(hig_pt), dtype=float)
        # print(f'nnlops sf len(hig_pt): {len(hig_pt)}')
        result = ak.ones_like(hig_pt)
        # njet0_filter = (hig_pt < 125) & (njets == 0)
        # interp_in = ak.where(njet0_filter, hig_pt, 125)
        njet0_interp = DelayedInterp(
            self.ratio_0jet[mode].member("fX"),
            self.ratio_0jet[mode].member("fY")
        )
        njet0_interp_out = dak.map_partitions(
            njet0_interp,
            ak.where((hig_pt < 125), hig_pt, 125.0)
        )
        # njet0_interp_out =  np.interp(
        #     ak.where((hig_pt < 125), hig_pt, 125.0),
        #     self.ratio_0jet[mode].member("fX"),
        #     self.ratio_0jet[mode].member("fY"),
        # )
        result = ak.where((njets == 0), njet0_interp_out, result)
        # ------------------------------------------------------------#
        njet1_interp = DelayedInterp(
            self.ratio_1jet[mode].member("fX"),
            self.ratio_1jet[mode].member("fY")
        )
        njet1_interp_out = dak.map_partitions(
            njet1_interp,
            ak.where((hig_pt < 800), hig_pt, 800.0)
        )
        # print(f"njet1_interp_out: {njet1_interp_out.compute()}")
        # njet1_interp_out =  np.interp(
        #     ak.where((hig_pt < 800), hig_pt, 800.0),
        #     self.ratio_1jet[mode].member("fX"),
        #     self.ratio_1jet[mode].member("fY"),
        # )
        result = ak.where((njets == 1), njet1_interp_out, result)
        # ------------------------------------------------------------#
        njet2_interp = DelayedInterp(
            self.ratio_2jet[mode].member("fX"),
            self.ratio_2jet[mode].member("fY")
        )
        njet2_interp_out = dak.map_partitions(
            njet2_interp,
            ak.where((hig_pt < 800), hig_pt, 800.0)
        )
        # njet2_interp_out =  np.interp(
        #     ak.where((hig_pt < 800), hig_pt, 800.0),
        #     self.ratio_2jet[mode].member("fX"),
        #     self.ratio_2jet[mode].member("fY"),
        # )
        result = ak.where((njets == 2), njet2_interp_out, result)
        # ------------------------------------------------------------#
        njet3_interp = DelayedInterp(
            self.ratio_3jet[mode].member("fX"),
            self.ratio_3jet[mode].member("fY")
        )
        njet3_interp_out = dak.map_partitions(
            njet3_interp,
            ak.where((hig_pt < 925), hig_pt, 925.0)
        )
        # njet3_interp_out =  np.interp(
        #     ak.where((hig_pt < 925), hig_pt, 925.0),
        #     self.ratio_3jet[mode].member("fX"),
        #     self.ratio_3jet[mode].member("fY"),
        # )
        result = ak.where((njets > 2), njet3_interp_out, result)
        
        # njet0_interp_out =  np.interp(
        #     ak.where((hig_pt < 125), hig_pt, 125.0),
        #     self.ratio_0jet[mode].member("fX"),
        #     self.ratio_0jet[mode].member("fY"),
        # )
        # result = ak.where((njets == 0), njet0_interp_out, result)
        
        # result[njets == 0] = np.interp(
        #     np.minimum(hig_pt[njets == 0], 125.0),
        #     self.ratio_0jet[mode].member("fX"),
        #     self.ratio_0jet[mode].member("fY"),
        # )
        # result[njets == 1] = np.interp(
        #     np.minimum(hig_pt[njets == 1], 625.0),
        #     self.ratio_1jet[mode].member("fX"),
        #     self.ratio_1jet[mode].member("fY"),
        # )
        # result[njets == 2] = np.interp(
        #     np.minimum(hig_pt[njets == 2], 800.0),
        #     self.ratio_2jet[mode].member("fX"),
        #     self.ratio_2jet[mode].member("fY"),
        # )
        # result[njets > 2] = np.interp(
        #     np.minimum(hig_pt[njets > 2], 925.0),
        #     self.ratio_3jet[mode].member("fX"),
        #     self.ratio_3jet[mode].member("fY"),
        # )
        return result


# def nnlops_weights(events, parameters, dataset):
#     nnlops = NNLOPS_Evaluator(parameters["nnlops_file"])
#     if "amc" in dataset:
#         mc_generator = "mcatnlo"
#     elif "powheg" in dataset:
#         mc_generator = "powheg"
#     nnlops_w = nnlops.evaluate(events.HTXS.Higgs_pt, events.HTXS.njets30, mc_generator)
#     # print(f'nnlops_weights nnlops_w: {ak.to_numpy(nnlops_w)}')
#     return nnlops_w

def nnlops_weights(Higgs_pt, njets30, parameters, dataset):
    nnlops = NNLOPS_Evaluator(parameters["nnlops_file"])
    if "amc" in dataset:
        mc_generator = "mcatnlo"
    elif "powheg" in dataset:
        mc_generator = "powheg"
    nnlops_w = nnlops.evaluate(Higgs_pt, njets30, mc_generator)
    # print(f'nnlops_weights nnlops_w: {ak.to_numpy(nnlops_w)}')
    return nnlops_w


# Mu SF-------------------------------------------------------------------------

def get_musf_lookup(parameters):
    mu_id_vals = 0
    mu_id_err = 0
    mu_iso_vals = 0
    mu_iso_err = 0
    mu_trig_vals_data = 0
    mu_trig_err_data = 0
    mu_trig_vals_mc = 0
    mu_trig_err_mc = 0

    for scaleFactors in parameters["muSFFileList"]:
        id_file = uproot.open(scaleFactors["id"][0])
        iso_file = uproot.open(scaleFactors["iso"][0])
        # print(f'lepton sf scaleFactors["trig"][0: {scaleFactors["trig"][0]}')
        trig_file = uproot.open(scaleFactors["trig"][0])
        mu_id_vals += id_file[scaleFactors["id"][1]].values() * scaleFactors["scale"]
        mu_id_err += (
            id_file[scaleFactors["id"][1]].variances() ** 0.5 * scaleFactors["scale"]
        )
        mu_id_edges = [
            id_file[scaleFactors["id"][1]].axis(0).edges(),
            id_file[scaleFactors["id"][1]].axis(1).edges(),
        ]
        mu_iso_vals += iso_file[scaleFactors["iso"][1]].values() * scaleFactors["scale"]
        mu_iso_err += (
            iso_file[scaleFactors["iso"][1]].variances() ** 0.5 * scaleFactors["scale"]
        )
        mu_iso_edges = [
            iso_file[scaleFactors["iso"][1]].axis(0).edges(),
            iso_file[scaleFactors["iso"][1]].axis(1).edges(),
        ]
        mu_trig_vals_data += (
            trig_file[scaleFactors["trig"][1]].values() * scaleFactors["scale"]
        )
        mu_trig_vals_mc += (
            trig_file[scaleFactors["trig"][2]].values() * scaleFactors["scale"]
        )
        mu_trig_err_data += (
            trig_file[scaleFactors["trig"][1]].variances() ** 0.5
            * scaleFactors["scale"]
        )
        mu_trig_err_mc += (
            trig_file[scaleFactors["trig"][2]].variances() ** 0.5
            * scaleFactors["scale"]
        )
        mu_trig_edges = [
            trig_file[scaleFactors["trig"][1]].axis(0).edges(),
            trig_file[scaleFactors["trig"][1]].axis(1).edges(),
        ]

    mu_id_sf = dense_lookup.dense_lookup(mu_id_vals, mu_id_edges)
    mu_id_err = dense_lookup.dense_lookup(mu_id_err, mu_id_edges)
    mu_iso_sf = dense_lookup.dense_lookup(mu_iso_vals, mu_iso_edges)
    mu_iso_err = dense_lookup.dense_lookup(mu_iso_err, mu_iso_edges)

    mu_trig_eff_data = dense_lookup.dense_lookup(mu_trig_vals_data, mu_trig_edges)
    # print(f'lepton sf mu_trig_vals_mc: {mu_trig_vals_mc}')
    # print(f'lepton sf mu_trig_edges: {mu_trig_edges}')
    mu_trig_eff_mc = dense_lookup.dense_lookup(mu_trig_vals_mc, mu_trig_edges)
    mu_trig_err_data = dense_lookup.dense_lookup(mu_trig_err_data, mu_trig_edges)
    mu_trig_err_mc = dense_lookup.dense_lookup(mu_trig_err_mc, mu_trig_edges)

    return {
        "mu_id_sf": mu_id_sf,
        "mu_id_err": mu_id_err,
        "mu_iso_sf": mu_iso_sf,
        "mu_iso_err": mu_iso_err,
        "mu_trig_eff_data": mu_trig_eff_data,
        "mu_trig_eff_mc": mu_trig_eff_mc,
        "mu_trig_err_data": mu_trig_err_data,
        "mu_trig_err_mc": mu_trig_err_mc,
    }



# def musf_evaluator(lookups, year, muons):
#     sf = {
#         # "muID_nom": ak.ones_like(muons.pt[:,0]),
#         # "muID_up": ak.ones_like(muons.pt[:,0]),
#         # "muID_down": ak.ones_like(muons.pt[:,0]),
#         # "muIso_nom": ak.ones_like(muons.pt[:,0]),
#         # "muIso_up": ak.ones_like(muons.pt[:,0]),
#         # "muIso_down": ak.ones_like(muons.pt[:,0]),
#         # "muTrig_nom": ak.ones_like(muons.pt[:,0]),
#         # "muTrig_up": ak.ones_like(muons.pt[:,0]),
#         # "muTrig_down": ak.ones_like(muons.pt[:,0]),
#     }

#     for how in ["nom", "up", "down"]:
#         sf[f"trig_num_{how}"] = 1.0
#         sf[f"trig_denom_{how}"] = 1.0

#     pt = muons.pt_raw
#     eta = muons.eta_raw
#     abs_eta = abs(muons.eta_raw)
#     # pt = muons.pt
#     # eta = muons.eta
#     # abs_eta = abs(muons.eta)

#     if "2016" in year:
#         muID_ = lookups["mu_id_sf"](eta, pt)
#         muIso_ = lookups["mu_iso_sf"](eta, pt)
#         muIDerr = lookups["mu_id_err"](eta, pt)
#         muIsoerr = lookups["mu_iso_err"](eta, pt)
#     else:
#         muID_ = lookups["mu_id_sf"](pt, abs_eta)
#         muIso_ = lookups["mu_iso_sf"](pt, abs_eta)
#         muIDerr = lookups["mu_id_err"](pt, abs_eta)
#         muIsoerr = lookups["mu_iso_err"](pt, abs_eta)

#     muTrig_data = lookups["mu_trig_eff_data"](abs_eta, pt)
#     muTrig_mc = lookups["mu_trig_eff_mc"](abs_eta, pt)
#     muTrigerr_data = lookups["mu_trig_err_data"](abs_eta, pt)
#     muTrigerr_mc = lookups["mu_trig_err_mc"](abs_eta, pt)

#     sf["trig_num_nom"] = ak.prod(1.0 - muTrig_data, axis=1)
#     sf["trig_num_up"] = ak.prod(1.0 - (muTrig_data - muTrigerr_data), axis=1)
#     sf["trig_num_down"] = ak.prod(1.0 - (muTrig_data + muTrigerr_data), axis=1)
#     sf["trig_denom_nom"] = ak.prod(1.0 - muTrig_mc, axis=1)
#     sf["trig_denom_up"] = ak.prod(1.0 - (muTrig_mc - muTrigerr_mc), axis=1)
#     sf["trig_denom_down"] = ak.prod(1.0 - (muTrig_mc + muTrigerr_mc), axis=1)

#     # print(f'copperheadV2 lepton sf  sf["trig_num_nom"]: \n {ak.to_numpy(sf["trig_num_nom"])}')
#     # print(f'copperheadV2 lepton sf  sf["trig_num_up"]: \n {ak.to_numpy(sf["trig_num_up"])}')
#     # print(f'copperheadV2 lepton sf  sf["trig_num_down"]: \n {ak.to_numpy(sf["trig_num_down"])}')
#     # print(f'copperheadV2 lepton sf  sf["trig_denom_nom"]: \n {ak.to_numpy(sf["trig_denom_nom"])}')
#     # print(f'copperheadV2 lepton sf  sf["trig_denom_up"]: \n {ak.to_numpy(sf["trig_denom_up"])}')
#     # print(f'copperheadV2 lepton sf  sf["trig_denom_down"]: \n {ak.to_numpy(sf["trig_denom_down"])}')

    
#     sf["muID_nom"] =  ak.prod(muID_, axis=1)
#     sf["muID_up"] = ak.prod(muID_ + muIDerr, axis=1)
#     sf["muID_down"] = ak.prod(muID_ - muIDerr, axis=1)
#     sf["muIso_nom"] = ak.prod(muIso_, axis=1)
#     sf["muIso_up"] = ak.prod(muIso_ + muIsoerr, axis=1)
#     sf["muIso_down"] = ak.prod(muIso_ - muIsoerr, axis=1)
    
#     # print(f'copperheadV2 lepton sf  sf["muID_nom"]: \n {ak.to_numpy(sf["muID_nom"])}')
#     # print(f'copperheadV2 lepton sf  sf["muID_up"]: \n {ak.to_numpy(sf["muID_up"])}')
#     # print(f'copperheadV2 lepton sf  sf["muID_down"]: \n {ak.to_numpy(sf["muID_down"])}')
#     # print(f'copperheadV2 lepton sf  sf["muIso_nom"]: \n {ak.to_numpy(sf["muIso_nom"])}')
#     # print(f'copperheadV2 lepton sf  sf["muIso_up"]: \n {ak.to_numpy(sf["muIso_up"])}')
#     # print(f'copperheadV2 lepton sf  sf["muIso_down"]: \n {ak.to_numpy(sf["muIso_down"])}')

#     #for trig SF
#     for how in ["nom", "up", "down"]:
#         sf[f"trig_num_{how}"] = 1 - sf[f"trig_num_{how}"]
#         sf[f"trig_denom_{how}"] = 1 - sf[f"trig_denom_{how}"]
#         cut = sf[f"trig_denom_{how}"] != 0
#         # sf.loc[cut, f"muTrig_{how}"] = (
#         #     sf.loc[cut, f"trig_num_{how}"] / sf.loc[cut, f"trig_denom_{how}"]
#         # )
#         cut_val = sf[f"trig_num_{how}"] / sf[f"trig_denom_{how}"]
#         # print(f'copperheadV2 lepton sf {how} cut_val: \n {ak.to_numpy(cut_val)}')
#         # print(f'copperheadV2 lepton sf ak.ones_like(muons.pt[:,0]): \n {ak.to_numpy(ak.ones_like(muons.pt[:,0]))}')
#         sf[f"muTrig_{how}"] = ak.where(cut, cut_val, ak.ones_like(muons.pt[:,0]))
#     muID = {"nom": sf["muID_nom"], "up": sf["muID_up"], "down": sf["muID_down"]}
#     muIso = {"nom": sf["muIso_nom"], "up": sf["muIso_up"], "down": sf["muIso_down"]}
#     muTrig = {"nom": sf["muTrig_nom"], "up": sf["muTrig_up"], "down": sf["muTrig_down"]}
#     # print(f'copperheadV2 lepton sf  sf["muTrig_nom"]: \n {(sf["muTrig_nom"])}')
#     return muID, muIso, muTrig

def musf_evaluator(lookups, year, mu1, mu2):
    sf = {
        "muID_nom": ak.ones_like(mu1.pt),
        "muID_up": ak.ones_like(mu1.pt),
        "muID_down": ak.ones_like(mu1.pt),
        "muIso_nom": ak.ones_like(mu1.pt),
        "muIso_up": ak.ones_like(mu1.pt),
        "muIso_down": ak.ones_like(mu1.pt),
        "muTrig_nom": ak.ones_like(mu1.pt),
        "muTrig_up": ak.ones_like(mu1.pt),
        "muTrig_down": ak.ones_like(mu1.pt),
    }

    for how in ["nom", "up", "down"]:
        sf[f"trig_num_{how}"] = 1.0
        sf[f"trig_denom_{how}"] = 1.0

    for mu in [mu1, mu2]:
        pt = mu.pt_raw
        eta = mu.eta_raw
        abs_eta = abs(mu.eta_raw)

        if "2016" in year:
            muID_ = lookups["mu_id_sf"](eta, pt)
            muIso_ = lookups["mu_iso_sf"](eta, pt)
            muIDerr = lookups["mu_id_err"](eta, pt)
            muIsoerr = lookups["mu_iso_err"](eta, pt)
        else:
            muID_ = lookups["mu_id_sf"](pt, abs_eta)
            muIso_ = lookups["mu_iso_sf"](pt, abs_eta)
            muIDerr = lookups["mu_id_err"](pt, abs_eta)
            muIsoerr = lookups["mu_iso_err"](pt, abs_eta)
    
        muTrig_data = lookups["mu_trig_eff_data"](abs_eta, pt)
        muTrig_mc = lookups["mu_trig_eff_mc"](abs_eta, pt)
        muTrigerr_data = lookups["mu_trig_err_data"](abs_eta, pt)
        muTrigerr_mc = lookups["mu_trig_err_mc"](abs_eta, pt)
    
        sf["trig_num_nom"] = sf["trig_num_nom"] * ( 1.0 - muTrig_data)
        sf["trig_num_up"] = sf["trig_num_up"] * (1.0 - (muTrig_data - muTrigerr_data))
        sf["trig_num_down"] = sf["trig_num_down"] * (1.0 - (muTrig_data + muTrigerr_data))
        sf["trig_denom_nom"] = sf["trig_denom_nom"] * (1.0 - muTrig_mc)
        sf["trig_denom_up"] = sf["trig_denom_up"] * (1.0 - (muTrig_mc - muTrigerr_mc))
        sf["trig_denom_down"] = sf["trig_denom_down"] *(1.0 - (muTrig_mc + muTrigerr_mc))
        
        sf["muID_nom"] =  sf["muID_nom"] * (muID_)
        sf["muID_up"] = sf["muID_up"] * (muID_ + muIDerr)
        sf["muID_down"] = sf["muID_down"] * (muID_ - muIDerr)
        sf["muIso_nom"] = sf["muIso_nom"] * (muIso_)
        sf["muIso_up"] = sf["muIso_up"] * (muIso_ + muIsoerr)
        sf["muIso_down"] = sf["muIso_down"] * (muIso_ - muIsoerr)
    
   

    #for trig SF
    for how in ["nom", "up", "down"]:
        sf[f"trig_num_{how}"] = 1 - sf[f"trig_num_{how}"]
        sf[f"trig_denom_{how}"] = 1 - sf[f"trig_denom_{how}"]
        cut = sf[f"trig_denom_{how}"] != 0
        cut_val = sf[f"trig_num_{how}"] / sf[f"trig_denom_{how}"]
        sf[f"muTrig_{how}"] = ak.where(cut, cut_val, ak.ones_like(mu1.pt))
    muID = {"nom": sf["muID_nom"], "up": sf["muID_up"], "down": sf["muID_down"]}
    muIso = {"nom": sf["muIso_nom"], "up": sf["muIso_up"], "down": sf["muIso_down"]}
    muTrig = {"nom": sf["muTrig_nom"], "up": sf["muTrig_up"], "down": sf["muTrig_down"]}
    return muID, muIso, muTrig


# LHE SF-------------------------------------------------------------------------

def lhe_weights(events, dataset, year):
    factor2 = ("dy_m105_160_amc" in dataset) and (("2017" in year) or ("2018" in year))
    if factor2:
        lhefactor = 2.0
    else:
        lhefactor = 1.0
    nLHEScaleWeight = ak.count(events.LHEScaleWeight, axis=1)
    lhe_events = {}
    nLHEScaleWeights_to_iterate = [1, 3, 4, 5, 6, 7, 15, 24, 34]
    max_i = max(nLHEScaleWeights_to_iterate)
    padded_LHEScaleWeight = ak.pad_none(events.LHEScaleWeight, max_i+1)
    for i in nLHEScaleWeights_to_iterate:
        cut = nLHEScaleWeight > i
        cut_ak = nLHEScaleWeight > i
        ones = ak.ones_like(events.Muon.pt[:,0])
        # print(f'copperheadV2 lepton sf ones: \n {ak.to_numpy(ones)}')
        lhe_events[f"LHE{i}"] = ak.where(cut, padded_LHEScaleWeight[:, i], ones)
        # print(f'copperheadV2 lepton sf lhe_events[f"LHE{i}"]: \n {ak.to_numpy(lhe_events[f"LHE{i}"])}')
        # lhe_events[f"LHE{i}"] = 1.0
        # lhe_events.loc[cut, f"LHE{i}"] = ak.to_numpy(events.LHEScaleWeight[cut_ak][:, i])

    cut8 = nLHEScaleWeight > 8
    cut30 = nLHEScaleWeight > 30
    lhe_ren_up = lhe_events["LHE6"] * lhefactor
    lhe_ren_up = ak.where(cut8, (lhe_events["LHE7"] * lhefactor), lhe_ren_up)
    lhe_ren_up = ak.where(cut30, (lhe_events["LHE34"] * lhefactor), lhe_ren_up)
    lhe_ren_down = lhe_events["LHE1"] * lhefactor
    lhe_ren_down = ak.where(cut8, (lhe_events["LHE1"] * lhefactor), lhe_ren_down)
    lhe_ren_down = ak.where(cut30, (lhe_events["LHE5"] * lhefactor), lhe_ren_down)

    lhe_fac_up = lhe_events["LHE4"] * lhefactor
    lhe_fac_up = ak.where(cut8, (lhe_events["LHE5"] * lhefactor), lhe_fac_up)
    lhe_fac_up = ak.where(cut30, (lhe_events["LHE24"] * lhefactor), lhe_fac_up)
    lhe_fac_down = lhe_events["LHE3"] * lhefactor
    lhe_fac_down = ak.where(cut8, (lhe_events["LHE3"] * lhefactor), lhe_fac_down)
    lhe_fac_down = ak.where(cut30, (lhe_events["LHE15"] * lhefactor), lhe_fac_down)

    # print(f'copperheadV2 lepton sf lhe_ren_up: \n {ak.to_numpy(lhe_ren_up)}')
    # print(f'copperheadV2 lepton sf lhe_ren_down: \n {ak.to_numpy(lhe_ren_down)}')
    # print(f'copperheadV2 lepton sf lhe_fac_up: \n {ak.to_numpy(lhe_fac_up)}')
    # print(f'copperheadV2 lepton sf lhe_fac_down: \n {ak.to_numpy(lhe_fac_down)}')
    
    lhe_ren = {"up": lhe_ren_up, "down": lhe_ren_down}
    lhe_fac = {"up": lhe_fac_up, "down": lhe_fac_down}
    return lhe_ren, lhe_fac


# THU SF-------------------------------------------------------------------------
# STXS   TOT,  PTH200,  Mjj60 , Mjj120 , Mjj350 ,
# Mjj700, Mjj1000, Mjj1500,  25, JET01
stxs_acc = {
    200: [0.07, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    201: [0.0744, 0, 0, 0, 0, 0, 0, 0, 0, -0.1649],
    # Jet0
    202: [0.3367, 0, 0, 0, 0, 0, 0, 0, 0, -0.7464],
    # Jet1
    203: [0.0092, 0, -0.6571, 0, 0, 0, 0, 0, -0.0567, 0.0178],
    # Mjj 0-60, PTHjj 0-25
    204: [0.0143, 0, 0.0282, -0.5951, 0, 0, 0, 0, -0.0876, 0.0275],
    # Mjj 60-120, PTHjj 0-25
    205: [0.0455, 0, 0.0902, 0.0946, -0.3791, 0, 0, 0, -0.2799, 0.0877],
    # Mjj 120-350, PTHjj 0-25
    206: [0.0048, 0, -0.3429, 0, 0, 0, 0, 0, 0.0567, 0.0093],
    # Mjj 0-60, PTHjj 25-inf
    207: [0.0097, 0, 0.0192, -0.4049, 0, 0, 0, 0, 0.0876, 0.0187],
    # Mjj 60-120, PTHjj 25-inf
    208: [0.0746, 0, 0.1477, 0.0155, -0.6209, 0, 0, 0, 0.2799, 0.1437],
    # Mjj 120-350, PTHjj 25-inf
    209: [0.0375, 0.1166, 0.0743, 0.078, 0.1039, -0.2757, 0, 0, -0.2306, 0.0723],
    # Mjj 350-700, PTHjj 0-25, pTH 0-200
    210: [0.0985, 0.3062, 0.1951, 0.2048, 0.273, -0.7243, 0, 0, 0.2306, 0.1898],
    # Mjj 350-700, PTHjj 25-inf, pTH 0-200
    211: [0.0166, 0.0515, 0.0328, 0.0345, 0.0459, 0.0773, -0.2473, 0, -0.1019, 0.0319],
    # Mjj 700-1000, PTHjj 0-25, pTH 0-200
    212: [0.0504, 0.1568, 0.0999, 0.1049, 0.1398, 0.2353, -0.7527, 0, 0.1019, 0.0972],
    # Mjj 700-1000, PTHjj 25-inf, pTH 0-200
    213: [
        0.0137,
        0.0426,
        0.0271,
        0.0285,
        0.0379,
        0.0639,
        0.0982,
        -0.2274,
        -0.0842,
        0.0264,
    ],
    # Mjj 1000-1500, PTHjj 0-25, pTH 0-200
    214: [
        0.0465,
        0.1446,
        0.0922,
        0.0967,
        0.1289,
        0.2171,
        0.3335,
        -0.7726,
        0.0842,
        0.0897,
    ],
    # Mjj 1000-1500, PTHjj 25-inf, pTH 0-200
    215: [
        0.0105,
        0.0327,
        0.0208,
        0.0219,
        0.0291,
        0.0491,
        0.0754,
        0.1498,
        -0.0647,
        0.0203,
    ],
    # Mjj 1500-inf, PTHjj 0-25, pTH 0-200
    216: [0.048, 0.1491, 0.095, 0.0998, 0.133, 0.2239, 0.344, 0.6836, 0.0647, 0.0925],
    # Mjj 1500-inf, PTHjj 25-inf, pTH 0-200
    217: [
        0.0051,
        -0.1304,
        0.0101,
        0.0106,
        0.0141,
        0.0238,
        0.0366,
        0.0727,
        -0.0314,
        0.0098,
    ],
    # Mjj 350-700, PTHjj 0-25, pTH 200-inf
    218: [
        0.0054,
        -0.1378,
        0.0107,
        0.0112,
        0.0149,
        0.0251,
        0.0386,
        0.0768,
        0.0314,
        0.0104,
    ],
    # Mjj 350-700, PTHjj 25-inf, pTH 200-inf
    219: [
        0.0032,
        -0.0816,
        0.0063,
        0.0066,
        0.0088,
        0.0149,
        0.0229,
        0.0455,
        -0.0196,
        0.0062,
    ],
    # Mjj 700-1000, PTHjj 0-25, pTH 200-inf
    220: [
        0.0047,
        -0.1190,
        0.0092,
        0.0097,
        0.0129,
        0.0217,
        0.0334,
        0.0663,
        0.0196,
        0.0090,
    ],
    # Mjj 700-1000, PTHjj 25-inf, pTH 200-inf
    221: [
        0.0034,
        -0.0881,
        0.0068,
        0.0072,
        0.0096,
        0.0161,
        0.0247,
        0.0491,
        -0.0212,
        0.0066,
    ],
    # Mjj 1000-1500, PTHjj 0-25, pTH 200-inf
    222: [
        0.0056,
        -0.1440,
        0.0112,
        0.0117,
        0.0156,
        0.0263,
        0.0404,
        0.0802,
        0.0212,
        0.0109,
    ],
    # Mjj 1000-1500, PTHjj 25-inf, pTH 200-inf
    223: [
        0.0036,
        -0.0929,
        0.0072,
        0.0076,
        0.0101,
        0.0169,
        0.026,
        0.0518,
        -0.0223,
        0.0070,
    ],
    # Mjj 1500-inf, PTHjj 0-25, pTH 200-inf
    224: [
        0.0081,
        -0.2062,
        0.016,
        0.0168,
        0.0223,
        0.0376,
        0.0578,
        0.1149,
        0.0223,
        0.0155,
    ]
    # Mjj 1500-inf, PTHjj 25-inf, pTH 200-inf
}
uncert_deltas = [
    14.867,
    0.394,
    9.762,
    6.788,
    7.276,
    3.645,
    2.638,
    1.005,
    20.073,
    18.094,
]
powheg_xsec = {
    200: 273.952,
    201: 291.030,
    202: 1317.635,
    203: 36.095,
    204: 55.776,
    205: 178.171,
    206: 18.839,
    207: 37.952,
    208: 291.846,
    209: 146.782,
    210: 385.566,
    211: 64.859,
    212: 197.414,
    213: 53.598,
    214: 182.107,
    215: 41.167,
    216: 187.823,
    217: 19.968,
    218: 21.092,
    219: 12.496,
    220: 18.215,
    221: 13.490,
    222: 22.044,
    223: 14.220,
    224: 31.565,
}


def stxs_lookups():
    stxs_acc_lookups = {}
    # edges = np.array([])
    # values = np.array([])
    for i in range(10):
        # for k, v in stxs_acc.items():
        #     edges = np.append(edges, k)
        #     values = np.append(values, v[i])
        # print(f'stxs_lookups {i} v[i]: {v[i]}')
        edges = list(stxs_acc.keys())
        values = [v[i] for v in stxs_acc.values()]
        # convert values and edge to np arrays, as ak array doesn't work with dense_lookup initialization
        values = np.array(values) 
        edges = ak.Array(edges)
        # print(f'stxs_lookups {i} edges: {edges}')
        # print(f'stxs_lookups {i} values: {values}')
        stxs_acc_lookups[i] = dense_lookup.dense_lookup(values, [edges])
        # print(f'stxs_lookups stxs_acc_lookups[i]._axes: {stxs_acc_lookups[i]._axes}')
    powheg_xsec_lookup = dense_lookup.dense_lookup(
        np.array(list(powheg_xsec.values())), [np.array(list(powheg_xsec.keys()))]
    )
    return stxs_acc_lookups, powheg_xsec_lookup


# def add_stxs_variations(
#     events, 
#     weights, 
#     parameters
# ):
#     # STXS VBF cross-section uncertainty
#     stxs_acc_lookups, powheg_xsec_lookup = stxs_lookups()
#     for i, name in enumerate(parameters["sths_names"]):
#         # print(f"add_stxs_variations i: {i}, name: {name}")
#         wgt_up = stxs_uncert(
#             i,
#             events.HTXS.stage1_1_fine_cat_pTjet30GeV,
#             1.0,
#             stxs_acc_lookups,
#             powheg_xsec_lookup,
#         )
#         wgt_down = stxs_uncert(
#             i,
#             events.HTXS.stage1_1_fine_cat_pTjet30GeV,
#             -1.0,
#             stxs_acc_lookups,
#             powheg_xsec_lookup,
#         )
#         thu_wgts = {"up": wgt_up, "down": wgt_down}
#         weights.add_weight("THU_VBF_" + name, thu_wgts, how="only_vars")


def add_stxs_variations(
    events, 
    weights,
    parameters
):
    # STXS VBF cross-section uncertainty
    stxs_acc_lookups, powheg_xsec_lookup = stxs_lookups()
    for i, name in enumerate(parameters["sths_names"]):
        # print(f"add_stxs_variations i: {i}, name: {name}")
        wgt_up = stxs_uncert(
            i,
            events.HTXS.stage1_1_fine_cat_pTjet30GeV,
            1.0,
            stxs_acc_lookups,
            powheg_xsec_lookup,
        )
        wgt_down = stxs_uncert(
            i,
            events.HTXS.stage1_1_fine_cat_pTjet30GeV,
            -1.0,
            stxs_acc_lookups,
            powheg_xsec_lookup,
        )
        thu_wgts = {"up": wgt_up, "down": wgt_down}
        weights.add("THU_VBF_" + name, 
                    weight=ak.ones_like(thu_wgts["up"]),
                    weightUp=thu_wgts["up"],
                    weightDown=thu_wgts["down"]
        )
    return
    
def stxs_uncert(source, event_STXS, Nsigma, stxs_acc_lookups, powheg_xsec_lookup):
    """
    NOTE: source numbering seems arbitrary, I gotta ask Dmitry about this
    Moreover, source is always < 10, so idk what the use case is 
    """
    # vbf_uncert_stage_1_1
    # return a single weight for a given souce
    if source < 10: # idk why we have this
        # print(f'stxs_lookups source: {source}')
        # print(f'stxs_lookups stxs_acc_lookups.keys(): {stxs_acc_lookups.keys()}')
        # print(f'stxs_uncert stxs_acc_lookups[source]: {stxs_acc_lookups[source]}')
        # print(f'stxs_uncert event_STXS: {event_STXS}')
        # print(f'stxs_uncert stxs_acc_lookups[source](event_STXS): {stxs_acc_lookups[source](event_STXS)}')
        # print(f'stxs_uncert uncert_deltas[source]: {uncert_deltas[source]}')
        delta_var = stxs_acc_lookups[source](event_STXS) * uncert_deltas[source]
        ret = ak.ones_like(event_STXS, dtype="float") + Nsigma * (
            delta_var / powheg_xsec_lookup(event_STXS)
        )
        return ret
    else:
        return ak.zeros_like(event_STXS, dtype="float")

# PDF SF-------------------------------------------------------------------------

# def add_pdf_variations(events, weights, config, dataset):
#     if "2016" in config["year"]:
#         max_replicas = 0
#         if "dy" in dataset:
#             max_replicas = 100
#         elif "ewk" in dataset:
#             max_replicas = 33
#         else:
#             max_replicas = 100
#         pdf_wgts = events.LHEPdfWeight[:, 0 : config["n_pdf_variations"]]

#         #---------------- No idea why output instead of weights
#         for i in range(100):
#             if (i < max_replicas) and do_pdf:
#                 output[f"pdf_mcreplica{i}"] = pdf_wgts[:, i]
#             else:
#                 output[f"pdf_mcreplica{i}"] = np.nan
#         #--------------------------------------------
    
#     else:
#         # pdf_wgts = events.LHEPdfWeight[:, 0 : config["n_pdf_variations"]][0]
#         pdf_wgts = events.LHEPdfWeight[:, 0 : config["n_pdf_variations"]]
#         # pdf_wgts = np.array(pdf_wgts)
#         # print(f"add_pdf_variations pdf_wgts: {pdf_wgts}")
#         pdf_std = ak.std(pdf_wgts, axis=1)
#         pdf_vars = {
#             # "up": (1 + 2 * pdf_wgts.std()),
#             # "down": (1 - 2 * pdf_wgts.std()),
#             "up": (1 + 2 * pdf_std),
#             "down": (1 - 2 * pdf_std),
#         }
#         # print(f"add_pdf_variations pdf_vars up: {ak.to_numpy(pdf_vars['up'])}")
#         # print(f"add_pdf_variations pdf_vars down: {ak.to_numpy(pdf_vars['down'])}")
#         weights.add_weight("pdf_2rms", pdf_vars, how="only_vars")

def add_pdf_variations(events, config, dataset):
    if "2016" in config["year"]:
        max_replicas = 0
        if "dy" in dataset:
            max_replicas = 100
        elif "ewk" in dataset:
            max_replicas = 33
        else:
            max_replicas = 100
        pdf_wgts = events.LHEPdfWeight[:, 0 : config["n_pdf_variations"]]

        #---------------- No idea why output instead of weights comment out for now
        # for i in range(100):
        #     if (i < max_replicas):
        #         output[f"pdf_mcreplica{i}"] = pdf_wgts[:, i]
        #     else:
        #         output[f"pdf_mcreplica{i}"] = np.nan
        #--------------------------------------------
    
    else:
        # pdf_wgts = events.LHEPdfWeight[:, 0 : config["n_pdf_variations"]][0]
        pdf_wgts = events.LHEPdfWeight[:, 0 : config["n_pdf_variations"]]
        # pdf_wgts = np.array(pdf_wgts)
        # print(f"add_pdf_variations pdf_wgts: {pdf_wgts}")
    
    pdf_std = ak.std(pdf_wgts, axis=1)
    pdf_vars = {
        # "up": (1 + 2 * pdf_wgts.std()),
        # "down": (1 - 2 * pdf_wgts.std()),
        "up": (1 + 2 * pdf_std),
        "down": (1 - 2 * pdf_std),
    }
    # pdf_wgts = events.LHEPdfWeight[:, 0 : config["n_pdf_variations"]][0]
    # # print(f"pdf_wgts: {pdf_wgts.compute()}")
    # pdf_std = ak.std(pdf_wgts, axis=0)
    # print(f"pdf_std: {pdf_std.compute()}")
    # pdf_vars = {
    #     # "up": (1 + 2 * pdf_wgts.std()),
    #     # "down": (1 - 2 * pdf_wgts.std()),
    #     "up": (1 + 2 * pdf_std* ak.ones_like(events.LHEPdfWeight[:,0])),
    #     "down": (1 - 2 * pdf_std* ak.ones_like(events.LHEPdfWeight[:,0])),
    # }
    # # print(f"add_pdf_variations pdf_vars up: {ak.to_numpy(pdf_vars['up'])}")
    # # print(f"add_pdf_variations pdf_vars down: {ak.to_numpy(pdf_vars['down'])}")
    return pdf_vars
        



# QGL SF-------------------------------------------------------------------------

def qgl_weights_keepDim(jet1, jet2, njets, isHerwig):
    """
    We assume that event filtering/selection has been already applied
    params:
    jet1 = leading pt jet variable if doens't exist, it's padded with None
    jet2 = subleading pt jet variable if doens't exist, it's padded with None
    """

    qgl1 = get_qgl_weights(jet1, isHerwig)
    qgl1 = ak.fill_none(qgl1, value=1.0)
    qgl2 = get_qgl_weights(jet2, isHerwig)

    
    qgl_nom = (qgl1*qgl2)
    ones = ak.ones_like(qgl1) # qgl1 is picked bc we assume there's no none values in it. ones_like function copies None values as well
    qgl_nom = ak.where((njets==1), ones, qgl_nom)  # 1D array


    njet_selection = njets > 2 # think this is a bug, but have to double check
    qgl_mean = dak.map_partitions(np.mean, qgl_nom[njet_selection], keepdims=True)
    qgl_nom = qgl_nom/ qgl_mean
    qgl_nom = ak.fill_none(qgl_nom, value=1.0) # we got rid of jet2==None case, but jet1 could still be None


    qgl_down = ak.ones_like(qgl_nom, dtype="float")

    wgts = {"nom": qgl_nom, "up": qgl_nom * qgl_nom, "down": qgl_down}
    return wgts

def qgl_weights(jet1, jet2, njets, isHerwig):
    """
    We assume that event filtering/selection has been already applied
    params:
    jet1 = leading pt jet variable if doens't exist, it's padded with None
    jet2 = subleading pt jet variable if doens't exist, it's padded with None
    """
    # qgl = pd.DataFrame(index=output.index, columns=["wgt", "wgt_down"]).fillna(1.0)
    
    
    # qgl1 = get_qgl_weights(jet1, isHerwig).fillna(1.0)
    # qgl2 = get_qgl_weights(jet2, isHerwig).fillna(1.0)
    # qgl.wgt *= qgl1 * qgl2
    qgl1 = get_qgl_weights(jet1, isHerwig)
    qgl1 = ak.fill_none(qgl1, value=1.0)
    qgl2 = get_qgl_weights(jet2, isHerwig)
    # print(f"qgl_weights jet1: {qgl1.compute()}")
    # print(f"qgl_weights jet2: {qgl2.compute()}")
    
    qgl_nom = (qgl1*qgl2)
    ones = ak.ones_like(qgl1) # qgl1 is picked bc we assume there's no none values in it. ones_like function copies None values as well
    qgl_nom = ak.where((njets==1), ones, qgl_nom)  # 1D array

    # print(f"qgl_weights qgl_nom b4: {qgl_nom.compute()}")
    # qgl.wgt[variables.njets == 1] = 1.0 # fill_none does this
    # qgl.wgt = qgl.wgt / qgl.wgt[selected].mean()
    # selected = output.event_selection & (njets > 2)
    njet_selection = njets > 2 # think this is a bug, but have to double check
    sum = ak.sum(qgl_nom[njet_selection], axis=None)
    count = ak.count(qgl_nom[njet_selection], axis=None)
    qgl_mean = sum/count
    # qgl_nom = qgl_nom/ qgl_mean
    qgl_nom = qgl_nom/ (ak.ones_like(qgl_nom)*qgl_mean)
    # print(f"qgl_weights qgl_nom after: {ak.to_numpy(qgl_nom)}")
    qgl_nom = ak.fill_none(qgl_nom, value=1.0) # we got rid of jet2==None case, but jet1 could still be None
    # print(f"qgl_weights qgl_nom after after: {ak.to_numpy(qgl_nom)}")
    # print(f"qgl_weights qgl_nom[njet_selection]: {ak.to_numpy(qgl_nom[njet_selection])}")

    # print(f"qgl_nom: {ak.to_numpy(qgl_nom.compute())}")

    qgl_down = ak.ones_like(qgl_nom, dtype="float")

    wgts = {"nom": qgl_nom, "up": qgl_nom * qgl_nom, "down": qgl_down}
    return wgts

def qgl_weights_eager(jet1, jet2, njets, isHerwig):
    """
    Eager version of calculating qgl weights. This is a workaround
    dask_awkward.mean() not supported over axis=0 or axis=None
    """
    qgl1 = get_qgl_weights(jet1, isHerwig)
    qgl1 = ak.fill_none(qgl1, value=1.0)
    qgl2 = get_qgl_weights(jet2, isHerwig)
    # qgl2 = ak.fill_none(qgl2, value=1.0)
    qgl_nom = (qgl1*qgl2)
    # qgl_nom = ak.fill_none(qgl_nom, value=1.0)
    print(f"(qgl1*qgl2): {ak.to_numpy((qgl1*qgl2).compute())}")
    print(f"ak.sum(njets==1): {ak.sum(njets==1).compute()}")
    print(f"(njets==1): {ak.to_numpy((njets==1).compute())}")
    ones = ak.ones_like(qgl1) # qgl1 is picked bc we assume there's no none values in it. ones_like function copies None values as well
    qgl_nom = ak.where((njets==1), ones, qgl_nom) 
    print(f"qgl_nom after njet==1 selection: {ak.to_numpy((qgl_nom).compute())}")
    return qgl_nom

def get_qgl_weights(jet, isHerwig):
    # df = pd.DataFrame(index=jet.index, columns=["weights"])
    qgl_weights = ak.ones_like(jet.pt, dtype="float")

    wgt_mask = (jet.partonFlavour != 0) & (abs(jet.eta) < 2) & (jet.qgl > 0)
    light = wgt_mask & (abs(jet.partonFlavour) < 4)
    gluon = wgt_mask & (jet.partonFlavour == 21)

    qgl = jet.qgl

    if isHerwig:
        # df.weights[light] = (
        #     1.16636 * qgl[light] ** 3
        #     - 2.45101 * qgl[light] ** 2
        #     + 1.86096 * qgl[light]
        #     + 0.596896
        # )
        # df.weights[gluon] = (
        #     -63.2397 * qgl[gluon] ** 7
        #     + 111.455 * qgl[gluon] ** 6
        #     - 16.7487 * qgl[gluon] ** 5
        #     - 72.8429 * qgl[gluon] ** 4
        #     + 56.7714 * qgl[gluon] ** 3
        #     - 19.2979 * qgl[gluon] ** 2
        #     + 3.41825 * qgl[gluon]
        #     + 0.919838
        # )
        light_val =  (
            1.16636 * qgl ** 3
            - 2.45101 * qgl ** 2
            + 1.86096 * qgl
            + 0.596896
        )
        # qgl_weights = ak.where(light, light_val, qgl_weights)
        gluon_val = (
            -63.2397 * qgl ** 7
            + 111.455 * qgl ** 6
            - 16.7487 * qgl ** 5
            - 72.8429 * qgl ** 4
            + 56.7714 * qgl ** 3
            - 19.2979 * qgl ** 2
            + 3.41825 * qgl
            + 0.919838
        )
        # qgl_weights = ak.where(gluon, gluon_val, qgl_weights)
    else:
        # df.weights[light] = (
        #     -0.666978 * qgl[light] ** 3
        #     + 0.929524 * qgl[light] ** 2
        #     - 0.255505 * qgl[light]
        #     + 0.981581
        # )
        # df.weights[gluon] = (
        #     -55.7067 * qgl[gluon] ** 7
        #     + 113.218 * qgl[gluon] ** 6
        #     - 21.1421 * qgl[gluon] ** 5
        #     - 99.927 * qgl[gluon] ** 4
        #     + 92.8668 * qgl[gluon] ** 3
        #     - 34.3663 * qgl[gluon] ** 2
        #     + 6.27 * qgl[gluon]
        #     + 0.612992
        # )
        light_val = (
            -0.666978 * qgl ** 3
            + 0.929524 * qgl ** 2
            - 0.255505 * qgl
            + 0.981581
        )
        # qgl_weights = ak.where(light, light_val, qgl_weights)
        gluon_val= (
            -55.7067 * qgl ** 7
            + 113.218 * qgl ** 6
            - 21.1421 * qgl ** 5
            - 99.927 * qgl ** 4
            + 92.8668 * qgl ** 3
            - 34.3663 * qgl ** 2
            + 6.27 * qgl
            + 0.612992
        )
        # qgl_weights = ak.where(gluon, gluon_val, qgl_weights)
    
    # apply two filters, first light, then gluon
    qgl_weights = ak.where(light, light_val, qgl_weights)
    qgl_weights = ak.where(gluon, gluon_val, qgl_weights)
    return qgl_weights

# Btag SF-------------------------------------------------------------------------

def btag_weights_jsonKeepDim(processor, systs, jets, weights, bjet_sel_mask, btag_file):
    """
    We assume jets to be non padded jet that has passed the base jet selection.
    I don't think jets need to be sorted after JEC for this to work, however
    """
    # btag = pd.DataFrame(index=bjet_sel_mask.index)
    btag_jet_selection = abs(jets.eta) < 2.4
    jets = ak.to_packed(jets[btag_jet_selection])
    jets["pt"] = ak.where((jets.pt > 1000), 1000, jets.pt) # clip max pt
    
    
    btag_json=btag_file["deepJet_shape"]
    correctionlib_out = btag_json.evaluate(
        "central",
        jets.hadronFlavour,
        abs(jets.eta),
        jets.pt,
        jets.btagDeepFlavB,
    )

    btag_wgt = ak.prod(correctionlib_out, axis=1) # for events with no qualified jets(empty row), the value is 1.0
    btag_wgt = ak.where((btag_wgt < 0.01), 1.0, btag_wgt)
    # print(f"btag_wgt b4 normalization: {ak.to_numpy(btag_wgt.compute())}")

    flavors = {
        0: ["jes", "lf", "lfstats1", "lfstats2"],
        # 1: ["jes", "lf", "lfstats1", "lfstats2"],
        # 2: ["jes", "lf", "lfstats1", "lfstats2"],
        # 3: ["jes", "lf", "lfstats1", "lfstats2"],
        4: ["cferr1", "cferr2"],
        5: ["jes", "hf", "hfstats1", "hfstats2"],
        # 21: ["jes", "lf", "lfstats1", "lfstats2"],
    }# printiing the correctionlib input description returns: "hadron flavor definition: 5=b, 4=c, 0=udsg", so corretionlib lookup table only accepts flavours of 0, 4 or 5
    
    btag_syst = {}
    for sys in systs:


        btag_wgt_up = ak.ones_like(jets.pt)
        btag_wgt_down = ak.ones_like(jets.pt)
        # 
        

        for flavor, f_syst in flavors.items():
            if sys in f_syst:
                # print(f"sys: {sys}")
                # print(f"flavor: {flavor}")
                btag_mask = (abs(jets.hadronFlavour)) == flavor #& (abs(jets.eta) < 2.4))
                # enforce input hadronFlavour to match the target, otherwise, the lookup table will fail
                dummy_flavor = flavor*ak.ones_like(jets.hadronFlavour)
                hadronFlavour = ak.where(btag_mask, jets.hadronFlavour, dummy_flavor)
                sys_wgts =  btag_json.evaluate(
                    f"up_{sys}",
                    hadronFlavour,
                    abs(jets.eta),
                    jets.pt,
                    jets.btagDeepB,
                )
                btag_wgt_up = ak.where(btag_mask, sys_wgts, btag_wgt_up)

                    
                sys_wgts =  btag_json.evaluate(
                    f"down_{sys}",
                    hadronFlavour,
                    abs(jets.eta),
                    jets.pt,
                    jets.btagDeepB,
                )
                btag_wgt_down = ak.where(btag_mask, sys_wgts, btag_wgt_down)

        btag_wgt_up = ak.prod(btag_wgt_up, axis=1)
        btag_wgt_down = ak.prod(btag_wgt_down, axis=1)

        btag_syst[sys] = {"up": btag_wgt_up, "down": btag_wgt_down}

    weights = weights.weight()
    sum_before = dak.map_partitions(ak.sum, weights, keepdims=True)
    sum_after = dak.map_partitions(ak.sum, weights*btag_wgt, keepdims=True)
    btag_wgt = btag_wgt * sum_before / sum_after
    return btag_wgt, btag_syst

def btag_weights_json(processor, systs, jets, weights, bjet_sel_mask, btag_file):
    """
    We assume jets to be non padded jet that has passed the base jet selection.
    I don't think jets need to be sorted after JEC for this to work, however
    """
    # btag = pd.DataFrame(index=bjet_sel_mask.index)
    btag_jet_selection = abs(jets.eta) < 2.4
    jets = ak.to_packed(jets[btag_jet_selection])
    jets["pt"] = ak.where((jets.pt > 1000), 1000, jets.pt) # clip max pt
    
    
    # btag_json=btag_file["deepCSV_shape"]
    btag_json=btag_file["deepJet_shape"]
    correctionlib_out = btag_json.evaluate(
        "central",
        jets.hadronFlavour,
        abs(jets.eta),
        jets.pt,
        jets.btagDeepFlavB,
    )
    # print(f"correctionlib_out: {correctionlib_out.compute()}")
    # correctionlib_out = ak.pad_none(correctionlib_out, target=1)
    btag_wgt = ak.prod(correctionlib_out, axis=1) # for events with no qualified jets(empty row), the value is 1.0
    btag_wgt = ak.where((btag_wgt < 0.01), 1.0, btag_wgt)
    # print(f"btag_wgt b4 normalization: {ak.to_numpy(btag_wgt.compute())}")

    flavors = {
        0: ["jes", "lf", "lfstats1", "lfstats2"],
        # 1: ["jes", "lf", "lfstats1", "lfstats2"],
        # 2: ["jes", "lf", "lfstats1", "lfstats2"],
        # 3: ["jes", "lf", "lfstats1", "lfstats2"],
        4: ["cferr1", "cferr2"],
        5: ["jes", "hf", "hfstats1", "hfstats2"],
        # 21: ["jes", "lf", "lfstats1", "lfstats2"],
    }# printiing the correctionlib input description returns: "hadron flavor definition: 5=b, 4=c, 0=udsg", so corretionlib lookup table only accepts flavours of 0, 4 or 5
    
    btag_syst = {}
    for sys in systs:

        # jets[f"btag_{sys}_up"] = 1.0
        # jets[f"btag_{sys}_down"] = 1.0
        # btag[f"{sys}_up"] = 1.0
        # btag[f"{sys}_down"] = 1.0
        btag_wgt_up = ak.ones_like(jets.pt)
        btag_wgt_down = ak.ones_like(jets.pt)
        # 
        

        for flavor, f_syst in flavors.items():
            if sys in f_syst:
                # print(f"sys: {sys}")
                # print(f"flavor: {flavor}")
                btag_mask = (abs(jets.hadronFlavour)) == flavor #& (abs(jets.eta) < 2.4))
                # enforce input hadronFlavour to match the target, otherwise, the lookup table will fail
                dummy_flavor = flavor*ak.ones_like(jets.hadronFlavour)
                hadronFlavour = ak.where(btag_mask, jets.hadronFlavour, dummy_flavor)
                sys_wgts =  btag_json.evaluate(
                    f"up_{sys}",
                    hadronFlavour,
                    abs(jets.eta),
                    jets.pt,
                    jets.btagDeepB,
                )
                # print(f"sys_wgts up: {sys_wgts.compute()}")
                btag_wgt_up = ak.where(btag_mask, sys_wgts, btag_wgt_up)
                # jets.loc[btag_mask, f"btag_{sys}_up"] = onedimeval(partial(btag_json[0].evaluate,
                #     f"up_{sys}"),
                #     jets.hadronFlavour[btag_mask].values,
                #     abs(jets.eta)[btag_mask].values,
                #     jets.pt[btag_mask].values,
                #     jets.btagDeepB[btag_mask].values,
                    
                # )
                sys_wgts =  btag_json.evaluate(
                    f"down_{sys}",
                    hadronFlavour,
                    abs(jets.eta),
                    jets.pt,
                    jets.btagDeepB,
                )
                # print(f"sys_wgts down: {sys_wgts.compute()}")
                btag_wgt_down = ak.where(btag_mask, sys_wgts, btag_wgt_down)
                # jets.loc[btag_mask, f"btag_{sys}_down"] = onedimeval(partial(btag_json[0].evaluate,
                #     f"down_{sys}"),
                #     jets.hadronFlavour[btag_mask].values,
                #     abs(jets.eta)[btag_mask].values,
                #     jets.pt[btag_mask].values,
                #     jets.btagDeepB[btag_mask].values,
                    
                # )
        btag_wgt_up = ak.prod(btag_wgt_up, axis=1)
        btag_wgt_down = ak.prod(btag_wgt_down, axis=1)
        # print(f"btag_wgt_up: {ak.to_numpy(btag_wgt_up.compute())}")
        # print(f"btag_wgt_down: {ak.to_numpy(btag_wgt_down.compute())}")
        btag_syst[sys] = {"up": btag_wgt_up, "down": btag_wgt_down}

    weights = weights.weight()
    sum_before = ak.sum(weights, axis=None)
    sum_after = ak.sum(weights*btag_wgt, axis=None)
    btag_wgt = btag_wgt * sum_before / sum_after
    # print(f"btag_wgt after normalization: {ak.to_numpy(btag_wgt.compute())}")
    return btag_wgt, btag_syst
    # sum_before = weights.df["nominal"][bjet_sel_mask].sum()
    # sum_after = (
    #     weights.df["nominal"][bjet_sel_mask]
    #     .multiply(btag.wgt[bjet_sel_mask], axis=0)
    #     .sum()
    # )
    # btag.wgt = btag.wgt * sum_before / sum_after
    # #print(f"len btag.wgt {len(btag.wgt)}")
    # #print(f"len jets3 {len(jets)}")
    # return btag.wgt, btag_syst

# def btag_weights_csv(processor, lookup, systs, jets, weights, bjet_sel_mask):

#     btag = pd.DataFrame(index=bjet_sel_mask.index)
#     jets = jets[abs(jets.eta) < 2.4]
#     jets.loc[jets.pt > 1000.0, "pt"] = 1000.0

#     jets["btag_wgt"] = lookup.eval(
#         "central",
#         jets.hadronFlavour.values,
#         abs(jets.eta.values),
#         jets.pt.values,
#         jets.btagDeepB.values,
#         True,
#     )
#     btag["wgt"] = jets["btag_wgt"].prod(level=0)
#     btag["wgt"] = btag["wgt"].fillna(1.0)
#     btag.loc[btag.wgt < 0.01, "wgt"] = 1.0

#     flavors = {
#         0: ["jes", "hf", "lfstats1", "lfstats2"],
#         1: ["jes", "hf", "lfstats1", "lfstats2"],
#         2: ["jes", "hf", "lfstats1", "lfstats2"],
#         3: ["jes", "hf", "lfstats1", "lfstats2"],
#         4: ["cferr1", "cferr2"],
#         5: ["jes", "lf", "hfstats1", "hfstats2"],
#         21: ["jes", "hf", "lfstats1", "lfstats2"],
#     }

#     btag_syst = {}
#     for sys in systs:
#         jets[f"btag_{sys}_up"] = 1.0
#         jets[f"btag_{sys}_down"] = 1.0
#         btag[f"{sys}_up"] = 1.0
#         btag[f"{sys}_down"] = 1.0

#         for f, f_syst in flavors.items():
#             if sys in f_syst:
#                 btag_mask = abs(jets.hadronFlavour) == f
#                 jets.loc[btag_mask, f"btag_{sys}_up"] = lookup.eval(
#                     f"up_{sys}",
#                     jets.hadronFlavour[btag_mask].values,
#                     abs(jets.eta)[btag_mask].values,
#                     jets.pt[btag_mask].values,
#                     jets.btagDeepB[btag_mask].values,
#                     True,
#                 )
#                 jets.loc[btag_mask, f"btag_{sys}_down"] = lookup.eval(
#                     f"down_{sys}",
#                     jets.hadronFlavour[btag_mask].values,
#                     abs(jets.eta)[btag_mask].values,
#                     jets.pt[btag_mask].values,
#                     jets.btagDeepB[btag_mask].values,
#                     True,
#                 )

#         btag[f"{sys}_up"] = jets[f"btag_{sys}_up"].prod(level=0)
#         btag[f"{sys}_down"] = jets[f"btag_{sys}_down"].prod(level=0)
#         btag_syst[sys] = {"up": btag[f"{sys}_up"], "down": btag[f"{sys}_down"]}

#     sum_before = weights.df["nominal"][bjet_sel_mask].sum()
#     sum_after = (
#         weights.df["nominal"][bjet_sel_mask]
#         .multiply(btag.wgt[bjet_sel_mask], axis=0)
#         .sum()
#     )
#     btag.wgt = btag.wgt * sum_before / sum_after

#     return btag.wgt, btag_syst

# jet puid weight

def get_jetpuid_weights(evaluator, year, jets, pt_name, jet_puid_opt, jet_puid):
    if year == "2016preVFP":
        yearname = "UL2016APV"
    elif year == "2016postVFP":
        yearname = "UL2016"
    elif year == "2017":
        yearname = "UL2017"
    elif year == "2018":
        yearname = "UL2018"
    # define 1D array of ones for other arrays to copy off of
    ones = ak.fill_none(
        ak.pad_none(jets.pt, target=1, clip=True)[:,0], 
        value= 1.0
    )
    ones = ak.ones_like(ones)
    # if True:
    if "2017corrected" in jet_puid_opt:
        print("doing the 2017corrected jetPUID method !")
        h_eff_name_L = f"h2_eff_mc{yearname}_L"
        h_sf_name_L = f"h2_eff_sf{yearname}_L"
        h_eff_name_T = f"h2_eff_mc{yearname}_T"
        h_sf_name_T = f"h2_eff_sf{yearname}_T"
        puid_eff_L = evaluator[h_eff_name_L](jets[pt_name], jets.eta)
        puid_sf_L = evaluator[h_sf_name_L](jets[pt_name], jets.eta)
        puid_eff_T = evaluator[h_eff_name_T](jets[pt_name], jets.eta)
        puid_sf_T = evaluator[h_sf_name_T](jets[pt_name], jets.eta)

        jets_passed_L = (
            (jets[pt_name] > 25)
            & (jets[pt_name] < 50)
            & jet_puid
            & ((abs(jets.eta) < 2.6) | (abs(jets.eta) > 3.0))
        )
        jets_failed_L = (
            (jets[pt_name] > 25)
            & (jets[pt_name] < 50)
            & (~jet_puid)
            & ((abs(jets.eta) < 2.6) | (abs(jets.eta) > 3.0))
        )
        # tight PUID applied to all jets irrespective of pT https://indico.cern.ch/event/850507/contributions/3574693/subcontributions/291453/attachments/1932754/3201723/Raffaele_23_10_2019.pdf
        jets_passed_T = (
            (jets[pt_name] > 25)
            & jet_puid
            & ((abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0))
        )
        jets_failed_T = (
            (jets[pt_name] > 25)
            & (~jet_puid)
            & ((abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0))
        )
        # original start ---------------------------------------------------------
        # obtain the Loose jet puid 
        # oneminuspuid_eff_L = 1.0 - puid_eff_L
        # pMC_L = (
        #     ak.prod(ak.to_packed(puid_eff_L[jets_passed_L]), axis=1) *
        #     (ak.prod(ak.to_packed(oneminuspuid_eff_L[jets_failed_L]), axis=1))
        # )
        
        # oneminuspuid_effNSF_L = 1.0 - puid_eff_L * puid_sf_L
        # pData_L = (
        #     ak.prod(ak.to_packed(puid_eff_L[jets_passed_L]), axis=1)
        #     * ak.prod(ak.to_packed(puid_sf_L[jets_passed_L]), axis=1)
        #     * ak.prod(ak.to_packed(oneminuspuid_effNSF_L[jets_failed_L]), axis=1)
        # )
        
        # # obtain the Tight jet puid 
        # oneminuspuid_eff_T = 1.0 - puid_eff_T
        # pMC_T = (
        #     ak.prod(ak.to_packed(puid_eff_T[jets_passed_T]), axis=1) * 
        #     (ak.prod(ak.to_packed(oneminuspuid_eff_T[jets_failed_T]), axis=1))
        # )
        # oneminuspuid_effNSF_T = 1.0 - puid_eff_T * puid_sf_T
        # pData_T = (
        #     ak.prod(ak.to_packed(puid_eff_T[jets_passed_T]), axis=1)
        #     * ak.prod(ak.to_packed(puid_sf_T[jets_passed_T]), axis=1)
        #     * ak.prod(ak.to_packed((oneminuspuid_effNSF_T[jets_failed_T])), axis=1)
        # )
        # original end ---------------------------------------------------------
        
        # obtain the Loose jet puid 
        pMC_failed_L = ak.ones_like(ones)
        pMC_passed_bare_L = ak.prod(ak.to_packed(puid_eff_L[jets_passed_L==True]), axis=1)
        pMC_failed_bare_L = ak.prod(ak.to_packed(oneminuspuid_eff_L[jets_failed_L==True]), axis=1)
        pMC_passed_L = ak.ones_like(ones)
        pMC_failed_L = ak.ones_like(ones)
        pMC_passed_L = pMC_passed_bare_L
        pMC_failed_L = pMC_failed_bare_L
        # print(f"pMC_failed_L: {ak.to_numpy(pMC_failed_L.compute())}")
        pSF_L = ak.ones_like(ones)
        pfailSF_L = ak.ones_like(ones)
        pSF_bare_L = ak.prod(ak.to_packed(puid_sf_L[jets_passed_L==True]), axis=1)
        pfailSF_bare_L = ak.prod(ak.to_packed(oneminus_eff_sf_L[jets_failed_L==True]), axis=1)
        pSF_L = pSF_bare_L
        pfailSF_L = pfailSF_bare_L
        pMC_L = pMC_passed_L * pMC_failed_L
        pData_L = pMC_passed_L * pSF_L * pfailSF_L

        # obtain the Tight jet puid 
        pMC_failed_T = ak.ones_like(ones)
        pMC_passed_bare_T = ak.prod(ak.to_packed(puid_eff_T[jets_passed_T==True]), axis=1)
        pMC_failed_bare_T = ak.prod(ak.to_packed(oneminuspuid_eff_T[jets_failed_T==True]), axis=1)
        pMC_passed_T = ak.ones_like(ones)
        pMC_failed_T = ak.ones_like(ones)
        pMC_passed_T = pMC_passed_bare_T
        pMC_failed_T = pMC_failed_bare_T
        # print(f"pMC_failed_T: {ak.to_numpy(pMC_failed_T.compute())}")
        pSF_T = ak.ones_like(ones)
        pfailSF_T = ak.ones_like(ones)
        pSF_bare_T = ak.prod(ak.to_packed(puid_sf_T[jets_passed_T==True]), axis=1)
        pfailSF_bare_T = ak.prod(ak.to_packed(oneminus_eff_sf_T[jets_failed_T==True]), axis=1)
        pSF_T = pSF_bare_T
        pfailSF_T = pfailSF_bare_T
        pMC_T = pMC_passed_T * pMC_failed_T
        pData_T = pMC_passed_T * pSF_T * pfailSF_T


        # merge Loose and Tight values, they're supposed to be exlusive to one another, with None replaced with one, so just multiplying them should be good
        pMC = pMC_L * pMC_T
        pData = pData_L * pData_T
        puid_weight = ak.ones_like(ones)
        # print(f"puid_weight b4: {puid_weight}")
        # print(f"(pData/pMC): {ak.to_numpy((pData/pMC).compute())}")
        puid_weight = ak.where((pMC != 0), (pData/pMC), puid_weight)
        puid_weight = ak.to_packed(puid_weight) # this saves a bit of memory from a very superficial testing

    else:
        wp_dict = {"loose": "L", "medium": "M", "tight": "T"}
        wp = wp_dict[jet_puid_opt]
        h_eff_name = f"h2_eff_mc{yearname}_L"
        h_sf_name = f"h2_eff_sf{yearname}_L"
        jetpt = jets[pt_name]
        jeteta = jets.eta
        puid_eff = evaluator[h_eff_name](jetpt, jeteta)
        puid_sf = evaluator[h_sf_name](jetpt, jeteta)
        # print(f"ones: {ak.to_numpy(ones.compute())}")
        # jets["puid_eff"] = puid_eff
        # jets["oneminuspuid_eff"] = 1.0-puid_eff
        # jets["puid_sf"] = puid_sf
        # jets["eff_sf"] = puid_sf * puid_eff
        # jets["oneminus_eff_sf"] = 1.0-(puid_sf * puid_eff)
        oneminuspuid_eff = 1.0-puid_eff
        eff_sf = puid_sf * puid_eff
        oneminus_eff_sf = 1.0-(puid_sf * puid_eff)

        # apply pt< 50 cut as instructed on https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL
        jets_passed = (jets[pt_name] > 25) & (jets[pt_name] < 50) & jet_puid
        jets_failed = (jets[pt_name] > 25) & (jets[pt_name] < 50) & (~jet_puid)

        pMC_failed = ak.ones_like(ones)
        pMC_passed_bare = ak.prod(ak.to_packed(puid_eff[jets_passed==True]), axis=1)
        pMC_failed_bare = ak.prod(ak.to_packed(oneminuspuid_eff[jets_failed==True]), axis=1)
        pMC_passed = ak.ones_like(ones)
        pMC_failed = ak.ones_like(ones)
        pMC_passed = pMC_passed_bare
        pMC_failed = pMC_failed_bare
        # print(f"pMC_failed: {ak.to_numpy(pMC_failed.compute())}")
        pSF = ak.ones_like(ones)
        pfailSF = ak.ones_like(ones)
        pSF_bare = ak.prod(ak.to_packed(puid_sf[jets_passed==True]), axis=1)
        pfailSF_bare = ak.prod(ak.to_packed(oneminus_eff_sf[jets_failed==True]), axis=1)
        pSF = pSF_bare
        pfailSF = pfailSF_bare
        pMC = pMC_passed * pMC_failed
        pData = pMC_passed * pSF * pfailSF
        # print(f"pMC: {ak.to_numpy((pMC).compute())}")
        # print(f"pData: {ak.to_numpy((pData).compute())}")
        puid_weight = ak.ones_like(ones)
        # print(f"puid_weight b4: {puid_weight}")
        # print(f"(pData/pMC): {ak.to_numpy((pData/pMC).compute())}")
        puid_weight = ak.where((pMC != 0), (pData/pMC), puid_weight)
        puid_weight = ak.to_packed(puid_weight) # this saves a bit of memory from a very superficial testing
        # puid_weight[pMC != 0] = np.divide(pData[pMC != 0], pMC[pMC != 0])
        # print(f"puid_weight after: {puid_weight}")
        # print(f"puid_weight: {ak.to_numpy(puid_weight.compute())}")
    return puid_weight