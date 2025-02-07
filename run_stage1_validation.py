import hist
import awkward as ak
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import glob
import os
import mplhep as hep
from histogram.variable import variables_lookup, Entry
from hist.intervals import poisson_interval
import argparse
import re
import dask_awkward as dak
from distributed import LocalCluster, Client


# print(f"variables_lookup: {variables_lookup}")
style = hep.style.CMS
style["mathtext.fontset"] = "cm"
style["mathtext.default"] = "rm"
plt.style.use(style)

stat_err_opts = {
    "step": "post",
    "label": "Stat. unc.",
    "hatch": "//////",
    "facecolor": "none",
    "edgecolor": (0, 0, 0, 0.5),
    "linewidth": 0,
}
ratio_err_opts = {"step": "post", "facecolor": (0, 0, 0, 0.3), "linewidth": 0}


# data_samples = [
#     "data_A",
#     "data_B",
#     "data_C",
#     "data_D",
# ]
# bkg_samples = [
#     "dy_M-50",
#     "dy_M-100To200",
#     "ttjets_dl",
#     "ttjets_sl",
#     # "st_tw_top",
#     # "st_tw_antitop", 
#     # "ww_2l2nu",
#     # "wz_3lnu",
#     # "wz_2l2q",
#     # "wz_1l1nu2q",
#     # "zz",
#     # "ewk_lljj_mll50_mjj120",
# ]
# sig_samples = [
#     "ggh_powheg",
#     "vbf_powheg"
# ]
# mc_samples = [
#     *bkg_samples,
#     *sig_samples
# ]

variables = [
    'mu1_pt',
    'mu2_pt',
    'mu1_eta',
    'mu2_eta',
    # # 'mu1_phi',
    # # 'mu2_phi',
    # # # 'mu1_iso',
    # # # 'mu2_iso',
    # # 'mu1_pt_over_mass',
    # # 'mu2_pt_over_mass',
    # "dimuon_mass",
    # # "dimuon_ebe_mass_res",
    # # "dimuon_ebe_mass_res_rel",
    # "dimuon_pt",
    # # "dimuon_pt_log",
    # # "dimuon_eta",
    # # "dimuon_phi",
    # # "dimuon_dEta",
    # # "dimuon_dPhi",
    # # "dimuon_dR",
    "dimuon_cos_theta_cs",
    "dimuon_phi_cs",
    # "jet1_pt",
    # "jet1_eta",
    # # # "jet1_rap",
    # # # "jet1_phi",
    # # # "jet1_qgl",
    # # # "jet1_jetId",
    # # # "jet1_puId",
    # "jet2_pt",
    # "jet2_eta",
    # # # "jet2_rap",
    # # # "jet2_phi",
    # # "jet2_qgl",
    # # "jet2_jetId",
    # # # "jet2_puId",
    # "jj_mass",
    # # # # "jj_mass_log",
    # # # "jj_pt",
    # # "jj_eta",
    # # "jj_phi",
    # "jj_dEta",
    # "jj_dPhi",
    # # "mmj1_dEta",
    # # "mmj1_dPhi",
    # # "mmj1_dR",
    # # "mmj2_dEta",
    # # "mmj2_dPhi",
    # # "mmj2_dR",
    # "mmj_min_dEta",
    # "mmj_min_dPhi",
    # # "mmjj_pt",
    # # "mmjj_eta",
    # "mmjj_phi",
    # "mmjj_mass",
    # "rpt",
    # "zeppenfeld",
    # "njets",
]
regions = [
    # "z_peak",
    # "h_sidebands",
    # "h_peak",
    "signal_region",
]
default_load_path = "/depot/cms/users/yun79/results/stage1/test"
default_save_path = "./validation/figs"

# channels = ["vbf", "ggh", "DY", "TT+ST", "Data"]
channels =  ["Data", "MC", "Bkg", "Sig"]
# plot_group_config= {
#     # 'stack': ['DY', 'EWK', 'TT+ST', 'VV', 'VVV'],
#     # 'step': ['VBF', 'ggH'], 
#     # 'errorbar': ['Data']
#     'stack': ['Bkg'],
#     'step': ['Sig'], 
#     'errorbar': ['Data']
# }

parameters = {}
parameters["grouping"] = {
    "data_A": "Data",
    "data_B": "Data",
    "data_C": "Data",
    "data_D": "Data",
    "data_E": "Data",
    "data_F": "Data",
    "data_G": "Data",
    "data_H": "Data",
    "dy_M-50": "DY",
    "dy_M-100To200": "DY", # this gives really weird plot with this on in Z peak region
    "ttjets_dl": "TT+ST",
    "ttjets_sl": "TT+ST",
    "ggh_powheg": "ggH",
    "vbf_powheg": "VBF",
}
parameters["plot_group"] = {
    "stack": ["DY", "EWK", "TT+ST", "VV", "VVV"],
    "step": ["VBF", "ggH"],
    "errorbar": ["Data"],
}
# parameters["plot_group"] = {
#     "Data" : "errorbar",
#     "ggH" : "step",
#     "VBF" : "step",
#     "DY" : "stack",
#     "TT+ST" : "stack",
# }

def get_plottable(
    variable: str,
    # file_list: list,
    load_path: str,
    entry,
    region: str,
    # grouping: str,
    ):
    # print(f"mc_samples: {mc_samples}")
    # if len(samples)==0:
    #     print(f"empty samples for {channel}")
    #     return histogram
    # loop over data samples
    
    h_var = variables_lookup[variable]
    h_dict = {}
    for group in entry.groups:
        samples = [e for e, g in entry.entry_dict.items() if (group == g)]
        print(f"samples: {samples}")
        histogram = (
            hist.Hist.new
            .Reg(
                h_var.nbins,
                 h_var.xmin, 
                 h_var.xmax, 
                 name=h_var.name, 
                 # label=h_var.caption
            )
            .Double()
        )
        histogram_no_weight = (
            hist.Hist.new
            .Reg(
                h_var.nbins,
                 h_var.xmin, 
                 h_var.xmax, 
                 name=h_var.name, 
                 # label=h_var.caption
            )
            .Double()
        )
        histogram_w2 = ( # weight sq histogram for stat err calculation
            hist.Hist.new
            .Reg(
                h_var.nbins,
                 h_var.xmin, 
                 h_var.xmax, 
                 name=h_var.name, 
                 # label=h_var.caption
            )
            .Double()
        )
        for sample in samples:
            # if sample != "data_A":
            #     continue
            full_load_path = load_path+f"/{sample}/*/*.parquet"
            print(f"full_load_path: {full_load_path}")
            try:
                df = dak.from_parquet(full_load_path) 
            except:
                print(f"couldn't find parquet file in {full_load_path}")
                continue
            # print(f"df['jet1_pt']: {df['jet1_pt'].compute()}")
            # print(f"ak_arr: {ak_arr}")
            # print(f"len(ak_arr): {len(ak_arr)}")
            # print(f"type(ak_arr): {type(ak_arr)}")
            # raise ValueError
            # file_list = glob.glob(load_path+f"/{sample}/*.parquet")
            # print(f"file_list: {file_list}")
            # df = None
            # for file in file_list:
            # for df in dfs:
            #     print(f"df: {df}")
            #     print(f"type(df): {type(df)}")
            #     print(f"df[h_var.name]: {df[h_var.name]}")
            #     # if sample in file:
            #     #     print(f"{sample} matched with {file}")
            #     #     df = pd.read_csv(file)
            #     #     break
            #     # if df is None:
            #     #     print(f"{sample} not in stage1 results")
            #     #     continue
            #     # if var not in df.columns:
            #     # if var not in df.keys():
            #     #     print(f"{variable} not found in df keys")
            #     #     continue
            
            # get region array
            print(f"region: {region}")
            # print(f'df["z_peak"]: {df["z_peak"].compute()}')
            # print(f'df["h_peak"]: {df["h_peak"].compute()}')
            # print(f'df["h_sidebands"]: {df["h_sidebands"].compute()}')
            if region == "z_peak":
                region_filter = ak.fill_none(df["z_peak"], value = False)
            elif region == "h_sidebands":
                region_filter = ak.fill_none(df["h_sidebands"], value = False)
            elif region == "h_peak":
                region_filter = ak.fill_none(df["h_peak"], value = False)
            else: # signal region"
                print("signal region activated")
                region_filter = (ak.fill_none(df["h_sidebands"], value = False) | ak.fill_none(df["h_peak"], value = False) )

            vbf_cut = ak.fill_none(df["vbf_cut"], value  = False)
            region_filter = region_filter & ~vbf_cut
            # refine df
            # print(f"vbf_cut : {vbf_cut.compute()}")
            # print(f"(region_filter).compute() : {(region_filter).compute()}")
            # print(f"ak.sum(region_filter).compute() : {ak.sum(region_filter).compute()}")
            df = df[region_filter]
            # print(f"variable: {variable}")
            
            # print(f'df["fraction"][0]: {df["fraction"][0]}')
            # print(f"df[variable][region_filter]: {df[variable][region_filter].compute()}")
            # print(f"df[variable]: {df[variable]}")
            #fill histogram
            if "Data" in entry.groups: # data
                weight = 1/df["fraction"].compute()
            else: # MC
                # weight=1
                # wgt_load_path = load_path+f"/{sample}/{'weight_nominal'}/*.parquet"
                # weight = ak.from_parquet(wgt_load_path)
                # weight = df["weight_nominal"].compute()
                weight = df["weight_nominal"].compute() # * 1/df["fraction"].compute()
                # weight = df["weight_nominal"] / df["weight_nominal"].sum()
                print(f'df["fraction"].compute(): {df["fraction"].compute()}')
            fill_val = ak.fill_none(df[variable], value=-999).compute()
            # print(f"fill_val: {fill_val}")
            to_fill = {
                h_var.name: fill_val,
                # h_var.name: ak.fill_none(ak_arr, value=-999),
                # "grouping": grouping,
            }
            histogram.fill(**to_fill, weight=weight)
            histogram_no_weight.fill(**to_fill)
            histogram_w2.fill(**to_fill, weight=weight*weight)
        # print(f"histogram.sum(): {histogram.sum()}")
        # h_list.append(histogram)
        # need to sort by histogram sum later
        h_dict[histogram.sum()] = (histogram, histogram_no_weight, histogram_w2, group) 
    
    # print(f"h_dict b4: {h_dict}")
    h_dict  = dict(sorted(h_dict.items()))
    # print(f"h_dict after: {h_dict}")
    h_list = []
    h_n_w_list = []
    h_w2_list = []
    labels = []
    for histogram, histogram_no_weight, histogram_w2, group in h_dict.values():
        h_list.append(histogram)
        h_n_w_list.append(histogram_no_weight)
        h_w2_list.append(histogram_w2)
        labels.append(group)
    # return h_list
    
    yerr = np.sqrt(sum(h_n_w_list).values()) if entry.yerr else None # only data's yerr is not None
    
    if yerr is not None:
        print(f"yerr b4: {yerr}")
        fraction = df["fraction"].compute()[0] # the ak arr is just a scalar repeated N times
        yerr = yerr/fraction # increase the y_err proportional to the data weight increase
        print(f"yerr after: {yerr}")
    return h_list, yerr, h_w2_list, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-frac",
    "--fraction",
    dest="fraction",
    default=None,
    action="store",
    help="change fraction of steps of the data",
    )
    parser.add_argument(
    "-in_str",
    "--input_string",
    dest="input_string",
    default=None,
    action="store",
    help="string representation of options, in the format of Ratio_{Y or N}/LogY_{Y or N}/ShowLumi_{Y or N}/Status_{work or prelim}",
    )
    parser.add_argument(
    "-load",
    "--load_path",
    dest="load_path",
    default=None,
    action="store",
    help="path value where stage1 output files are saved in",
    )
    parser.add_argument(
    "-save",
    "--save_path",
    dest="save_path",
    default=None,
    action="store",
    help="path value where validation plots will be saved in",
    )
    args = parser.parse_args()
    fraction_str = (args.fraction).replace('.', '_')
    if args.load_path is None:
        load_path = default_load_path
    else:
        load_path = args.load_path
       
    load_path = load_path + f"f{fraction_str}"
    print(f"load_path: {load_path}")

    if args.save_path is None: 
        save_path = default_save_path
    else:
        save_path = args.save_path
    
    # file_list = glob.glob(load_path+"/*.csv")
    # print(f"file_list: {file_list}")
    # loop over variables

    plotsize = 8
    ratio_plot_size = 0.25
    fontsize=20
    year = "2018"
    plot_ratio = re.findall(r"Ratio_.", args.input_string)
    plot_ratio = [str.replace("Ratio_", "") for str in plot_ratio]
    plot_ratio = plot_ratio[0] # get rid of list
    plot_ratio = True if plot_ratio == "Y" else False
    print(f"plot_ratio: {plot_ratio}")
    
    log_y_scale = re.findall(r"LogY_.", args.input_string)
    log_y_scale = [str.replace("LogY_", "") for str in log_y_scale]
    log_y_scale = log_y_scale[0] # get rid of list
    log_y_scale = True if log_y_scale == "Y" else False # if false, linear scale
    print(f"log_y_scale: {log_y_scale}")
    
    show_lumi = re.findall(r"ShowLumi_.", args.input_string)
    show_lumi = [str.replace("ShowLumi_", "") for str in show_lumi]
    show_lumi = show_lumi[0] # get rid of list
    show_lumi = True if show_lumi == "Y" else False 
    print(f"show_lumi: {show_lumi}")
    
    status = re.findall(r"\bStatus_.*.", args.input_string)
    status = [str.replace("Status_", "") for str in status]
    status = status[0] # get rid of list
    print(f"status: {status}")

    cluster = LocalCluster()
    cluster.adapt(minimum=8, maximum=31) #min: 8 max: 32
    client = Client(cluster)
    print("Local scale Client created")
    
    # # temporary
    # variation = "nominal"

    # slicer = {"region": region, "channel": channel, "variation": variation, "category": category}
    

    # label: ['TT+ST', 'DY']
    # entry.stack: True
    # entry.histtype: fill
    # entry.plot_opts: {'alpha': 0.8, 'edgecolor': (0, 0, 0)}
    
    for region in regions:
        for var in variables:
            print(f"plotting: {var}")
            fig = plt.figure()
            
            if plot_ratio:
                fig.set_size_inches(plotsize * 1.2, plotsize * (1 + ratio_plot_size))
                gs = fig.add_gridspec(
                    2, 1, height_ratios=[(1 - ratio_plot_size), ratio_plot_size], hspace=0.07
                )
                # Top panel: Data/MC
                ax1 = fig.add_subplot(gs[0])
            else:
                fig, ax1 = plt.subplots()
                fig.set_size_inches(plotsize, plotsize)
                
            

            entries = {entry_type: Entry(entry_type, parameters) for entry_type in parameters["plot_group"].keys()}
            for entry in entries.values():
                print(f"entry.histtype: {entry.histtype}")
                print(f"entry.stack: {entry.stack}")
                print(f"entry.labels: {entry.labels}")
                print(f"entry.groups: {entry.groups}")
                # print(f"entry.entry_dict: {entry.entry_dict}")
                # print(f"entry.entry_list: {entry.entry_list}")
                # grouping = parameters["grouping"][data_samples[0]]
                # dists, yerr, dists_w2, labels = get_plottable(var, file_list, entry, region)
                dists, yerr, dists_w2, labels = get_plottable(var, load_path, entry, region)
                # print(f"sum(dists): {sum(dists)}")
                # print(f"sum(dists).values(): {(sum(dists).values())}")
                # yerr = np.sqrt(sum(dists).values()) if entry.yerr else None # only data's yerr is not None
                hep.histplot(
                    dists,
                    # label=entry.groups,
                    label=labels,
                    ax=ax1,
                    yerr=yerr,
                    stack=entry.stack,
                    histtype=entry.histtype,
                    **entry.plot_opts,
                )
                # Bkg MC errors
                if entry.entry_type == "stack":    
                    total_bkg = sum(dists).values()
                    total_sumw2 = sum(dists_w2).values()
                    if sum(total_bkg) > 0:
                        err = poisson_interval(total_bkg, total_sumw2)
                        ax1.fill_between(
                            x=dists[0].axes[0].edges,
                            y1=np.r_[err[0, :], err[0, -1]],
                            y2=np.r_[err[1, :], err[1, -1]],
                            **stat_err_opts,
                        )
            # # Bottom panel: Data/MC ratio plot
            if plot_ratio:
                ax2 = fig.add_subplot(gs[1], sharex=ax1)
                num = []
                den = []
        
                if len(entries["errorbar"].entry_list) > 0:
                    # get Data yields
                    # num_hists, _, _, _ = get_plottable(var, file_list, entries["errorbar"], region)
                    num_hists, yerr, _, _ = get_plottable(var, load_path, entries["errorbar"], region)
                    if len(num_hists) > 0:
                        num = sum(num_hists).values()
        
                if len(entries["stack"].entry_list) > 0:
                    # get MC yields and sumw2
                    # den_hists, _, den_hists_w2, _ = get_plottable(var, file_list, entries["stack"], region)
                    den_hists, _, den_hists_w2, _ = get_plottable(var, load_path, entries["stack"], region)
                    if len(den_hists) > 0:
                        edges = den_hists[0].axes[0].edges
                        den = sum(den_hists).values()  # total MC
                        den_w2 = sum(den_hists_w2).values()
        
                # print(f"num: {num}")
                # print(f"den: {den}")
                if len(num) * len(den) > 0:
                    # compute Data/MC ratio
                    ratio = np.divide(num, den)
                    print(f"ratio: {ratio}")
                    """
                    yerr = np.zeros_like(num)
                    yerr[den > 0] = np.sqrt(num[den > 0]) / den[den > 0]
                    """
                    yerr[den < 0] = 0
                    yerr[den > 0] = yerr[den > 0] / den[den > 0]
                    hep.histplot( # ratio plot
                        ratio,
                        bins=edges,
                        ax=ax2,
                        yerr=yerr,
                        histtype="errorbar",
                        **entries["errorbar"].plot_opts,
                    )
        
                if sum(den) > 0:
                    # compute MC uncertainty
                    unity = np.ones_like(den)
                    w2 = np.zeros_like(den)
                    w2[den > 0] = den_w2[den > 0] / den[den > 0] ** 2
                    den_unc = poisson_interval(unity, w2)
                    ax2.fill_between(
                        edges,
                        np.r_[den_unc[0], den_unc[0, -1]],
                        np.r_[den_unc[1], den_unc[1, -1]],
                        label="Stat. unc.",
                        **ratio_err_opts,
                    )
        
            # setting up axis labels    
            ax1.set_ylabel("Events", loc="center")
            if plot_ratio:
                ax1.set_xlabel("")
                ax1.tick_params(axis="x", labelbottom=False)
                ax2.axhline(1, ls="--")
                ax2.set_ylim([0.5, 1.5])
                ax2.set_ylabel("Data/MC", loc="center")
                ax2.set_xlabel(variables_lookup[var].caption, loc="right")
                ax2.legend(prop={"size": "x-small"})
            else:
                ax1.set_xlabel(variables_lookup[var].caption, loc="right")
    
            if status == "work":
                label = "Work in progress"
            elif status == "prelim":
                label = "Preliminary"
            else:
                label = ""

            if show_lumi:
                integrated_lumi = 59970.0 /1000 # get this from config in the future
            else:
                integrated_lumi = None
            hep.cms.label(ax=ax1, data=True, label=label, year=year, lumi=integrated_lumi, fontsize=fontsize)
            if log_y_scale: 
                ax1.set_yscale("log")
                ax1.set_ylim(0.001, 1e9)
            ax1.legend(prop={"size": "x-small"})
            # save figure
            local_save_path = save_path + f"/f{fraction_str}"
            if not os.path.exists(local_save_path):
                os.makedirs(local_save_path)
            fig.savefig(local_save_path+f"/V{var}_{region}.png")
    