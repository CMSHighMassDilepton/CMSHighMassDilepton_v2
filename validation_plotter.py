import os
import json
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import argparse
from coffea.lumi_tools import LumiMask
import dask_awkward as dak
hep.style.use("CMS")

# available_processes = ["Data2018C", "DYJetsToLL", "WW", "WZ", "ZZ", "TTTo2L2Nu", "TTToSemiLeptonic", "TTToHadronic"]
available_processes = ["data_A", "data_B", "data_C", "data_D", "dy_M-50", "dy_M-100To200", "ttjets_dl", "ttjets_sl", "vbf_powheg", "ggh_powheg"]
available_processes = ["data_A", "data_B", "data_C", "dy_M-100To200", "ttjets_dl", "ttjets_sl", "vbf_powheg", "ggh_powheg"]

# available_processes = ["data_A", "dy_M-100To200", "vbf_powheg", "ggh_powheg"]

# # Parser setup
parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", choices=["all"] + available_processes, default = "all", type=str, help="specifying which plot you want") # changeable in future... 
parser.add_argument("--groupProcesses", action="store_true", default = "false", help="saying which processes do you want to group")
parser.add_argument(
    "-frac",
    "--fraction",
    dest="fraction",
    default=None,
    action="store",
    help="change fraction of steps of the data",
)
args = parser.parse_args()
# dataset_name = args.datasetd
load_path = "/depot/cms/users/yun79/results/stage1/test_full3"
fraction_str = (args.fraction).replace('.', '_')
load_path = load_path + f"/f{fraction_str}"
# Load plot settings from JSON file
with open("./histogram/plot_settings.json", "r") as file:
    plot_settings = json.load(file)

# Directory creation for plots (directory_path = "plots/" + args.process)
directory_path = "./validation/figs"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created successfully.")



# Dictionary for histograms and binnings
histogram_dict = {}
binning_dict = {}


for process in available_processes:
    full_load_path = load_path+f"/{process}/*/*.parquet"
    # print(f"full_load_path: {full_load_path}")

    events = dak.from_parquet(full_load_path) 

    if "data" in process.lower():
        weights = np.ones_like(events["mu1_pt"].compute())
    else:
        weights = events["weight_nominal"].compute() 

    for plot_name, settings in plot_settings.items():
        if settings["variable"] not in events.fields:
            continue
        hist, _ = np.histogram(getattr(events, settings["variable"]).compute(), weights=weights, bins=np.linspace(*settings["binning_linspace"]))

        if plot_name not in histogram_dict:
            histogram_dict[plot_name] = {}

        if process not in histogram_dict[plot_name]:
            histogram_dict[plot_name][process] = []

        histogram_dict[plot_name][process].append(hist)

        if plot_name not in binning_dict:
            binning_dict[plot_name] = {}

        if process not in binning_dict[plot_name]:
            binning_dict[plot_name][process] = np.linspace(*settings["binning_linspace"])

# group_EW_processes = ["ZZ","WW", "WZ"]
# group_EW_processes = ["dy_M-100To200", "dy_M-50"]
# group_Top_processes = ["TTTo2L2Nu", "TTToSemiLeptonic", "TTToHadronic", ]
group_data_processes = ["data_A", "data_B", "data_C", "data_D",]
group_DY_processes = ["dy_M-100To200", "dy_M-50"]
group_Top_processes = ["ttjets_dl", "ttjets_sl"]
# Plotting
for plot_name, histograms in histogram_dict.items():
    print('INFO: Now making plot for', plot_name, '...')
    binning = np.linspace(*plot_settings[plot_name].get("binning_linspace"))
    do_stack = not plot_settings[plot_name].get("density")
    data_hist = np.zeros(len(binning)-1)
    if args.groupProcesses:
        hist_DY = np.zeros(len(binning)-1)
        hist_Top = np.zeros(len(binning)-1)
    hist_ggh= None
    hist_vbf= None
    hists_to_plot = []
    labels = []

    for process, histogram in histograms.items():
        histogram = np.asarray(histogram)
        # print(f"histogram.shape: {histogram.shape}")
        histogram = histogram.flatten()
        # Fix later, hist should not be a list in the first place and should be 60 1d and not (60,1)
        if "data" in process:
            data_hist += histogram
        else: # MC
            if "ggh" in process:
                hist_ggh = histogram
            elif  "vbf" in process:
                hist_vbf = histogram
            else:
                if args.groupProcesses:
                    if process in group_Top_processes:
                        hist_Top += histogram
                    elif process in group_DY_processes:
                        hist_DY += histogram
                else:
                    if process == "DYJetsToLL":
                        hists_to_plot.append(histogram)
                        labels.append(process)
                    else:
                        hists_to_plot.append(histogram)
                        labels.append(process)
    
    if args.groupProcesses:
        hists_to_plot.append(hist_Top)
        labels.append('Top')
        hists_to_plot.append(hist_DY)
        labels.append('DY')
        
        
    
    # colours = hep.style.cms.cmap_petroff[0:3]
    colours = hep.style.cms.cmap_petroff[0:2]
    # print(f"colours: {colours}")
    # print(f"labels: {labels}")
    
    fig, (ax_main, ax_ratio) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    fig.subplots_adjust(hspace=0.0)
    # obtain fraction weight, this should be the same for all processes and rows
    fraction_weight = 1/events.fraction[0].compute() # directly apply these to np hists
    # fraction_weight = 100
    print(f"fraction_weight: {(fraction_weight)}")
    #------------------------------------------
    mc_sum_histogram = np.sum(np.asarray(hists_to_plot), axis=0) # to be used in ratio plot later
    
    for hist in hists_to_plot:
        hist *= fraction_weight #hists are pointers so this gets
    hep.histplot(hists_to_plot, bins=binning, 
                 stack=do_stack, histtype='fill', 
                 label=labels, 
                 sort='label_r', 
                 color=colours, 
                 density=plot_settings[plot_name].get("density"), 
                 ax=ax_main)
    if hist_ggh is not None:
        hist_ggh = hist_ggh*fraction_weight
        hep.histplot(hist_ggh, bins=binning, 
                     stack=do_stack, histtype='step', 
                     label="ggH", 
                     sort='label_r', 
                     # color =  hep.style.cms.cmap_petroff[5],
                     color =  "black",
                     density=plot_settings[plot_name].get("density"), 
                     ax=ax_main)
    if hist_vbf is not None:
        hist_vbf = hist_vbf*fraction_weight
        hep.histplot(hist_vbf, bins=binning, 
                     stack=do_stack, histtype='step', 
                     label="VBF", 
                     sort='label_r', 
                     # color =  hep.style.cms.cmap_petroff[4],
                     color = "red",
                     density=plot_settings[plot_name].get("density"), 
                     ax=ax_main)
    
    
    # data_rel_err = np.zeros_like(data_hist)
    # data_rel_err[data_hist>0] = np.sqrt(data_hist)**(-1) # poisson err / value == inverse sqrt()
    #apply fraction weight to data hist and yerr
    data_err = np.sqrt(data_hist) # get yerr b4 fraction weight is applied
    data_hist = data_hist*fraction_weight
    data_err = data_err*fraction_weight
    hep.histplot(data_hist, xerr=True, yerr=data_err,
                 bins=binning, stack=False, histtype='errorbar', color='black', 
                 label='Data', density=plot_settings[plot_name].get("density"), ax=ax_main)
    ax_main.set_ylabel(plot_settings[plot_name].get("ylabel"))
    ax_main.set_yscale('log')
    ax_main.set_ylim(0.01, 1e9)
    ax_main.legend()
    
    if args.groupProcesses:
        # sum_histogram = np.sum(np.asarray(hists_to_plot), axis=0)
        mc_yerr = np.sqrt(mc_sum_histogram)
        mc_yerr *= fraction_weight # re apply fraction weights
        mc_sum_histogram  *= fraction_weight # re apply fraction weights
        ratio_hist = np.zeros_like(data_hist)
        ratio_hist[mc_sum_histogram>0] = data_hist[mc_sum_histogram>0]/  mc_sum_histogram[mc_sum_histogram>0]
        # add rel unc of data and mc by quadrature
        rel_unc_ratio = np.sqrt((mc_yerr/mc_sum_histogram)**2 + (data_err/data_hist)**2)
        ratio_err = rel_unc_ratio*ratio_hist
        # ratio_hist = np.asarray(histogram_dict[plot_name]["data_A"]).flatten() / (sum_histogram + np.finfo(float).eps)
        # # Adding relative sqrtN Poisson uncertainty for now, should be improved when using the hist package
        # # print(type(np.sqrt(np.asarray(histogram_dict[plot_name]["data_A"]).flatten())))
        # # print(type(np.asarray(histogram_dict[plot_name]["data_A"]).flatten()))
        # rel_unc = np.sqrt(np.asarray(histogram_dict[plot_name]["data_A"]).flatten()) / np.asarray(histogram_dict[plot_name]["data_A"]).flatten()
        # # print("flag")
        # rel_unc *= ratio_hist
        # rel_unc[rel_unc < 0] = 0 # Not exactly sure why we have negative values, but this solves it for the moment
        # print("flag2")
        # hep.histplot(ratio_hist, 
        #              bins=binning, histtype='errorbar', yerr=rel_unc, 
        #              color='black', label='Ratio', ax=ax_ratio)
        print(f" (mc_yerr/mc_sum_histogram).shape: {(mc_yerr/mc_sum_histogram).shape}")
        hep.histplot(ratio_hist, 
                     bins=binning, histtype='errorbar', yerr=ratio_err, 
                     color='black', label='Ratio', ax=ax_ratio)
        # hep.histplot(np.ones_like(ratio_hist), 
        #              bins=binning, histtype='fill', yerr=(mc_yerr/mc_sum_histogram).flatten(), 
        #              color='blue', label='MC err', ax=ax_ratio)
        # print("flag3")
        ax_ratio.axhline(1, color='gray', linestyle='--')
        # ax_ratio.set_xlabel(plot_settings[plot_name].get("xlabel"), usetex=True)
        ax_ratio.set_xlabel(plot_settings[plot_name].get("xlabel"))
        ax_ratio.set_ylabel('Data / MC')
        ax_ratio.set_xlim(binning[0], binning[-1])
        ax_ratio.set_ylim(0.6, 1.4)
    
    # Decorating with CMS label
    # hep.cms.label(data=True, loc=0, label="Private Work", com=13, lumi=round(int_lumi, 1), ax=ax_main)
    hep.cms.label(data=True, loc=0, label="Private Work", com=13, ax=ax_main)
    
    # Saving with special name
    filename = directory_path+f"/{plot_name}"
    # filename = directory_path+f"test"
    #if args.groupProcesses:
    if plot_settings[plot_name].get("density"):
        filename += "_normalized"
    else: 
        filename += "_stacked"
    filename += ".pdf"
    plt.savefig(filename)
    plt.clf()