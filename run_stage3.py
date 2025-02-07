import time
import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client
from omegaconf import OmegaConf

from typing import Tuple, List, Dict
import ROOT as rt
from src.lib.fit_functions import MakeFEWZxBernDof3
import argparse
import os

def normalizeFlatHist(x: rt.RooRealVar,rooHist: rt.RooDataHist) -> rt.RooDataHist :
    """
    Takes rootHistogram and returns a new copy with histogram values normalized to sum to one
    """
    x_name = x.GetName()
    # copy nbins and range from, rooHist, but make it empty, and fill with flat distribution
    THist = rooHist.createHistogram(x_name).Clone("clone") # clone it just in case
    THist.Reset()
    nEntries = 100000
    # print(f"THist.GetXaxis().GetXmin(): {THist.GetXaxis().GetXmin()}")
    # print(f"THist.GetXaxis().GetXmax(): {THist.GetXaxis().GetXmax()}")
    values = np.random.uniform(
        low=THist.GetXaxis().GetXmin(), 
        high=THist.GetXaxis().GetXmax(), 
        size=nEntries
    )
    weight = np.ones_like(values)
    THist.FillN(nEntries, values, weight)
    THist.Scale(1.0 / THist.Integral()) # normalize
    print(f"THist.Integral(): {THist.Integral()}")
    normalizedHist_name = rooHist.GetName() + "_normalized"
    roo_hist_normalized = rt.RooDataHist(normalizedHist_name, normalizedHist_name, rt.RooArgSet(x), THist) 
    return roo_hist_normalized

def normalizeRooHist(x: rt.RooRealVar,rooHist: rt.RooDataHist) -> rt.RooDataHist :
    """
    Takes rootHistogram and returns a new copy with histogram values normalized to sum to one
    """
    x_name = x.GetName()
    THist = rooHist.createHistogram(x_name).Clone("clone") # clone it just in case
    THist.Scale(1/THist.Integral())
    print(f"THist.Integral(): {THist.Integral()}")
    normalizedHist_name = rooHist.GetName() + "_normalized"
    roo_hist_normalized = rt.RooDataHist(normalizedHist_name, normalizedHist_name, rt.RooArgSet(x), THist) 
    return roo_hist_normalized
    
def plotBkgByCoreFunc(mass:rt.RooRealVar, model_dict_by_coreFunction: Dict, rooHist_list, save_path: str):
    """
    takes the dictionary of all Bkg RooAbsPdf models grouped by same corefunctions, and plot them
    in the frame() of mass and saves the plots on a given directory path
    """
    # make the save_path directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    color_list = [
        rt.kGreen,
        rt.kBlue,
        rt.kRed,
        rt.kOrange,
        rt.kViolet,
    ]
    for core_type, coreFunction_list in model_dict_by_coreFunction.items():
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        frame = mass.frame()
        frame.SetTitle(f"Normalized Shape Plot of {core_type} PDFs")
        frame.SetXTitle(f"Dimuon Mass (GeV)")
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        
        # print(f"normalized_hist integral: {normalized_hist.sum(False)}")
        for ix in range(len(coreFunction_list)):
        # for ix in [0,4]:
            # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
            color = color_list[ix]
            hist = rooHist_list[ix]
            normalized_hist = normalizeRooHist(mass, hist)
            # normalized_hist = normalizeFlatHist(mass, hist)
            normalized_hist.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0), Invisible=True )
            # normalized_hist.plotOn(frame, LineColor=color,MarkerColor=color)
            model = coreFunction_list[ix]
            name = model.GetName()
            print(f"index {ix} with name: {name}")
            # model.Print("v")
            model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=name, LineColor=color)
            legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
        frame.Draw()
        legend.Draw()        
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{save_path}/simultaneousPlotTestFromTutorial_{core_type}.pdf")

def plotBkgBySubCat_normalized(mass:rt.RooRealVar, model_dict_by_subCat: Dict, save_path: str):
    """
    takes the dictionary of all Bkg RooAbsPdf models grouped by same sub-category, and plot them
    in the frame() of mass and saves the plots on a given directory path
    """
    # make the save_path directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    color_list = [
        rt.kGreen,
        rt.kBlue,
        rt.kRed,
        rt.kOrange,
        rt.kViolet,
    ]
    
    for subCat_idx, subCat_list in model_dict_by_subCat.items():
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        frame = mass.frame()
        frame.SetTitle(f"Normalized Shape Plot of Sub-Category {subCat_idx} PDFs")
        frame.SetXTitle(f"Dimuon Mass (GeV)")
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
        normalized_hist = normalizeRooHist(mass, roo_histData_subCat1)
        normalized_hist.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0) )
        # print(f"normalized_hist integral: {normalized_hist.sum(False)}")
        for ix in range(len(subCat_list)):
            model = subCat_list[ix]
            name = model.GetName()
            color = color_list[ix]
            model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=name, LineColor=color)
            legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
        frame.Draw()
        legend.Draw()        
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{save_path}/simultaneousPlotTestFromTutorial_subCat{subCat_idx}.pdf")

def plotBkgBySubCat(mass:rt.RooRealVar, model_dict_by_subCat: Dict, data_dict_by_subCat:Dict, save_path: str):
    """
    takes the dictionary of all Bkg RooAbsPdf models grouped by same sub-category, and plot them
    in the frame() of mass and saves the plots on a given directory path
    """
    # make the save_path directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    color_list = [
        rt.kGreen,
        rt.kBlue,
        rt.kRed,
        rt.kOrange,
        rt.kViolet,
    ]
    max_list = [1300, 1000, 400, 300, 90]
    for subCat_idx, subCat_list in model_dict_by_subCat.items():
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        frame = mass.frame()
        frame.SetMaximum(max_list[subCat_idx])
        frame.SetTitle(f"Normalized Shape Plot of Sub-Category {subCat_idx} PDFs")
        frame.SetXTitle(f"Dimuon Mass (GeV)")
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
        data_hist = data_dict_by_subCat[subCat_idx]
        data_hist.plotOn(frame)
        for ix in range(len(subCat_list)):
            model = subCat_list[ix]
            name = model.GetName()
            color = color_list[ix]
            model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=name, LineColor=color)
            legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
        frame.Draw()
        legend.Draw()        
        canvas.Update()
        canvas.Draw()
        # canvas.SaveAs(f"{save_path}/simultaneousPlotTestFromTutorial_subCat{subCat_idx}.pdf")
        canvas.SaveAs(f"{save_path}/simultaneousPlotTestFromTutorial_subCat{subCat_idx}.png")



def plotSigBySample(mass:rt.RooRealVar, model_dict_by_sample: Dict, save_path: str):
    """
    takes the dictionary of all Signal RooAbsPdf models grouped by same sample, and plot them
    in the frame() of mass and saves the plots on a given directory path
    """
    # make the save_path directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    color_list = [
        rt.kGreen,
        rt.kBlue,
        rt.kRed,
        rt.kOrange,
        rt.kViolet,
    ]
    for model_type, model_list in model_dict_by_sample.items():
        name = "Canvas"
        canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
        canvas.cd()
        frame = mass.frame()
        frame.SetTitle(f"Normalized Shape Plot of {model_type} PDFs")
        frame.SetXTitle(f"Dimuon Mass (GeV)")
        legend = rt.TLegend(0.65,0.55,0.9,0.7)
        # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
        normalized_hist = normalizeRooHist(mass, roo_histData_subCat1)
        normalized_hist.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0) )
        # print(f"normalized_hist integral: {normalized_hist.sum(False)}")
        for ix in range(len(model_list)):
            model = model_list[ix]
            name = model.GetName()
            color = color_list[ix]
            model.plotOn(frame, Name=name, LineColor=color)
            legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
        frame.Draw()
        legend.Draw()        
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"{save_path}/simultaneousPlotTestFromTutorial_{model_type}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-load",
    "--load_path",
    dest="load_path",
    default=None,
    action="store",
    help="save path to store stage1 output files",
    )
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="all",
    action="store",
    help="string value of year we are calculating",
    )
    parser.add_argument(
    "-cat",
    "--category",
    dest="category",
    default="ggH",
    action="store",
    help="string value production category we're working on",
    )
    args = parser.parse_args()
    # check for valid arguments
    if args.load_path == None:
        print("load path to load stage1 output is not specified!")
        raise ValueError

    category = args.category.lower()
    # load_path = "/work/users/yun79/stage2_output/ggH/test/processed_events_data.parquet"
    # load_path = f"{args.load_path}/{category}/{args.year}/processed_events_data.parquet"
    if args.year=="all":
        load_path = f"{args.load_path}/{category}/*/processed_events_data.parquet"
    else:
        load_path = f"{args.load_path}/{category}/{args.year}/processed_events_data.parquet"
    print(f"load_path: {load_path}")
    processed_eventsData = ak.from_parquet(load_path)
    print("events loaded!")
    
    device = "cpu"
    # device = "cuda"
    # rt.RooAbsReal.setCudaMode(True)
    # Create model for physics sample
    # -------------------------------------------------------------
    # Create observables
    mass_name = "mh_ggh"
    mass = rt.RooRealVar(mass_name, mass_name, 120, 110, 150)
    nbins = 800
    mass.setBins(nbins)
    mass.setRange("hiSB", 135, 150 )
    mass.setRange("loSB", 110, 115 )
    mass.setRange("h_peak", 115, 135 )
    mass.setRange("full", 110, 150 )
    # fit_range = "loSB,hiSB" # we're fitting bkg only
    fit_range = "hiSB,loSB" # we're fitting bkg only
    
    subCatIdx_name = "subCategory_idx"
    # subCatIdx_name = "subCategory_idx_val"

    # Initialize BWZ Redux
    # --------------------------------------------------------------

    

    # # trying bigger range do that I don't get warning message from combine like: [WARNING] Found parameter BWZ_Redux_a_coeff at boundary (within ~1sigma)
    # # old start --------------------------------------------------
    # name = f"BWZ_Redux_a_coeff"
    # a_coeff = rt.RooRealVar(name,name, -0.02,-0.03,0.03)
    # name = f"BWZ_Redux_b_coeff"
    # b_coeff = rt.RooRealVar(name,name, -0.000111,-0.001,0.001)
    # name = f"BWZ_Redux_c_coeff"
    # c_coeff = rt.RooRealVar(name,name, 0.5,-5.0,5.0)
    # # old end --------------------------------------------------

    # # AN start --------------------------------------------------
    name = f"BWZ_Redux_a_coeff"
    a_coeff = rt.RooRealVar(name,name, 0.06231018619106862,-0.1,0.1)
    name = f"BWZ_Redux_b_coeff"
    b_coeff = rt.RooRealVar(name,name, -0.0001684318108879923,-0.1,0.1)
    name = f"BWZ_Redux_c_coeff"
    c_coeff = rt.RooRealVar(name,name, 2.14876669663328,0,5.0)
    # # AN end --------------------------------------------------
    
    # subCat 0
    name = "subCat0_BWZ_Redux"
    coreBWZRedux_SubCat0 = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
     
    # Construct background pdf
    # old start --------------------------------------------------------------------
    # a0_subCat0 = rt.RooRealVar("a0_subCat0", "a0_subCat0", -0.1, -1, 1)
    # a1_subCat0 = rt.RooRealVar("a1_subCat0", "a1_subCat0", 0.5, -0.5, 0.5)
    # a3_subCat0 = rt.RooRealVar("a3_subCat0", "a3_subCat0", 0.5, -0.5, 0.5)
    # old end --------------------------------------------------------------------
    # a0_subCat0 = rt.RooRealVar("a0_subCat0", "a0_subCat0", -0.03756867559, -1, 1)
    # a1_subCat0 = rt.RooRealVar("a1_subCat0", "a1_subCat0", -0.001975507853, -0.5, 0.5)
    # a3_subCat0 = rt.RooRealVar("a3_subCat0", "a3_subCat0", -0.001975507853, -0.5, 0.5)
    a0_subCat0 = rt.RooRealVar("a0_subCat0", "a0_subCat0", -0.03756867559, -0.06, 0.06)
    a1_subCat0 = rt.RooRealVar("a1_subCat0", "a1_subCat0", -0.001975507853, -0.06, 0.06)
    a3_subCat0 = rt.RooRealVar("a3_subCat0", "a3_subCat0", -0.001975507853, -0.06, 0.06)
    a0_subCat0.setConstant(True)
    a1_subCat0.setConstant(True)
    a3_subCat0.setConstant(True)
    

    name = "subCat0_SMF"
    subCat0_SMF = rt.RooChebychev(name, name, mass, [a0_subCat0, a1_subCat0, a3_subCat0])


    
    # Construct composite pdf
    name = "model_SubCat0_SMFxBWZRedux"
    model_subCat0_BWZRedux = rt.RooProdPdf(name, name, [coreBWZRedux_SubCat0, subCat0_SMF])


    
    # subCat 1
    name = "subCat1_BWZ_Redux"
    # coreBWZRedux_SubCat1 = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
    coreBWZRedux_SubCat1 = coreBWZRedux_SubCat0
    
    # Construct the background pdf
    """
    NOTE: unlike what's written on line 1721 of Run2 AN, workspace root file in 
    https://gitlab.cern.ch/cms-analysis/hig/HIG-19-006/datacards/-/blob/master/ggH/ucsd/workspace_bkg_cat1_ggh.root?ref_type=heads
    doesn't have a third degree of freedom
    """
    # a0_subCat1 = rt.RooRealVar("a0_subCat1", "a0_subCat1", -0.1, -1, 1)
    # a1_subCat1 = rt.RooRealVar("a1_subCat1", "a1_subCat1", 0.5, -0.5, 0.5)
    # a3_subCat1 = rt.RooRealVar("a3_subCat1", "a3_subCat1", 0.5, -0.5, 0.5)
    # values from AN workspace
    # a0_subCat1 = rt.RooRealVar("a0_subCat1", "a0_subCat1", 0.01949329222, -1, 1)
    # a1_subCat1 = rt.RooRealVar("a1_subCat1", "a1_subCat1", -0.001657932368, -0.5, 0.5)
    a0_subCat1 = rt.RooRealVar("a0_subCat1", "a0_subCat1", 0.01949329222, -0.06, 0.06)
    a1_subCat1 = rt.RooRealVar("a1_subCat1", "a1_subCat1", -0.001657932368, -0.06, 0.06)
    a0_subCat1.setConstant(True)
    a1_subCat1.setConstant(True)
    name =  "subCat1_SMF"
    subCat1_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat1, 
                              a1_subCat1, 
                              # a3_subCat1
                             ])
     
    # Construct the composite model
    name = "model_SubCat1_SMFxBWZRedux"
    model_subCat1_BWZRedux = rt.RooProdPdf(name, name, [coreBWZRedux_SubCat1, subCat1_SMF])

    # subCat 2
    name = "subCat2_BWZ_Redux"
    # coreBWZRedux_SubCat2 = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
    coreBWZRedux_SubCat2 = coreBWZRedux_SubCat0
    
    # Construct the background pdf
    # a0_subCat2 = rt.RooRealVar("a0_subCat2", "a0_subCat2", -0.1, -1, 1)
    # a1_subCat2 = rt.RooRealVar("a1_subCat2", "a1_subCat2", 0.5, -0.5, 0.5)
    # a0_subCat2 = rt.RooRealVar("a0_subCat2", "a0_subCat2", 0.04460447882, -0.001, 0.001)
    # a1_subCat2 = rt.RooRealVar("a1_subCat2", "a1_subCat2", -3.46E-05, -0.001, 0.001)
    a0_subCat2 = rt.RooRealVar("a0_subCat2", "a0_subCat2", 0.04460447882, -0.06, 0.06)
    a1_subCat2 = rt.RooRealVar("a1_subCat2", "a1_subCat2", -3.46E-05, -0.06, 0.06)
    a0_subCat2.setConstant(True)
    a1_subCat2.setConstant(True)
    name = "subCat2_SMF"
    subCat2_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat2, 
                              a1_subCat2, 
                             ])
    name = "model_SubCat2_SMFxBWZRedux"
    model_subCat2_BWZRedux = rt.RooProdPdf(name, name, [coreBWZRedux_SubCat2, subCat2_SMF])    

    # subCat 3
    name = "subCat3_BWZ_Redux"
    # coreBWZRedux_SubCat3 = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
    coreBWZRedux_SubCat3 = coreBWZRedux_SubCat0
    
    # Construct the background pdf
    # a0_subCat3 = rt.RooRealVar("a0_subCat3", "a0_subCat3", -0.1, -1, 1)
    # a1_subCat3 = rt.RooRealVar("a1_subCat3", "a1_subCat3", 0.5, -0.5, 0.5)
    # a0_subCat3 = rt.RooRealVar("a0_subCat3", "a0_subCat3", 0.07374242573, 0.05, 0.5)
    # a1_subCat3 = rt.RooRealVar("a1_subCat3", "a1_subCat3", -8.79E-06, -0.5, 0.5)
    a0_subCat3 = rt.RooRealVar("a0_subCat3", "a0_subCat3", 0.07374242573, -0.06, 0.1)
    a1_subCat3 = rt.RooRealVar("a1_subCat3", "a1_subCat3", -8.79E-06, -0.06, 0.06)
    # a0_subCat3.setConstant(True)
    # a1_subCat3.setConstant(True)
    name = "subCat3_SMF"
    subCat3_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat3, 
                              a1_subCat3, 
                             ])
    name = "model_SubCat3_SMFxBWZRedux"
    model_subCat3_BWZRedux = rt.RooProdPdf(name, name, [coreBWZRedux_SubCat3, subCat3_SMF])  

    # subCat 4
    name = "subCat4_BWZ_Redux"
    # coreBWZRedux_SubCat4 = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
    coreBWZRedux_SubCat4 = coreBWZRedux_SubCat0
    
    # Construct the background pdf
    # a0_subCat4 = rt.RooRealVar("a0_subCat4", "a0_subCat4", -0.1, -1, 1)
    # a1_subCat4 = rt.RooRealVar("a1_subCat4", "a1_subCat4", 0.5, -0.5, 0.5)
    # a0_subCat4 = rt.RooRealVar("a0_subCat4", "a0_subCat4", 0.2274725556, 0.2, 1)
    # a1_subCat4 = rt.RooRealVar("a1_subCat4", "a1_subCat4", -0.0006481800973, -0.5, 1)
    # a0_subCat4 = rt.RooRealVar("a0_subCat4", "a0_subCat4", 0.2274725556, -0.06, 0.56) # AN val
    # a1_subCat4 = rt.RooRealVar("a1_subCat4", "a1_subCat4", -0.0006481800973, -0.06, 0.06) # AN val
    a0_subCat4 = rt.RooRealVar("a0_subCat4", "a0_subCat4", 0.2274725556, -0.06, 1.06) # experiment
    a1_subCat4 = rt.RooRealVar("a1_subCat4", "a1_subCat4", -0.0006481800973, -0.06, 0.06) # experiment
    # a0_subCat4.setConstant(True)
    # a1_subCat4.setConstant(True)
    name = "subCat4_SMF"
    subCat4_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat4, 
                              a1_subCat4, 
                             ])
    name = "model_SubCat4_SMFxBWZRedux"
    model_subCat4_BWZRedux = rt.RooProdPdf(name, name, [coreBWZRedux_SubCat4, subCat4_SMF])  


    # ---------------------------------------------------------------
    # Initialize Data for Bkg models to fit to
    # ---------------------------------------------------------------
     
    # do for cat idx 0
    subCat_filter = (processed_eventsData[subCatIdx_name] == 0)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat0 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat0 = rt.RooDataHist("subCat0_rooHist_BWZRedux","subCat0_rooHist_BWZRedux", rt.RooArgSet(mass), roo_datasetData_subCat0)
    data_subCat0_BWZRedux = roo_histData_subCat0

    # do for cat idx 1
    subCat_filter = (processed_eventsData[subCatIdx_name] == 1)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat1 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat1 = rt.RooDataHist("subCat1_rooHist_BWZRedux","subCat1_rooHist_BWZRedux", rt.RooArgSet(mass), roo_datasetData_subCat1)
    data_subCat1_BWZRedux = roo_histData_subCat1

    # do for cat idx 2
    subCat_filter = (processed_eventsData[subCatIdx_name] == 2)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat2 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat2 = rt.RooDataHist("subCat2_rooHist_BWZRedux","subCat2_rooHist_BWZRedux", rt.RooArgSet(mass), roo_datasetData_subCat2)
    data_subCat2_BWZRedux = roo_histData_subCat2

    # do for cat idx 3
    subCat_filter = (processed_eventsData[subCatIdx_name] == 3)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat3 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat3 = rt.RooDataHist("subCat3_rooHist_BWZRedux","subCat3_rooHist_BWZRedux", rt.RooArgSet(mass), roo_datasetData_subCat3)
    data_subCat3_BWZRedux = roo_histData_subCat3

    # do for cat idx 4
    subCat_filter = (processed_eventsData[subCatIdx_name] == 4)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat4 = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat4 = rt.RooDataHist("subCat4_rooHist_BWZRedux","subCat4_rooHist_BWZRedux", rt.RooArgSet(mass), roo_datasetData_subCat4)
    data_subCat4_BWZRedux = roo_histData_subCat4




    # --------------------------------------------------------------
    # Initialize Sum Exponential
    # --------------------------------------------------------------
    # name = f"RooSumTwoExpPdf_a1_coeff"
    # a1_coeff = rt.RooRealVar(name,name, -0.0603,-2.0,1)
    # name = f"RooSumTwoExpPdf_a2_coeff"
    # a2_coeff = rt.RooRealVar(name,name, -0.0450,-2.0,1)
    # name = f"RooSumTwoExpPdf_f_coeff"
    # f_coeff = rt.RooRealVar(name,name, 0.742,0.0,1.0)

    # name = f"RooSumTwoExpPdf_a1_coeff"
    # a1_coeff = rt.RooRealVar(name,name, -0.2,-2.0,1)
    # name = f"RooSumTwoExpPdf_a2_coeff"
    # a2_coeff = rt.RooRealVar(name,name, -0.09,-2.0,1)
    # name = f"RooSumTwoExpPdf_f_coeff"
    # f_coeff = rt.RooRealVar(name,name, 0.02,0.0,1.0)

    # name = f"RooSumTwoExpPdf_a1_coeff"
    # a1_coeff = rt.RooRealVar(name,name, -0.059609,-2.0,1)
    # name = f"RooSumTwoExpPdf_a2_coeff"
    # a2_coeff = rt.RooRealVar(name,name, -0.0625122,-2.0,1)
    # name = f"RooSumTwoExpPdf_f_coeff"
    # f_coeff = rt.RooRealVar(name,name, 0.9,0.0,1.0)

    # original start --------------------------------------------------
    # name = f"RooSumTwoExpPdf_a1_coeff"
    # a1_coeff = rt.RooRealVar(name,name, -0.043657,-2.0,1)
    # name = f"RooSumTwoExpPdf_a2_coeff"
    # a2_coeff = rt.RooRealVar(name,name, -0.23726,-2.0,1)
    # name = f"RooSumTwoExpPdf_f_coeff"
    # f_coeff = rt.RooRealVar(name,name, 0.9,0.0,1.0)
    # original end --------------------------------------------------

    # trying bigger range do that I don't get warning message from combine like: [WARNING] Found parameter BWZ_Redux_a_coeff at boundary (within ~1sigma)
    # # new start --------------------------------------------------
    # name = f"RooSumTwoExpPdf_a1_coeff"
    # a1_coeff = rt.RooRealVar(name,name, 0.00001,-2.0,1)
    # name = f"RooSumTwoExpPdf_a2_coeff"
    # a2_coeff = rt.RooRealVar(name,name, 0.1,-2.0,1)
    # name = f"RooSumTwoExpPdf_f_coeff"
    # f_coeff = rt.RooRealVar(name,name, 0.9,0.0,1.0)
    # # new end --------------------------------------------------


    # AN start --------------------------------------------------
    name = f"RooSumTwoExpPdf_a1_coeff"
    a1_coeff = rt.RooRealVar(name,name, -0.034803252906117965,-1.0,0.0)
    name = f"RooSumTwoExpPdf_a2_coeff"
    a2_coeff = rt.RooRealVar(name,name, -0.1497754374262389,-1.0,0)
    name = f"RooSumTwoExpPdf_f_coeff"
    f_coeff = rt.RooRealVar(name,name, 0.7549173445209436,0.0,1.0)
    # AN end --------------------------------------------------
    # a1_coeff.setConstant(True)
    # a2_coeff.setConstant(True)
    # f_coeff.setConstant(True)
    
    name = "subCat0_sumExp"
    coreSumExp_SubCat0 = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
     
    name = "subCat0_SMF_sumExp"
    subCat0_SumExp_SMF = rt.RooChebychev(name, name, mass, [a0_subCat0, a1_subCat0, a3_subCat0])


    
    # Construct composite pdf
    name = "model_SubCat0_SMFxSumExp"
    model_subCat0_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat0, subCat0_SumExp_SMF])
     
    # subCat 1
    name = "subCat1_sumExp"
    # coreSumExp_SubCat1 = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
    coreSumExp_SubCat1 = coreSumExp_SubCat0
    

    name = "subCat1_SMF_sumExp"
    subCat1_SumExp_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat1, 
                              a1_subCat1, 
                              # a3_subCat1
                             ])
     
    # Construct the composite model
    name = "model_SubCat1_SMFxSumExp"
    model_subCat1_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat1, subCat1_SumExp_SMF])

    # subCat 2
    name = "subCat2_sumExp"
    # coreSumExp_SubCat2 = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
    coreSumExp_SubCat2 = coreSumExp_SubCat0
    
    name = "subCat2_SMF_sumExp"
    subCat2_SumExp_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat2, 
                              a1_subCat2, 
                             ])
    name = "model_SubCat2_SMFxSumExp"
    model_subCat2_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat2, subCat2_SumExp_SMF])    

    # subCat 3
    name = "subCat3_sumExp"
    # coreSumExp_SubCat3 = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
    coreSumExp_SubCat3 = coreSumExp_SubCat0
    
    name = "subCat3_SMF_sumExp"
    subCat3_SumExp_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat3, 
                              a1_subCat3, 
                             ])
    name = "model_SubCat3_SMFxSumExp"
    model_subCat3_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat3, subCat3_SumExp_SMF])    

    # subCat 4
    name = "subCat4_sumExp"
    # coreSumExp_SubCat4 = rt.RooSumTwoExpPdf(name, name, mass, a1_coeff, a2_coeff, f_coeff) 
    coreSumExp_SubCat4 = coreSumExp_SubCat0
    
    name = "subCat4_SMF_sumExp"
    subCat4_SumExp_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat4, 
                              a1_subCat4, 
                             ])
    name = "model_SubCat4_SMFxSumExp"
    model_subCat4_sumExp = rt.RooProdPdf(name, name, [coreSumExp_SubCat4, subCat4_SumExp_SMF])    
     
    # Initialize Data for Bkg models to fit to
    # ---------------------------------------------------------------
     
    # do for cat idx 0
    subCat_filter = (processed_eventsData[subCatIdx_name] == 0)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat0_sumExp = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat0_sumExp = rt.RooDataHist("subCat0_rooHist_sumExp","subCat0_rooHist_sumExp", rt.RooArgSet(mass), roo_datasetData_subCat0_sumExp)
    data_subCat0_sumExp = roo_histData_subCat0_sumExp

    # do for cat idx 1
    subCat_filter = (processed_eventsData[subCatIdx_name] == 1)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat1_sumExp = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat1_sumExp = rt.RooDataHist("subCat1_rooHist_sumExp","subCat1_rooHist_sumExp", rt.RooArgSet(mass), roo_datasetData_subCat1_sumExp)
    data_subCat1_sumExp = roo_histData_subCat1_sumExp

    # do for cat idx 2
    subCat_filter = (processed_eventsData[subCatIdx_name] == 2)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat2_sumExp = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat2_sumExp = rt.RooDataHist("subCat2_rooHist_sumExp","subCat2_rooHist_sumExp", rt.RooArgSet(mass), roo_datasetData_subCat2_sumExp)
    data_subCat2_sumExp = roo_histData_subCat2_sumExp

    # do for cat idx 3
    subCat_filter = (processed_eventsData[subCatIdx_name] == 3)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat3_sumExp = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat3_sumExp = rt.RooDataHist("subCat3_rooHist_sumExp","subCat3_rooHist_sumExp", rt.RooArgSet(mass), roo_datasetData_subCat3_sumExp)
    data_subCat3_sumExp = roo_histData_subCat3_sumExp


    # do for cat idx 4
    subCat_filter = (processed_eventsData[subCatIdx_name] == 4)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat4_sumExp = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat4_sumExp = rt.RooDataHist("subCat4_rooHist_sumExp","subCat4_rooHist_sumExp", rt.RooArgSet(mass), roo_datasetData_subCat4_sumExp)
    data_subCat4_sumExp = roo_histData_subCat4_sumExp


    # --------------------------------------------------------------
    # Initialize FEWZxBernstein
    # --------------------------------------------------------------
    

    # # old start --------------------------------------------------
    # name = f"FEWZxBern_c1"
    # c1 = rt.RooRealVar(name,name, 0.2,-2,2)
    # name = f"FEWZxBern_c2"
    # c2 = rt.RooRealVar(name,name, 1.0,-2,2)
    # name = f"FEWZxBern_c3"
    # c3 = rt.RooRealVar(name,name, 0.1,-2,2)
    # # old end --------------------------------------------------

    # # an start --------------------------------------------------
    # name = f"FEWZxBern_c1"
    # c1 = rt.RooRealVar(name,name, 0.956483450832728,0.5,1.5)
    # name = f"FEWZxBern_c2"
    # c2 = rt.RooRealVar(name,name, 0.9607652348517792,0.5,1.5)
    # name = f"FEWZxBern_c3"
    # c3 = rt.RooRealVar(name,name, 0.9214633453188963,0.5,1.5)
    # # an end --------------------------------------------------


    # new start --------------------------------------------------
    name = f"FEWZxBern_c1"
    c1 = rt.RooRealVar(name,name, 0.1,-10,10)
    name = f"FEWZxBern_c2"
    c2 = rt.RooRealVar(name,name, 0.2,-10,10)
    name = f"FEWZxBern_c3"
    c3 = rt.RooRealVar(name,name, 0.1,-10,10)
    # new end --------------------------------------------------

    # c1.setConstant(True)
    # c2.setConstant(True)
    # c3.setConstant(True)
    
    name = "subCat0_FEWZxBern"
    coreFEWZxBern_SubCat0, params_FEWZxBern_SubCat0 = MakeFEWZxBernDof3(name, name, mass, c1, c2, c3) 
     
    name = "subCat0_SMF_FEWZxBern"
    subCat0_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, [a0_subCat0, a1_subCat0, a3_subCat0])


    
    # Construct composite pdf
    name = "model_SubCat0_SMFxFEWZxBern"
    model_subCat0_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat0, subCat0_FEWZxBern_SMF])
     
    # subCat 1
    name = "subCat1_FEWZxBern"
    # coreFEWZxBern_SubCat1, params_FEWZxBern_SubCat1 = MakeFEWZxBernDof3(name, name, mass, c1, c2, c3) 
    coreFEWZxBern_SubCat1 = coreFEWZxBern_SubCat0
    

    name = "subCat1_SMF_FEWZxBern"
    subCat1_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat1, 
                              a1_subCat1, 
                              # a3_subCat1
                             ])
     
    # Construct the composite model
    name = "model_SubCat1_SMFxFEWZxBern"
    model_subCat1_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat1, subCat1_FEWZxBern_SMF])

    # subCat 2
    name = "subCat2_FEWZxBern"
    # coreFEWZxBern_SubCat2, params_FEWZxBern_SubCat2 = MakeFEWZxBernDof3(name, name, mass, c1, c2, c3) 
    coreFEWZxBern_SubCat2 = coreFEWZxBern_SubCat0
    
    name = "subCat2_SMF_FEWZxBern"
    subCat2_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat2, 
                              a1_subCat2, 
                             ])
    name = "model_SubCat2_SMFxFEWZxBern"
    model_subCat2_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat2, subCat2_FEWZxBern_SMF])    

    # subCat 3
    name = "subCat3_FEWZxBern"
    # coreFEWZxBern_SubCat3, params_FEWZxBern_SubCat3 = MakeFEWZxBernDof3(name, name, mass, c1, c2, c3)  
    coreFEWZxBern_SubCat3 = coreFEWZxBern_SubCat0
    
    name = "subCat3_SMF_FEWZxBern"
    subCat3_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat3, 
                              a1_subCat3, 
                             ])
    name = "model_SubCat3_SMFxFEWZxBern"
    model_subCat3_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat3, subCat3_FEWZxBern_SMF])    

    # subCat 4
    name = "subCat4_FEWZxBern"
    # coreFEWZxBern_SubCat4, params_FEWZxBern_SubCat4 = MakeFEWZxBernDof3(name, name, mass, c1, c2, c3)  
    coreFEWZxBern_SubCat4 = coreFEWZxBern_SubCat0
    
    name = "subCat4_SMF_FEWZxBern"
    subCat4_FEWZxBern_SMF = rt.RooChebychev(name, name, mass, 
                             [a0_subCat4, 
                              a1_subCat4, 
                             ])
    name = "model_SubCat4_SMFxFEWZxBern"
    model_subCat4_FEWZxBern = rt.RooProdPdf(name, name, [coreFEWZxBern_SubCat4, subCat4_FEWZxBern_SMF])        
     
    # Initialize Data for Bkg models to fit to
    # ---------------------------------------------------------------
     
    # do for cat idx 0
    subCat_filter = (processed_eventsData[subCatIdx_name] == 0)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat0_FEWZxBern = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat0_FEWZxBern = rt.RooDataHist("subCat0_rooHist_FEWZxBern","subCat0_rooHist_FEWZxBern", rt.RooArgSet(mass), roo_datasetData_subCat0_FEWZxBern)
    data_subCat0_FEWZxBern = roo_histData_subCat0_FEWZxBern

    # do for cat idx 1
    subCat_filter = (processed_eventsData[subCatIdx_name] == 1)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat1_FEWZxBern = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat1_FEWZxBern = rt.RooDataHist("subCat1_rooHist_FEWZxBern","subCat1_rooHist_FEWZxBern", rt.RooArgSet(mass), roo_datasetData_subCat1_FEWZxBern)
    data_subCat1_FEWZxBern = roo_histData_subCat1_FEWZxBern

    # do for cat idx 2
    subCat_filter = (processed_eventsData[subCatIdx_name] == 2)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat2_FEWZxBern = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat2_FEWZxBern = rt.RooDataHist("subCat2_rooHist_FEWZxBern","subCat2_rooHist_FEWZxBern", rt.RooArgSet(mass), roo_datasetData_subCat2_FEWZxBern)
    data_subCat2_FEWZxBern = roo_histData_subCat2_FEWZxBern

    # do for cat idx 3
    subCat_filter = (processed_eventsData[subCatIdx_name] == 3)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat3_FEWZxBern = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    # print(f"roo_datasetData_subCat3_FEWZxBern name: {roo_datasetData_subCat3_FEWZxBern.GetName()}")
    roo_histData_subCat3_FEWZxBern = rt.RooDataHist("subCat3_rooHist_FEWZxBern","subCat3_rooHist_FEWZxBern", rt.RooArgSet(mass), roo_datasetData_subCat3_FEWZxBern)
    data_subCat3_FEWZxBern = roo_histData_subCat3_FEWZxBern
    # print(f"data_subCat3_FEWZxBern name: {data_subCat3_FEWZxBern.GetName()}")


    # do for cat idx 4
    subCat_filter = (processed_eventsData[subCatIdx_name] == 4)
    subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
    subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
    roo_datasetData_subCat4_FEWZxBern = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
    roo_histData_subCat4_FEWZxBern = rt.RooDataHist("subCat4_rooHist_FEWZxBern","subCat4_rooHist_FEWZxBern", rt.RooArgSet(mass), roo_datasetData_subCat4_FEWZxBern)
    data_subCat4_FEWZxBern = roo_histData_subCat4_FEWZxBern

    
    #----------------------------------------------------------------------------
    # Create index category and join samples
    # ---------------------------------------------------------------------------
     
    # Define category to distinguish physics and control samples events
    sample = rt.RooCategory("sample", "sample")
    sample.defineType("subCat0_BWZRedux")
    sample.defineType("subCat1_BWZRedux")
    sample.defineType("subCat2_BWZRedux")
    sample.defineType("subCat3_BWZRedux")
    sample.defineType("subCat4_BWZRedux")
    sample.defineType("subCat0_sumExp")
    sample.defineType("subCat1_sumExp")
    sample.defineType("subCat2_sumExp")
    sample.defineType("subCat3_sumExp")
    sample.defineType("subCat4_sumExp")
    sample.defineType("subCat0_FEWZxBern")
    sample.defineType("subCat1_FEWZxBern")
    sample.defineType("subCat2_FEWZxBern")
    sample.defineType("subCat3_FEWZxBern")
    sample.defineType("subCat4_FEWZxBern")
     
    # Construct combined dataset in (x,sample)
    combData = rt.RooDataSet(
        "combData",
        "combined data",
        {mass},
        Index=sample,
        Import={
            "subCat0_BWZRedux": data_subCat0_BWZRedux, 
            "subCat1_BWZRedux": data_subCat1_BWZRedux,
            "subCat2_BWZRedux": data_subCat2_BWZRedux,
            "subCat3_BWZRedux": data_subCat3_BWZRedux,
            "subCat4_BWZRedux": data_subCat4_BWZRedux,
            "subCat0_sumExp": data_subCat0_sumExp, 
            "subCat1_sumExp": data_subCat1_sumExp,
            "subCat2_sumExp": data_subCat2_sumExp,
            "subCat3_sumExp": data_subCat3_sumExp,
            "subCat4_sumExp": data_subCat4_sumExp,
            # "subCat0_FEWZxBern": data_subCat0_FEWZxBern, 
            # "subCat1_FEWZxBern": data_subCat1_FEWZxBern,
            # "subCat2_FEWZxBern": data_subCat2_FEWZxBern,
            # "subCat3_FEWZxBern": data_subCat3_FEWZxBern,
            # "subCat4_FEWZxBern": data_subCat4_FEWZxBern,
        },
    )
    # ---------------------------------------------------
    # Construct a simultaneous pdf in (x, sample)
    # -----------------------------------------------------------------------------------
     
    simPdf = rt.RooSimultaneous(
                                "simPdf", 
                                "simultaneous pdf", 
                                {
                                    "subCat0_BWZRedux": model_subCat0_BWZRedux, 
                                    "subCat1_BWZRedux": model_subCat1_BWZRedux,
                                    "subCat2_BWZRedux": model_subCat2_BWZRedux,
                                    "subCat3_BWZRedux": model_subCat3_BWZRedux,
                                    "subCat4_BWZRedux": model_subCat4_BWZRedux,
                                    "subCat0_sumExp": model_subCat0_sumExp, 
                                    "subCat1_sumExp": model_subCat1_sumExp,
                                    "subCat2_sumExp": model_subCat2_sumExp,
                                    "subCat3_sumExp": model_subCat3_sumExp,
                                    "subCat4_sumExp": model_subCat4_sumExp,
                                    # "subCat0_FEWZxBern": model_subCat0_FEWZxBern, 
                                    # "subCat1_FEWZxBern": model_subCat1_FEWZxBern,
                                    # "subCat2_FEWZxBern": model_subCat2_FEWZxBern,
                                    # "subCat3_FEWZxBern": model_subCat3_FEWZxBern,
                                    # "subCat4_FEWZxBern": model_subCat4_FEWZxBern,
                                }, 
                                sample,
    )
    # ---------------------------------------------------
    # Perform a simultaneous fit
    # ---------------------------------------------------
     
    start = time.time()

    _ = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend=device,  PrintLevel=0 ,Save=True, Strategy=0)
    fitResult = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend=device, PrintLevel=0 ,Save=True,)
    end = time.time()
    
    fitResult.Print()
    print(f"runtime: {end-start} seconds")

    # ---------------------------------------------------
    # Make CORE-PDF
    # ---------------------------------------------------

    # subCat 0 
    cat_subCat0 = rt.RooCategory("pdf_index_ggh","Index of Pdf which is active"); # name of category index should stay same across subCategories
    
    # // Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # // 0 == BWZ_Redux
    # // 1 == sumExp
    # // 2 == PowerSum
    
    # FEWZxBern Sumexp is less dependent to dimuon mass as stated in line 1585 of RERECO AN
    # I suppose BWZredux is there bc it's the one function with overall least bias (which is why BWZredux is used if CORE-PDF is not used)
    pdf_list_subCat0 = rt.RooArgList(
        model_subCat0_BWZRedux,
        model_subCat0_sumExp,
        # model_subCat0_FEWZxBern,
    )
    corePdf_subCat0 = rt.RooMultiPdf("CorePdf_subCat0","CorePdf_subCat0",cat_subCat0,pdf_list_subCat0)
    penalty = 0 # as told in https://cms-talk.web.cern.ch/t/combine-fitting-not-working-with-roomultipdf-leading-to-bad-signal-significance/44238/
    corePdf_subCat0.setCorrectionFactor(penalty) 
    nevents = roo_datasetData_subCat0.sumEntries() # these are data, so all weights are one, thus no need to sum over the weights, though ofc you can just do that too
    bkg_subCat0_norm = rt.RooRealVar(corePdf_subCat0.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value


    # subCat 1 
    cat_subCat1 = rt.RooCategory("pdf_index_ggh","Index of Pdf which is active"); # name of category index should stay same across subCategories
    
    # // Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # // 0 == BWZ_Redux
    # // 1 == sumExp
    # // 2 == PowerSum
    
    # FEWZxBern Sumexp is less dependent to dimuon mass as stated in line 1585 of RERECO AN
    # I suppose BWZredux is there bc it's the one function with overall least bias (which is why BWZredux is used if CORE-PDF is not used)
    pdf_list_subCat1 = rt.RooArgList(
        model_subCat1_BWZRedux,
        model_subCat1_sumExp,
        # model_subCat1_FEWZxBern,
    )
    corePdf_subCat1 = rt.RooMultiPdf("CorePdf_subCat1","CorePdf_subCat1",cat_subCat1,pdf_list_subCat1)
    penalty = 0 # as told in https://cms-talk.web.cern.ch/t/combine-fitting-not-working-with-roomultipdf-leading-to-bad-signal-significance/44238/
    corePdf_subCat1.setCorrectionFactor(penalty) 
    nevents = roo_datasetData_subCat1.sumEntries() # these are data, so all weights are one, thus no need to sum over the weights, though ofc you can just do that too
    bkg_subCat1_norm = rt.RooRealVar(corePdf_subCat1.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value

    # subCat 2 
    cat_subCat2 = rt.RooCategory("pdf_index_ggh","Index of Pdf which is active"); # name of category index should stay same across subCategories
    
    # // Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # // 0 == BWZ_Redux
    # // 1 == sumExp
    # // 2 == PowerSum
    
    # FEWZxBern Sumexp is less dependent to dimuon mass as stated in line 1585 of RERECO AN
    # I suppose BWZredux is there bc it's the one function with overall least bias (which is why BWZredux is used if CORE-PDF is not used)
    pdf_list_subCat2 = rt.RooArgList(
        model_subCat2_BWZRedux,
        model_subCat2_sumExp,
        # model_subCat2_FEWZxBern,
    )
    corePdf_subCat2 = rt.RooMultiPdf("CorePdf_subCat2","CorePdf_subCat2",cat_subCat2,pdf_list_subCat2)
    penalty = 0 # as told in https://cms-talk.web.cern.ch/t/combine-fitting-not-working-with-roomultipdf-leading-to-bad-signal-significance/44238/
    corePdf_subCat2.setCorrectionFactor(penalty) 
    nevents = roo_datasetData_subCat2.sumEntries() # these are data, so all weights are one, thus no need to sum over the weights, though ofc you can just do that too
    bkg_subCat2_norm = rt.RooRealVar(corePdf_subCat2.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value

    # subCat 3 
    cat_subCat3 = rt.RooCategory("pdf_index_ggh","Index of Pdf which is active"); # name of category index should stay same across subCategories
    
    # // Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # // 0 == BWZ_Redux
    # // 1 == sumExp
    # // 2 == PowerSum
    
    # FEWZxBern Sumexp is less dependent to dimuon mass as stated in line 1585 of RERECO AN
    # I suppose BWZredux is there bc it's the one function with overall least bias (which is why BWZredux is used if CORE-PDF is not used)
    pdf_list_subCat3 = rt.RooArgList(
        model_subCat3_BWZRedux,
        model_subCat3_sumExp,
        # model_subCat3_FEWZxBern,
    )
    corePdf_subCat3 = rt.RooMultiPdf("CorePdf_subCat3","CorePdf_subCat3",cat_subCat3,pdf_list_subCat3)
    penalty = 0 # as told in https://cms-talk.web.cern.ch/t/combine-fitting-not-working-with-roomultipdf-leading-to-bad-signal-significance/44238/
    corePdf_subCat3.setCorrectionFactor(penalty) 
    nevents = roo_datasetData_subCat3.sumEntries() # these are data, so all weights are one, thus no need to sum over the weights, though ofc you can just do that too
    bkg_subCat3_norm = rt.RooRealVar(corePdf_subCat3.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value

    # subCat 4
    cat_subCat4 = rt.RooCategory("pdf_index_ggh","Index of Pdf which is active"); # name of category index should stay same across subCategories
    
    # // Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # // 0 == BWZ_Redux
    # // 1 == sumExp
    # // 2 == PowerSum
    
    # FEWZxBern Sumexp is less dependent to dimuon mass as stated in line 1585 of RERECO AN
    # I suppose BWZredux is there bc it's the one function with overall least bias (which is why BWZredux is used if CORE-PDF is not used)
    pdf_list_subCat4 = rt.RooArgList(
        model_subCat4_BWZRedux,
        model_subCat4_sumExp,
        # model_subCat4_FEWZxBern,
    )
    corePdf_subCat4 = rt.RooMultiPdf("CorePdf_subCat4","CorePdf_subCat4",cat_subCat4,pdf_list_subCat4)
    penalty = 0 # as told in https://cms-talk.web.cern.ch/t/combine-fitting-not-working-with-roomultipdf-leading-to-bad-signal-significance/44238/
    corePdf_subCat4.setCorrectionFactor(penalty) 
    nevents = roo_datasetData_subCat4.sumEntries() # these are data, so all weights are one, thus no need to sum over the weights, though ofc you can just do that too
    bkg_subCat4_norm = rt.RooRealVar(corePdf_subCat4.GetName()+"_norm","Background normalization value",nevents,0,3*nevents) # free floating value
    


    # ---------------------------------------------------
    # Obtain signal MC events
    # ---------------------------------------------------

    # load_path = f"{args.load_path}/{category}/{args.year}/processed_events_signalMC.parquet"
    if args.year=="all":
        load_path = f"{args.load_path}/{category}/*/processed_events_sigMC_ggh.parquet"
    else:
        load_path = f"{args.load_path}/{category}/{args.year}/processed_events_sigMC_ggh.parquet" # Fig 6.15 was only with ggH process, though with all 2016, 2017 and 2018
    # load_path = f"{args.load_path}/{category}/{args.year}/processed_events_sigMC*.parquet"
    
    processed_eventsSignalMC = ak.from_parquet(load_path)
    print(f"ggH yield: {np.sum(processed_eventsSignalMC.wgt_nominal_total)}")
    print("signal events loaded")
    
    # ---------------------------------------------------
    # Define signal model's Doubcl Crystal Ball PDF
    # ---------------------------------------------------
    
    # subCat 0
    MH_subCat0 = rt.RooRealVar("MH" , "MH", 125, 115,135)
    MH_subCat0.setConstant(True) # this shouldn't change, I think
    # sigma_subCat0 = rt.RooRealVar("sigma_subCat0" , "sigma_subCat0", 2, .1, 4.0)
    # alpha1_subCat0 = rt.RooRealVar("alpha1_subCat0" , "alpha1_subCat0", 2, 0.01, 65)
    # n1_subCat0 = rt.RooRealVar("n1_subCat0" , "n1_subCat0", 10, 0.01, 100)
    # alpha2_subCat0 = rt.RooRealVar("alpha2_subCat0" , "alpha2_subCat0", 2.0, 0.01, 65)
    # n2_subCat0 = rt.RooRealVar("n2_subCat0" , "n2_subCat0", 25, 0.01, 100)

    # copying parameters from official AN workspace as starting params
    sigma_subCat0 = rt.RooRealVar("sigma_subCat0" , "sigma_subCat0", 1.8228, .1, 4.0)
    alpha1_subCat0 = rt.RooRealVar("alpha1_subCat0" , "alpha1_subCat0", 1.12842, 0.01, 65)
    n1_subCat0 = rt.RooRealVar("n1_subCat0" , "n1_subCat0", 4.019960, 0.01, 100)
    alpha2_subCat0 = rt.RooRealVar("alpha2_subCat0" , "alpha2_subCat0", 1.3132, 0.01, 65)
    n2_subCat0 = rt.RooRealVar("n2_subCat0" , "n2_subCat0", 9.97411, 0.01, 100)

    # # temporary test
    # sigma_subCat0.setConstant(True)
    # alpha1_subCat0.setConstant(True)
    # n1_subCat0.setConstant(True)
    # alpha2_subCat0.setConstant(True)
    # n2_subCat0.setConstant(True)
    
    
    CMS_hmm_sigma_cat0_ggh = rt.RooRealVar("CMS_hmm_sigma_cat0_ggh" , "CMS_hmm_sigma_cat0_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat0_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat0_ggh_fsigma = rt.RooFormulaVar("ggH_cat0_ggh_fsigma", "ggH_cat0_ggh_fsigma",'@0*(1+@1)',[sigma_subCat0, CMS_hmm_sigma_cat0_ggh])
    CMS_hmm_peak_cat0_ggh = rt.RooRealVar("CMS_hmm_peak_cat0_ggh" , "CMS_hmm_peak_cat0_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat0_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat0_ggh_fpeak = rt.RooFormulaVar("ggH_cat0_ggh_fpeak", "ggH_cat0_ggh_fpeak",'@0*(1+@1)',[MH_subCat0, CMS_hmm_peak_cat0_ggh])
    
    # n1_subCat0.setConstant(True) # freeze for stability
    # n2_subCat0.setConstant(True) # freeze for stability
    name = "signal_subCat0"
    signal_subCat0 = rt.RooCrystalBall(name,name,mass, ggH_cat0_ggh_fpeak, ggH_cat0_ggh_fsigma, alpha1_subCat0, n1_subCat0, alpha2_subCat0, n2_subCat0)

    # subCat 1
    MH_subCat1 = rt.RooRealVar("MH" , "MH", 125, 115,135)
    MH_subCat1.setConstant(True) # this shouldn't change, I think
    # sigma_subCat1 = rt.RooRealVar("sigma_subCat1" , "sigma_subCat1", 2, .1, 4.0)
    # alpha1_subCat1 = rt.RooRealVar("alpha1_subCat1" , "alpha1_subCat1", 2, 0.01, 65)
    # n1_subCat1 = rt.RooRealVar("n1_subCat1" , "n1_subCat1", 10, 0.01, 100)
    # alpha2_subCat1 = rt.RooRealVar("alpha2_subCat1" , "alpha2_subCat1", 2.0, 0.01, 65)
    # n2_subCat1 = rt.RooRealVar("n2_subCat1" , "n2_subCat1", 25, 0.01, 100)

    # copying parameters from official AN workspace as starting params
    sigma_subCat1 = rt.RooRealVar("sigma_subCat1" , "sigma_subCat1", 1.503280, .1, 4.0)
    alpha1_subCat1 = rt.RooRealVar("alpha1_subCat1" , "alpha1_subCat1", 1.3364, 0.01, 65)
    n1_subCat1 = rt.RooRealVar("n1_subCat1" , "n1_subCat1", 2.815022, 0.01, 100)
    alpha2_subCat1 = rt.RooRealVar("alpha2_subCat1" , "alpha2_subCat1", 1.57127749, 0.01, 65)
    n2_subCat1 = rt.RooRealVar("n2_subCat1" , "n2_subCat1", 9.99687, 0.01, 100)

    # # temporary test
    # sigma_subCat1.setConstant(True)
    # alpha1_subCat1.setConstant(True)
    # n1_subCat1.setConstant(True)
    # alpha2_subCat1.setConstant(True)
    # n2_subCat1.setConstant(True)
    
    CMS_hmm_sigma_cat1_ggh = rt.RooRealVar("CMS_hmm_sigma_cat1_ggh" , "CMS_hmm_sigma_cat1_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat1_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat1_ggh_fsigma = rt.RooFormulaVar("ggH_cat1_ggh_fsigma", "ggH_cat1_ggh_fsigma",'@0*(1+@1)',[sigma_subCat1, CMS_hmm_sigma_cat1_ggh])
    CMS_hmm_peak_cat1_ggh = rt.RooRealVar("CMS_hmm_peak_cat1_ggh" , "CMS_hmm_peak_cat1_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat1_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat1_ggh_fpeak = rt.RooFormulaVar("ggH_cat1_ggh_fpeak", "ggH_cat1_ggh_fpeak",'@0*(1+@1)',[MH_subCat1, CMS_hmm_peak_cat1_ggh])
    
    # n1_subCat1.setConstant(True) # freeze for stability
    # n2_subCat1.setConstant(True) # freeze for stability
    name = "signal_subCat1"
    signal_subCat1 = rt.RooCrystalBall(name,name,mass, ggH_cat1_ggh_fpeak, ggH_cat1_ggh_fsigma, alpha1_subCat1, n1_subCat1, alpha2_subCat1, n2_subCat1)

    # subCat 2
    MH_subCat2 = rt.RooRealVar("MH" , "MH", 125, 115,135)
    MH_subCat2.setConstant(True) # this shouldn't change, I think
    # sigma_subCat2 = rt.RooRealVar("sigma_subCat2" , "sigma_subCat2", 2, .1, 4.0)
    # alpha1_subCat2 = rt.RooRealVar("alpha1_subCat2" , "alpha1_subCat2", 2, 0.01, 65)
    # n1_subCat2 = rt.RooRealVar("n1_subCat2" , "n1_subCat2", 10, 0.01, 100)
    # alpha2_subCat2 = rt.RooRealVar("alpha2_subCat2" , "alpha2_subCat2", 2.0, 0.01, 65)
    # n2_subCat2 = rt.RooRealVar("n2_subCat2" , "n2_subCat2", 25, 0.01, 100)

    # copying parameters from official AN workspace as starting params
    sigma_subCat2 = rt.RooRealVar("sigma_subCat2" , "sigma_subCat2", 1.36025, .1, 4.0)
    alpha1_subCat2 = rt.RooRealVar("alpha1_subCat2" , "alpha1_subCat2", 1.4173626, 0.01, 65)
    n1_subCat2 = rt.RooRealVar("n1_subCat2" , "n1_subCat2", 2.42748, 0.01, 100)
    alpha2_subCat2 = rt.RooRealVar("alpha2_subCat2" , "alpha2_subCat2", 1.629120, 0.01, 65)
    n2_subCat2 = rt.RooRealVar("n2_subCat2" , "n2_subCat2", 9.983334, 0.01, 100)

    # # temporary test
    # sigma_subCat2.setConstant(True)
    # alpha1_subCat2.setConstant(True)
    # n1_subCat2.setConstant(True)
    # alpha2_subCat2.setConstant(True)
    # n2_subCat2.setConstant(True)

    CMS_hmm_sigma_cat2_ggh = rt.RooRealVar("CMS_hmm_sigma_cat2_ggh" , "CMS_hmm_sigma_cat2_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat2_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat2_ggh_fsigma = rt.RooFormulaVar("ggH_cat2_ggh_fsigma", "ggH_cat2_ggh_fsigma",'@0*(1+@1)',[sigma_subCat2, CMS_hmm_sigma_cat2_ggh])
    CMS_hmm_peak_cat2_ggh = rt.RooRealVar("CMS_hmm_peak_cat2_ggh" , "CMS_hmm_peak_cat2_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat2_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat2_ggh_fpeak = rt.RooFormulaVar("ggH_cat2_ggh_fpeak", "ggH_cat2_ggh_fpeak",'@0*(1+@1)',[MH_subCat2, CMS_hmm_peak_cat2_ggh])
    
    # n1_subCat2.setConstant(True) # freeze for stability
    # n2_subCat2.setConstant(True) # freeze for stability
    name = "signal_subCat2"
    signal_subCat2 = rt.RooCrystalBall(name,name,mass, ggH_cat2_ggh_fpeak, ggH_cat2_ggh_fsigma, alpha1_subCat2, n1_subCat2, alpha2_subCat2, n2_subCat2)

    # subCat 3
    MH_subCat3 = rt.RooRealVar("MH" , "MH", 125, 115,135)
    MH_subCat3.setConstant(True) # this shouldn't change, I think
    # sigma_subCat3 = rt.RooRealVar("sigma_subCat3" , "sigma_subCat3", 2, .1, 4.0)
    # alpha1_subCat3 = rt.RooRealVar("alpha1_subCat3" , "alpha1_subCat3", 2, 0.01, 65)
    # n1_subCat3 = rt.RooRealVar("n1_subCat3" , "n1_subCat3", 10, 0.01, 100)
    # alpha2_subCat3 = rt.RooRealVar("alpha2_subCat3" , "alpha2_subCat3", 2.0, 0.01, 65)
    # n2_subCat3 = rt.RooRealVar("n2_subCat3" , "n2_subCat3", 25, 0.01, 100)

    # copying parameters from official AN workspace as starting params
    sigma_subCat3 = rt.RooRealVar("sigma_subCat3" , "sigma_subCat3", 1.25359, .1, 4.0)
    alpha1_subCat3 = rt.RooRealVar("alpha1_subCat3" , "alpha1_subCat3", 1.4199, 0.01, 65)
    n1_subCat3 = rt.RooRealVar("n1_subCat3" , "n1_subCat3", 2.409953, 0.01, 100)
    alpha2_subCat3 = rt.RooRealVar("alpha2_subCat3" , "alpha2_subCat3", 1.64675, 0.01, 65)
    n2_subCat3 = rt.RooRealVar("n2_subCat3" , "n2_subCat3", 9.670221, 0.01, 100)

    # # temporary test
    # sigma_subCat3.setConstant(True)
    # alpha1_subCat3.setConstant(True)
    # n1_subCat3.setConstant(True)
    # alpha2_subCat3.setConstant(True)
    # n2_subCat3.setConstant(True)

    CMS_hmm_sigma_cat3_ggh = rt.RooRealVar("CMS_hmm_sigma_cat3_ggh" , "CMS_hmm_sigma_cat3_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat3_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat3_ggh_fsigma = rt.RooFormulaVar("ggH_cat3_ggh_fsigma", "ggH_cat3_ggh_fsigma",'@0*(1+@1)',[sigma_subCat3, CMS_hmm_sigma_cat3_ggh])
    CMS_hmm_peak_cat3_ggh = rt.RooRealVar("CMS_hmm_peak_cat3_ggh" , "CMS_hmm_peak_cat3_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat3_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat3_ggh_fpeak = rt.RooFormulaVar("ggH_cat3_ggh_fpeak", "ggH_cat3_ggh_fpeak",'@0*(1+@1)',[MH_subCat3, CMS_hmm_peak_cat3_ggh])
    
    # n1_subCat3.setConstant(True) # freeze for stability
    # n2_subCat3.setConstant(True) # freeze for stability
    name = "signal_subCat3"
    signal_subCat3 = rt.RooCrystalBall(name,name,mass, ggH_cat3_ggh_fpeak, ggH_cat3_ggh_fsigma, alpha1_subCat3, n1_subCat3, alpha2_subCat3, n2_subCat3)

    # subCat 4
    MH_subCat4 = rt.RooRealVar("MH" , "MH", 125, 115,135)
    MH_subCat4.setConstant(True) # this shouldn't change, I think
    # sigma_subCat4 = rt.RooRealVar("sigma_subCat4" , "sigma_subCat4", 2, .1, 4.0)
    # alpha1_subCat4 = rt.RooRealVar("alpha1_subCat4" , "alpha1_subCat4", 2, 0.01, 65)
    # n1_subCat4 = rt.RooRealVar("n1_subCat4" , "n1_subCat4", 10, 0.01, 100)
    # alpha2_subCat4 = rt.RooRealVar("alpha2_subCat4" , "alpha2_subCat4", 2.0, 0.01, 65)
    # n2_subCat4 = rt.RooRealVar("n2_subCat4" , "n2_subCat4", 25, 0.01, 100)

    # copying parameters from official AN workspace as starting params
    sigma_subCat4 = rt.RooRealVar("sigma_subCat4" , "sigma_subCat4", 1.28250, .1, 4.0)
    alpha1_subCat4 = rt.RooRealVar("alpha1_subCat4" , "alpha1_subCat4", 1.47936, 0.01, 65)
    n1_subCat4 = rt.RooRealVar("n1_subCat4" , "n1_subCat4", 2.24104, 0.01, 100)
    alpha2_subCat4 = rt.RooRealVar("alpha2_subCat4" , "alpha2_subCat4", 1.67898, 0.01, 65)
    n2_subCat4 = rt.RooRealVar("n2_subCat4" , "n2_subCat4", 8.8719, 0.01, 100)

    # # temporary test
    # sigma_subCat4.setConstant(True)
    # alpha1_subCat4.setConstant(True)
    # n1_subCat4.setConstant(True)
    # alpha2_subCat4.setConstant(True)
    # n2_subCat4.setConstant(True)

    CMS_hmm_sigma_cat4_ggh = rt.RooRealVar("CMS_hmm_sigma_cat4_ggh" , "CMS_hmm_sigma_cat4_ggh", 0, -5 , 5 )
    CMS_hmm_sigma_cat4_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat4_ggh_fsigma = rt.RooFormulaVar("ggH_cat4_ggh_fsigma", "ggH_cat4_ggh_fsigma",'@0*(1+@1)',[sigma_subCat4, CMS_hmm_sigma_cat4_ggh])
    CMS_hmm_peak_cat4_ggh = rt.RooRealVar("CMS_hmm_peak_cat4_ggh" , "CMS_hmm_peak_cat4_ggh", 0, -5 , 5 )
    CMS_hmm_peak_cat4_ggh.setConstant(True) # this is going to be param in datacard
    ggH_cat4_ggh_fpeak = rt.RooFormulaVar("ggH_cat4_ggh_fpeak", "ggH_cat4_ggh_fpeak",'@0*(1+@1)',[MH_subCat4, CMS_hmm_peak_cat4_ggh])
    
    # n1_subCat4.setConstant(True) # freeze for stability
    # n2_subCat4.setConstant(True) # freeze for stability
    name = "signal_subCat4"
    signal_subCat4 = rt.RooCrystalBall(name,name,mass, ggH_cat4_ggh_fpeak, ggH_cat4_ggh_fsigma, alpha1_subCat4, n1_subCat4, alpha2_subCat4, n2_subCat4)
    
    
    # ---------------------------------------------------
    # Define signal MC samples to fit to
    # ---------------------------------------------------

    # subCat 0
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 0)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat0_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal_total[subCat_filter]
    ) # weights

    # generate a weighted histogram 
    roo_histData_subCat0_signal = rt.TH1F("subCat0_rooHist_signal", "subCat0_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat0_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat0_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat0_signal = rt.RooDataHist("subCat0_rooHist_signal", "subCat0_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat0_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat0_signal = roo_histData_subCat0_signal

    # define normalization value from signal MC event weights 
    flat_MC_SF = 1.00
    # flat_MC_SF = 0.92 # temporary flat SF to match my Data/MC agreement to that of AN's
    norm_val = np.sum(wgt_subCat0_SigMC)* flat_MC_SF 
    sig_norm_subCat0 = rt.RooRealVar(signal_subCat0.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat0 norm_val: {norm_val}")
    sig_norm_subCat0.setConstant(True)

    # subCat 1
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 1)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat1_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal_total[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat1_signal = rt.TH1F("subCat1_rooHist_signal", "subCat1_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat1_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat1_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat1_signal = rt.RooDataHist("subCat1_rooHist_signal", "subCat1_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat1_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat1_signal = roo_histData_subCat1_signal

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat1_SigMC)* flat_MC_SF
    sig_norm_subCat1 = rt.RooRealVar(signal_subCat1.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat1 norm_val: {norm_val}")
    sig_norm_subCat1.setConstant(True)

    # subCat 2
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 2)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat2_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal_total[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat2_signal = rt.TH1F("subCat2_rooHist_signal", "subCat2_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat2_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat2_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat2_signal = rt.RooDataHist("subCat2_rooHist_signal", "subCat2_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat2_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat2_signal = roo_histData_subCat2_signal

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat2_SigMC) * flat_MC_SF
    sig_norm_subCat2 = rt.RooRealVar(signal_subCat2.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat2 norm_val: {norm_val}")
    sig_norm_subCat2.setConstant(True)

    # subCat 3
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 3)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat3_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal_total[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat3_signal = rt.TH1F("subCat3_rooHist_signal", "subCat3_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat3_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat3_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat3_signal = rt.RooDataHist("subCat3_rooHist_signal", "subCat3_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat3_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat3_signal = roo_histData_subCat3_signal

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat3_SigMC)* flat_MC_SF
    sig_norm_subCat3 = rt.RooRealVar(signal_subCat3.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat3 norm_val: {norm_val}")
    sig_norm_subCat3.setConstant(True)
    
    # subCat 4
    subCat_filter = (processed_eventsSignalMC[subCatIdx_name] == 4)
    subCat_mass_arr = ak.to_numpy(
        processed_eventsSignalMC.dimuon_mass[subCat_filter]
    ) # mass values
    wgt_subCat4_SigMC = ak.to_numpy(
        processed_eventsSignalMC.wgt_nominal_total[subCat_filter]
    ) # weights
    
    # generate a weighted histogram 
    roo_histData_subCat4_signal = rt.TH1F("subCat4_rooHist_signal", "subCat4_rooHist_signal", nbins, mass.getMin(), mass.getMax())
       
    roo_histData_subCat4_signal.FillN(len(subCat_mass_arr), subCat_mass_arr, wgt_subCat4_SigMC) # fill the histograms with mass and weights 
    roo_histData_subCat4_signal = rt.RooDataHist("subCat4_rooHist_signal", "subCat4_rooHist_signal", rt.RooArgSet(mass), roo_histData_subCat4_signal) # convert to RooDataHist with (picked same name, bc idk)
    
    data_subCat4_signal = roo_histData_subCat4_signal

    # define normalization value from signal MC event weights 
    
    norm_val = np.sum(wgt_subCat4_SigMC)* flat_MC_SF
    sig_norm_subCat4 = rt.RooRealVar(signal_subCat4.GetName()+"_norm","Number of signal events",norm_val)
    print(f"signal_subCat4 norm_val: {norm_val}")
    sig_norm_subCat4.setConstant(True)
    
    # ---------------------------------------------------
    # Fit signal model individually, not simultaneous. Sigma, and left and right tails are different for each category
    # ---------------------------------------------------

    # subCat 0
    _ = signal_subCat0.fitTo(data_subCat0_signal,  EvalBackend=device, Save=True, )
    fit_result = signal_subCat0.fitTo(data_subCat0_signal,  EvalBackend=device, Save=True, )
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat0.setConstant(True)
    alpha1_subCat0.setConstant(True)
    n1_subCat0.setConstant(True)
    alpha2_subCat0.setConstant(True)
    n2_subCat0.setConstant(True)

    # unfreeze the param for datacard
    CMS_hmm_sigma_cat0_ggh.setConstant(False)
    CMS_hmm_peak_cat0_ggh.setConstant(False)

    # subCat 1
    _ = signal_subCat1.fitTo(data_subCat1_signal,  EvalBackend=device, Save=True, )
    fit_result = signal_subCat1.fitTo(data_subCat1_signal,  EvalBackend=device, Save=True, )
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat1.setConstant(True)
    alpha1_subCat1.setConstant(True)
    n1_subCat1.setConstant(True)
    alpha2_subCat1.setConstant(True)
    n2_subCat1.setConstant(True)

    # unfreeze the param for datacard
    CMS_hmm_sigma_cat1_ggh.setConstant(False)
    CMS_hmm_peak_cat1_ggh.setConstant(False)

    # subCat 2
    _ = signal_subCat2.fitTo(data_subCat2_signal,  EvalBackend=device, Save=True, )
    fit_result = signal_subCat2.fitTo(data_subCat2_signal,  EvalBackend=device, Save=True, )
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat2.setConstant(True)
    alpha1_subCat2.setConstant(True)
    n1_subCat2.setConstant(True)
    alpha2_subCat2.setConstant(True)
    n2_subCat2.setConstant(True)

    # unfreeze the param for datacard
    CMS_hmm_sigma_cat2_ggh.setConstant(False)
    CMS_hmm_peak_cat2_ggh.setConstant(False)
    
    # subCat 3
    _ = signal_subCat3.fitTo(data_subCat3_signal,  EvalBackend=device, Save=True, )
    fit_result = signal_subCat3.fitTo(data_subCat3_signal,  EvalBackend=device, Save=True, )
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat3.setConstant(True)
    alpha1_subCat3.setConstant(True)
    n1_subCat3.setConstant(True)
    alpha2_subCat3.setConstant(True)
    n2_subCat3.setConstant(True)

    # unfreeze the param for datacard
    CMS_hmm_sigma_cat3_ggh.setConstant(False)
    CMS_hmm_peak_cat3_ggh.setConstant(False)

    # subCat 4
    _ = signal_subCat4.fitTo(data_subCat4_signal,  EvalBackend=device, Save=True, )
    fit_result = signal_subCat4.fitTo(data_subCat4_signal,  EvalBackend=device, Save=True, )
    # if fit_result is not None:
        # fit_result.Print()

    # freeze Signal's shape parameters before adding to workspace as specified in line 1339 of the Run2 RERECO AN
    sigma_subCat4.setConstant(True)
    alpha1_subCat4.setConstant(True)
    n1_subCat4.setConstant(True)
    alpha2_subCat4.setConstant(True)
    n2_subCat4.setConstant(True)

    # unfreeze the param for datacard
    CMS_hmm_sigma_cat4_ggh.setConstant(False)
    CMS_hmm_peak_cat4_ggh.setConstant(False)
    
    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    plot_save_path = f"./validation/figs/{args.year}"
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    # -------------------------------------------------------------------------
    # do signal plotting with fit and data
    # -------------------------------------------------------------------------
    
    # subCat 0
    
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat0_signal.GetName()
    data_subCat0_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat0.GetName()
    signal_subCat0.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat0.pdf")

    # subCat 1
    
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat1_signal.GetName()
    data_subCat1_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat1.GetName()
    signal_subCat1.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat1.pdf")

    # subCat 2
    
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat2_signal.GetName()
    data_subCat2_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat2.GetName()
    signal_subCat2.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat2.pdf")

    # subCat 3
    
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat3_signal.GetName()
    data_subCat3_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat3.GetName()
    signal_subCat3.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat3.pdf")

    # subCat 4
    
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)
    name = data_subCat4_signal.GetName()
    data_subCat4_signal.plotOn(frame, DataError="SumW2", Name=name)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "P")
    name = signal_subCat4.GetName()
    signal_subCat4.plotOn(frame, Name=name, LineColor=rt.kGreen)
    legend.AddEntry(frame.getObject(int(frame.numItems())-1),name, "L")
    
    frame.Draw()
    legend.Draw()
    
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{plot_save_path}/stage3_plot_{category}_subCat4.pdf")

    # ---------------------------------------------------
    # Save to Signal, Background and Data to Workspace
    # ---------------------------------------------------
    workspace_path = "./workspaces"
    
    # subCat 0 
    fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat0_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat0.SetName("data_cat0_ggh");
    corePdf_subCat0.SetName("bkg_cat0_ggh_pdf");
    bkg_subCat0_norm.SetName(corePdf_subCat0.GetName()+"_norm"); 
    # make norm for data
    nevents = roo_histData_subCat0.sumEntries()
    roo_histData_subCat0_norm = rt.RooRealVar(roo_histData_subCat0.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat0_norm);
    wout.Import(roo_histData_subCat0);
    wout.Import(cat_subCat0);
    wout.Import(bkg_subCat0_norm);
    wout.Import(corePdf_subCat0);
    # wout.Print();
    wout.Write();

    fout = rt.TFile(f"{workspace_path}/workspace_sig_cat0_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat0.SetName("ggH_cat0_ggh_pdf");
    roo_histData_subCat0_signal.SetName("data_ggH_cat0_ggh");
    sig_norm_subCat0.SetName(signal_subCat0.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat0);
    wout.Import(signal_subCat0); 
    wout.Import(roo_histData_subCat0_signal); 
    # wout.Print();
    wout.Write();
    

    # subCat 1 
    fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat1_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat1.SetName("data_cat1_ggh");
    corePdf_subCat1.SetName("bkg_cat1_ggh_pdf");
    bkg_subCat1_norm.SetName(corePdf_subCat1.GetName()+"_norm");
    # make norm for data
    nevents = roo_histData_subCat1.sumEntries()
    roo_histData_subCat1_norm = rt.RooRealVar(roo_histData_subCat1.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat1_norm);
    wout.Import(roo_histData_subCat1);
    wout.Import(cat_subCat1);
    wout.Import(bkg_subCat1_norm);
    wout.Import(corePdf_subCat1);
    # wout.Print();
    wout.Write();

    fout = rt.TFile(f"{workspace_path}/workspace_sig_cat1_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat1.SetName("ggH_cat1_ggh_pdf"); 
    roo_histData_subCat1_signal.SetName("data_ggH_cat1_ggh");
    sig_norm_subCat1.SetName(signal_subCat1.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat1);
    wout.Import(signal_subCat1); 
    wout.Import(roo_histData_subCat1_signal); 
    # wout.Print();
    wout.Write();

    # subCat 2
    fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat2_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat2.SetName("data_cat2_ggh");
    corePdf_subCat2.SetName("bkg_cat2_ggh_pdf");
    bkg_subCat2_norm.SetName(corePdf_subCat2.GetName()+"_norm");
    # make norm for data
    nevents = roo_histData_subCat2.sumEntries()
    roo_histData_subCat2_norm = rt.RooRealVar(roo_histData_subCat2.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat2_norm);
    wout.Import(roo_histData_subCat2);
    wout.Import(cat_subCat2);
    wout.Import(bkg_subCat2_norm);
    wout.Import(corePdf_subCat2);
    # wout.Print();
    wout.Write();

    fout = rt.TFile(f"{workspace_path}/workspace_sig_cat2_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat2.SetName("ggH_cat2_ggh_pdf"); 
    roo_histData_subCat2_signal.SetName("data_ggH_cat2_ggh");
    sig_norm_subCat2.SetName(signal_subCat2.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat2);
    wout.Import(signal_subCat2); 
    wout.Import(roo_histData_subCat2_signal); 
    # wout.Print();
    wout.Write();


    # subCat 3
    fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat3_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat3.SetName("data_cat3_ggh");
    corePdf_subCat3.SetName("bkg_cat3_ggh_pdf");
    bkg_subCat3_norm.SetName(corePdf_subCat3.GetName()+"_norm");
    # make norm for data
    nevents = roo_histData_subCat3.sumEntries()
    roo_histData_subCat3_norm = rt.RooRealVar(roo_histData_subCat3.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat3_norm);
    wout.Import(roo_histData_subCat3);
    wout.Import(cat_subCat3);
    wout.Import(bkg_subCat3_norm);
    wout.Import(corePdf_subCat3);
    # wout.Print();
    wout.Write();

    fout = rt.TFile(f"{workspace_path}/workspace_sig_cat3_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat3.SetName("ggH_cat3_ggh_pdf"); 
    roo_histData_subCat3_signal.SetName("data_ggH_cat3_ggh");
    sig_norm_subCat3.SetName(signal_subCat3.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat3);
    wout.Import(signal_subCat3); 
    wout.Import(roo_histData_subCat3_signal); 
    # wout.Print();
    wout.Write();

    # subCat 4
    fout = rt.TFile(f"{workspace_path}/workspace_bkg_cat4_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    roo_histData_subCat4.SetName("data_cat4_ggh");
    corePdf_subCat4.SetName("bkg_cat4_ggh_pdf");
    bkg_subCat4_norm.SetName(corePdf_subCat4.GetName()+"_norm");
    # make norm for data
    nevents = roo_histData_subCat4.sumEntries()
    roo_histData_subCat4_norm = rt.RooRealVar(roo_histData_subCat4.GetName()+"_norm","Background normalization value",nevents,0,3*nevents)
    wout.Import(roo_histData_subCat4_norm);
    wout.Import(roo_histData_subCat4);
    wout.Import(cat_subCat4);
    wout.Import(bkg_subCat4_norm);
    wout.Import(corePdf_subCat4);
    # wout.Print();
    wout.Write();

    fout = rt.TFile(f"{workspace_path}/workspace_sig_cat4_{category}.root","RECREATE")
    wout = rt.RooWorkspace("w","workspace")
    # matching names consistent with UCSD's naming scheme
    signal_subCat4.SetName("ggH_cat4_ggh_pdf"); 
    roo_histData_subCat4_signal.SetName("data_ggH_cat4_ggh");
    sig_norm_subCat4.SetName(signal_subCat4.GetName()+"_norm"); 
    wout.Import(sig_norm_subCat4);
    wout.Import(signal_subCat4); 
    wout.Import(roo_histData_subCat4_signal); 
    # wout.Print();
    wout.Write();
    
    # ---------------------------------------------------
    # Group plotting start here
    # ---------------------------------------------------
    
    
    # -------------------------------------------------------------------------
    # do signal plotting for all sub-Cats in one plot
    # -------------------------------------------------------------------------
    sig_dict_by_sample = {
        "ggh_signal" : [
            signal_subCat0, 
            signal_subCat1,
            signal_subCat2,
            signal_subCat3,
            signal_subCat4,
        ]
    }

    plotSigBySample(mass, sig_dict_by_sample, plot_save_path)
        

    # -------------------------------------------------------------------------
    # do Bkg plotting loop divided into core-function
    # -------------------------------------------------------------------------
    
    model_dict_by_coreFunction = {
        "BWZRedux" : [
            model_subCat0_BWZRedux, 
            model_subCat1_BWZRedux,
            model_subCat2_BWZRedux,
            model_subCat3_BWZRedux,
            model_subCat4_BWZRedux,
        ],
        "sumExp" : [
            model_subCat0_sumExp, 
            model_subCat1_sumExp,
            model_subCat2_sumExp,
            model_subCat3_sumExp,
            model_subCat4_sumExp,
        ],
        # "FEWZxBern" : [
        #     model_subCat0_FEWZxBern, 
        #     model_subCat1_FEWZxBern,
        #     model_subCat2_FEWZxBern,
        #     model_subCat3_FEWZxBern,
        #     model_subCat4_FEWZxBern,
        # ],
        "SMF" : [
            subCat0_SMF, 
            subCat1_SMF,
            subCat2_SMF,
            subCat3_SMF,
            subCat4_SMF,
        ],
    }
    rooHist_list = [ # for normalization histogram reference
        roo_histData_subCat0,
        roo_histData_subCat1,
        roo_histData_subCat2,
        roo_histData_subCat3,
        roo_histData_subCat4
    ]
    plotBkgByCoreFunc(mass, model_dict_by_coreFunction, rooHist_list, plot_save_path)
    

    # -------------------------------------------------------------------------
    # do Bkg plotting loop divided into Sub Categories
    # -------------------------------------------------------------------------

    model_dict_by_subCat = {
        0 : [
            model_subCat0_BWZRedux, 
            model_subCat0_sumExp,
            # model_subCat0_FEWZxBern,
        ],
        1 : [
            model_subCat1_BWZRedux, 
            model_subCat1_sumExp,
            # model_subCat1_FEWZxBern,
        ],
        2 : [
            model_subCat2_BWZRedux, 
            model_subCat2_sumExp,
            # model_subCat2_FEWZxBern,
        ],
        3 : [
            model_subCat3_BWZRedux, 
            model_subCat3_sumExp,
            # model_subCat3_FEWZxBern,
        ],
        4 : [
            model_subCat4_BWZRedux, 
            model_subCat4_sumExp,
            # model_subCat4_FEWZxBern,
        ],
    }
    data_dict_by_subCat = {
        0 : roo_histData_subCat0,
        1 : roo_histData_subCat1,
        2 : roo_histData_subCat2,
        3 : roo_histData_subCat3,
        4 : roo_histData_subCat4,
    }
    plotBkgBySubCat(mass, model_dict_by_subCat, data_dict_by_subCat, plot_save_path)

    


