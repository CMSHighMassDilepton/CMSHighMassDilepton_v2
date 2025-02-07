import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client
from omegaconf import OmegaConf

from typing import Tuple, List, Dict
import ROOT as rt
import time


from quickSMFtest_functions import MakeBWZ_Redux, MakeFEWZxBern, MakeSumExponential

if __name__ == "__main__":
    """
    loading stage1 output data and evalauting BDT is deletegated to run_stage2.py
    """
    load_path = "/work/users/yun79/stage2_output/test/processed_events_data.parquet"
    processed_eventsData = ak.from_parquet(load_path)
    start_time = time.time()

    
    print("events loaded!")
    
    # comence roofit fitting for each subcategory 
    n_subCats = 5
    poly_order_by_cat = {
        0:3,
        1:3,
        2:2,
        3:2,
        4:2,
    }
    subCatColorMap = {
        "subCat0":rt.kBlue,
        "subCat1":rt.kGreen,
        "subCat2":rt.kRed,
        "subCat3":rt.kYellow,
        "subCat4":rt.kOrange,
    }
    dof = 3 # degrees of freedom for the core-functions. This should be same for all the functions
    smf_coeffStartVals = {
        0: {
            1: 0.330, # smf val start with index one bc zeroth order coeff is assumed to be zero for RooChebychev
            2: 0.142,
            3: -0.0283,
        },
        1: {
            1: 0.5,
            2: 0.2,
            3: -0.002,
        },
        2: {
            1: 0.4,
            2: 0.2,
        },
        3: {
            1: 0.4,
            2: 0.2,
        },
        4: {
            1: 0.4,
            2: 0.2,
        },
    }
    
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    nbins = 800 # Bin size = 50 MeV -> line 1762 of RERECO AN
    mass_name = f"mh_ggh"
    mass =  rt.RooRealVar(mass_name,mass_name,120,110,150)
    mass.setBins(nbins)
    # defin fit range
    mass.setRange("hiSB", 135, 150 )
    mass.setRange("loSB", 110, 115 )
    mass.setRange("h_peak", 115, 135 )
    mass.setRange("full", 110, 150 )
    # fit_range = "loSB,hiSB" # we're fitting bkg only
    fit_range = "hiSB,loSB" # we're fitting bkg only

    # Define category to distinguish physics and control samples events
    sample = rt.RooCategory("sample", "sample")
    subCat_dataHists = {}
    model_dict = {}
    misc_dict = {} # to keep python from erasing variables for Roofit to run properly

    # intialize the core functions 
    BWZ_Redux, params_redux =  MakeBWZ_Redux(mass, dof)
    FEWZxBern, params_fewz = MakeFEWZxBern(mass, dof)
    sumExp, params_exp = MakeSumExponential(mass, dof)

    core_models = [
        BWZ_Redux, 
        # FEWZxBern, 
        # sumExp,
    ]
    
    rt.EnableImplicitMT() # for fitting
    # for cat_ix in range(5):
    for cat_ix in [0,1]:
        subCat_name = f"subCat{cat_ix}"
        smfVarList = []
        smf_order= poly_order_by_cat[cat_ix]
        print(f"smf_order: {smf_order}")
        for ix in range(smf_order): 
            name = subCat_name+f"_SMF_Coeff{ix+1}"
            try:
                start_val = smf_coeffStartVals[cat_ix][ix+1]
            except:
                start_val = -0.005
            smf_coeff = rt.RooRealVar(name, name,  start_val, -10, 10)
                
            smfVarList.append(smf_coeff)
        name = subCat_name + "_SMF"
        polynomial_model = rt.RooChebychev(name, name, mass, smfVarList)

        # add model and variable info so they don't get deleted
        misc_dict[polynomial_model.GetName()] = polynomial_model
        for smf_coeff in smfVarList:
            misc_dict[smf_coeff.GetName()] = smf_coeff

        
        for core_model in core_models:
            subCat_name4coreModel = subCat_name + "_" + core_model.GetName()
            sample.defineType(subCat_name4coreModel)
            print(f"subCat_name4coreModel :{subCat_name4coreModel}")
            # generate
            subCat_filter = (processed_eventsData["subCategory_idx"] == cat_ix)
            subCat_mass_arr = processed_eventsData.dimuon_mass[subCat_filter]
            subCat_mass_arr  = ak.to_numpy(subCat_mass_arr) # convert to numpy for rt.RooDataSet
            roo_datasetData = rt.RooDataSet.from_numpy({mass_name: subCat_mass_arr}, [mass])
            roo_datasetData.SetName(subCat_name4coreModel+"_Dataset_Data")
            roo_histData = rt.RooDataHist(subCat_name4coreModel+"_DataHist_Data",f"binned version of roo_datasetData of subcat {cat_ix}", rt.RooArgSet(mass), roo_datasetData)
            print(f"roo_datasetData.GetName(): {roo_datasetData.GetName()}")
    
            
            name = subCat_name4coreModel+"_final"
            final_model = rt.RooProdPdf(name, name, [polynomial_model,core_model])
            print(f"final_model name: {name}")
            
            # add the information to relevant dictionaries
            model_dict[subCat_name4coreModel] = final_model
            subCat_dataHists[subCat_name4coreModel] = roo_histData
            misc_dict[roo_datasetData.GetName()] = roo_datasetData
            
            # # fit model individually once b4 doing simiultaneous fit 
            # _ = final_model.fitTo(roo_histData, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, Strategy=0)

    print(f"misc_dict: {misc_dict}")
    print(f"model_dict: {model_dict}")
    # Construct combined dataset
    combData = rt.RooDataSet(
        "combData",
        "combined data",
        {mass},
        Index=sample,
        Import=subCat_dataHists,
    ) 
    # combData.Print("v")
    sample.Print("v")


    # Construct a simultaneous pdf using category sample as index: associate model
    simPdf = rt.RooSimultaneous("simPdf", "simultaneous pdf", model_dict, sample)
    
    _ = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, Strategy=0)
    fit_result = simPdf.fitTo(combData, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True, Strategy=0)

    end_time  =time.time()
    print(f"time for fitting to take plsace: {end_time-start_time} sec")
    # # save workspace for plotting later
    # fout = rt.TFile("./simultFitTestWorkspace.root","RECREATE")
    # wout = rt.RooWorkspace("workspace","workspace")
    # wout.Import(roo_datasetData);
    # for model_name, model in model_dict.items():
    #     print(f"model_name: {model_name}")
    #     wout.Import(model);
    # wout.Print();
    # wout.Write();
    # plot 
    frame = mass.frame()
    legend = rt.TLegend(0.65,0.55,0.9,0.7)

    for core_model in core_models:
        
        canvas.Clear()
        legend.Clear()
        frame.Clear()
        # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
        roo_datasetData.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0) )

        core_modelName = core_model.GetName()
        for model_name, model in model_dict.items():
            if core_modelName in model_name:
                color = rt.kBlue
                for subCat, col in subCatColorMap.items():
                    if subCat in model_name:
                        color = col
                model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=model.GetName(), LineColor=color)
                # model.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name=model.GetName())
                legend.AddEntry(frame.getObject(int(frame.numItems())-1),model.GetName(), "L")
    
        frame.Draw()
        legend.Draw()        
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs(f"./quick_plots/simultaneousPlotTest_{core_modelName}.pdf")
        
        