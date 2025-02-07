import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client

"""
time to fit : define the core functions with their parameters
"""
from typing import Tuple, List, Dict
import ROOT as rt

def MakeBWZ_Redux(mass: rt.RooRealVar, order: int) ->Tuple[rt.RooAddPdf, Dict]:
    # collect all variables that we don't want destroyed by Python once function ends
    out_dict = {}
    
    name = f"BWZ_Redux_a_coeff"
    a_coeff = rt.RooRealVar(name,name, -0.00001,-0.001,0.001)
    name = "exp_model_mass"
    exp_model_mass = rt.RooExponential(name, name, mass, a_coeff)
    
    mass_sq = rt.RooFormulaVar("mass_sq", "@0*@0", rt.RooArgList(mass))
    name = f"BWZ_Redux_b_coeff"
    b_coeff = rt.RooRealVar(name,name, -0.00001,-0.001,0.001)
    
    name = "exp_model_mass_sq"
    exp_model_mass_sq = rt.RooExponential(name, name, mass_sq, b_coeff)

    # add in the variables and models
    out_dict[a_coeff.GetName()] = a_coeff 
    out_dict[exp_model_mass.GetName()] = exp_model_mass
    out_dict[mass_sq.GetName()] = mass_sq
    out_dict[b_coeff.GetName()] = b_coeff
    out_dict[exp_model_mass_sq.GetName()] = exp_model_mass_sq
    
    # make Z boson related stuff
    bwWidth = rt.RooRealVar("bwWidth", "bwWidth", 2.5, 0, 30)
    bwmZ = rt.RooRealVar("bwmZ", "bwmZ", 91.2, 90, 92)
    bwWidth.setConstant(True)
    bwmZ.setConstant(True)

    # start multiplying them all
    name = f"BWZ_Redux_c_coeff"
    c_coeff = rt.RooRealVar(name,name, 2,-5.0,5.0)
    BWZ_redux_main = rt.RooGenericPdf(
        "BWZ_redux_main", "@1/ ( pow((@0-@2), @3) + 0.25*pow(@1, @3) )", rt.RooArgList(mass, bwWidth, bwmZ, c_coeff)
    )
    # add in the variables and models
    out_dict[bwWidth.GetName()] = bwWidth 
    out_dict[bwmZ.GetName()] = bwmZ 
    out_dict[c_coeff.GetName()] = c_coeff 
    out_dict[BWZ_redux_main.GetName()] = BWZ_redux_main 

    name = "BWZ_Redux"
    final_model = rt.RooProdPdf(name, name, [BWZ_redux_main, exp_model_mass, exp_model_mass_sq]) 
    return (final_model, out_dict)

def MakeBWZxBern(mass: rt.RooRealVar, order: int) ->Tuple[rt.RooAddPdf, Dict]:
    """
    params:
    mass = rt.RooRealVar that we will fitTo
    order = order of the sum of exponential, that we assume to be >= 2
    """
    # collect all variables that we don't want destroyed by Python once function ends
    out_dict = {}


    
    # make BernStein
    bern_order = order-1
    BernCoeff_list = []
    for ix in range(bern_order):
        name = f"Bernstein_c_{ix}"
        if ix == 0:
            coeff = rt.RooRealVar(name,name, 1,-5.0,5.0)
        else:
            coeff = rt.RooRealVar(name,name, 1,-5.0,5.0)
        out_dict[name] = coeff # add variable to make python remember 
        BernCoeff_list.append(coeff)
    name = f"Bernstein_model_order_{bern_order}"
    bern_model = rt.RooBernstein(name, name, mass, BernCoeff_list)
    out_dict[name] = bern_model # add variable to make python remember

    
    # make BWZ
    bwWidth = rt.RooRealVar("bwWidth", "bwWidth", 2.5, 0, 30)
    bwmZ = rt.RooRealVar("bwmZ", "bwmZ", 91.2, 90, 92)
    bwWidth.setConstant(True)
    bwmZ.setConstant(True)
    out_dict[bwWidth.GetName()] = bwWidth 
    out_dict[bwmZ.GetName()] = bwmZ 
    
    name = "VanillaBW_model"
    BWZ = rt.RooBreitWigner(name, name, mass, bwmZ,bwWidth)
    # our BWZ model is also multiplied by exp(a* mass) as defined in the AN
    name = "BWZ_exp_coeff"
    expCoeff = rt.RooRealVar(name, name, -0.0, -3.0, 1.0)
    name = "BWZ_exp_model"
    exp_model = rt.RooExponential(name, name, mass, expCoeff)
    # name = "BWZxExp"
    # full_BWZ = rt.RooProdPdf(name, name, [BWZ, exp_model]) 

    # add variables
    out_dict[BWZ.GetName()] = BWZ 
    out_dict[expCoeff.GetName()] = expCoeff 
    out_dict[exp_model.GetName()] = exp_model 
    # out_dict[full_BWZ.GetName()] = full_BWZ 
    
    # multiply BWZ and Bernstein
    name = f"BWZxBern_order_{order}"
    # final_model = rt.RooProdPdf(name, name, [bern_model, full_BWZ]) 
    final_model = rt.RooProdPdf(name, name, [bern_model, BWZ, exp_model]) 
   
    return (final_model, out_dict)
    

def MakeSumExponential(mass: rt.RooRealVar, order: int) ->Tuple[rt.RooAddPdf, Dict]:
    """
    params:
    mass = rt.RooRealVar that we will fitTo
    order = order of the sum of exponential, that we assume to be >= 2
    returns:
    rt.RooAddPdf
    dictionary of variables with {variable name : rt.RooRealVar or rt.RooExponential} format mainly for keep python from
    destroying these variables, but also useful in debugging
    """
    model_list = [] # list of RooExp models for RooAddPdf
    a_i_list = [] # list of RooExp coeffs for RooAddPdf
    rest_list = [] # list of rest of variables to save it from being destroyed
    for ix in range(order):
        name = f"S_exp_b_{ix}"
        b_i = rt.RooRealVar(name, name, -0.5, -5.0, 1.0)
        rest_list.append(b_i)
        
        name = f"S_exp_model_{ix}"
        model = rt.RooExponential(name, name, mass, b_i)
        model_list.append(model)
        
        if ix >0:
            name = f"S_exp_a_{ix}"
            a_i = rt.RooRealVar(name, name, 0.5, 0, 1.0)
            a_i_list.append(a_i)
            
    name = f"S_exp_order_{order}"
    final_model = rt.RooAddPdf(name, name, model_list, a_i_list)
    # collect all variables that we don't want destroyed by Python once function ends
    out_dict = {}
    for model in model_list:
        out_dict[model.GetName()] = model
    for a_i in a_i_list:
        out_dict[a_i.GetName()] = a_i
    for var in rest_list:
        out_dict[var.GetName()] = var
    return (final_model, out_dict)


if __name__ == "__main__":
    client =  Client(n_workers=31,  threads_per_worker=1, processes=True, memory_limit='4 GiB') 
    load_path = "/depot/cms/users/yun79/results/stage1/test_VBF-filter_JECon_07June2024/2018/f1_0/"
    full_load_path = load_path+f"/data_C/*/*.parquet"
    events = dak.from_parquet(full_load_path)
    # figure out the discontinuous fit range later ---------------------
    
    mass_arr = ak.to_numpy(events.dimuon_mass.compute())
    
    # start Root fit 
    name = "Canvas"
    canvas = rt.TCanvas(name,name,800, 800) # giving a specific name for each canvas prevents segfault?
    canvas.cd()
    mass_name = "dimuon_mass"
    mass =  rt.RooRealVar(mass_name,"mass (GeV)",120,110,150)
    nbins = 80
    mass.setBins(nbins)

    # for debugging purposes -----------------
    binning = np.linspace(110, 150, nbins)
    np_hist, _ = np.histogram(mass_arr, bins=binning)
    print(f"np_hist: {np_hist}")
    # -------------------------------------------


    
    roo_dataset = rt.RooDataSet.from_numpy({mass_name: mass_arr}, [mass])

    # set sideband mass range after initializing dataset (idk why this order matters, but that's how it's shown here https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/tutorial2023/parametric_exercise/?h=sideband#background-modelling)
    mass.setRange("loSB", 110, 115 )
    mass.setRange("hiSB", 135, 150 )
    mass.setRange("h_peak", 115, 135 )
    mass.setRange("full", 110, 150 )
    fit_range = "loSB,hiSB" # we're fitting bkg only

    
    order = 3
    BWZxBern, params_bern = MakeBWZxBern(mass, order)
    sumExp, params_exp = MakeSumExponential(mass, order)
    BWZ_Redux, params_redux =  MakeBWZ_Redux(mass, order)
    #-------------
    # # collect all variables that we don't want destroyed by Python once function ends
    # out_dict = {}
    
    # name = f"BWZ_Redux_a_coeff"
    # a_coeff = rt.RooRealVar(name,name, -0.00001,-0.001,0.001)
    # name = "exp_model_mass"
    # exp_model_mass = rt.RooExponential(name, name, mass, a_coeff)
    
    # mass_sq = rt.RooFormulaVar("mass_sq", "@0*@0", rt.RooArgList(mass))
    # name = f"BWZ_Redux_b_coeff"
    # b_coeff = rt.RooRealVar(name,name, -0.00001,-0.001,0.001)
    
    # name = "exp_model_mass_sq"
    # exp_model_mass_sq = rt.RooExponential(name, name, mass_sq, b_coeff)

    # # add in the variables and models
    # out_dict[a_coeff.GetName()] = a_coeff 
    # out_dict[exp_model_mass.GetName()] = exp_model_mass
    # out_dict[mass_sq.GetName()] = mass_sq
    # out_dict[b_coeff.GetName()] = b_coeff
    # out_dict[exp_model_mass_sq.GetName()] = exp_model_mass_sq
    
    # # make Z boson related stuff
    # bwWidth = rt.RooRealVar("bwWidth", "bwWidth", 2.5, 0, 30)
    # bwmZ = rt.RooRealVar("bwmZ", "bwmZ", 91.2, 90, 92)
    # bwWidth.setConstant(True)
    # bwmZ.setConstant(True)

    # # start multiplying them all
    # name = f"BWZ_Redux_c_coeff"
    # c_coeff = rt.RooRealVar(name,name, 2,-5.0,5.0)
    # BWZ_redux_main = rt.RooGenericPdf(
    #     "BWZ_redux_main", "@1/ ( pow((@0-@2), @3) + 0.25*pow(@1, @3) )", rt.RooArgList(mass, bwWidth, bwmZ, c_coeff)
    # )
    # # add in the variables and models
    # out_dict[bwWidth.GetName()] = bwWidth 
    # out_dict[bwmZ.GetName()] = bwmZ 
    # out_dict[c_coeff.GetName()] = c_coeff 
    # out_dict[BWZ_redux_main.GetName()] = BWZ_redux_main 

    # name = "BWZ_Redux"
    # BWZ_Redux = rt.RooProdPdf(name, name, [BWZ_redux_main, exp_model_mass, exp_model_mass_sq]) 
    #-----------------
   
    # print(f"params: {params}")
    # roo_dataset.Print()
    roo_hist = rt.RooDataHist("data_hist","binned version of roo_dataset", rt.RooArgSet(mass), roo_dataset)  # copies binning from mass variable
    # roo_hist.Print()

    # # begin multi-pdf
    # cat = rt.RooCategory("pdf_index","Index of Pdf which is active")
    # pdflist = rt.RooArgList()
    # # Make a RooMultiPdf object. The order of the pdfs will be the order of their index, ie for below
    # # 0 == BWZxBern
    # # 1 == sumExp
    # pdflist.add(BWZxBern)
    # pdflist.add(sumExp)
    
    # multipdf = rt.RooMultiPdf(
    #     # f"multipdf_{self.channel}_{category}", 
    #     "multipdf",
    #     "multipdf", 
    #     cat, 
    #     pdflist
    # )
    

    rt.EnableImplicitMT()
    # _ = BWZxBern.fitTo(roo_hist, rt.RooFit.Range(fit_range), Save=True,  EvalBackend ="cpu")
    # fit_result = BWZxBern.fitTo(roo_hist, rt.RooFit.Range(fit_range), Save=True,  EvalBackend ="cpu")
    # _ = BWZxBern.fitTo(roo_hist, rt.RooFit.Range(fit_range),  rt.RooFit.BatchMode("cpu"), Save=True, )
    # fit_result = BWZxBern.fitTo(roo_hist, rt.RooFit.Range(fit_range),  rt.RooFit.BatchMode("cpu"), Save=True, )
    # _ = sumExp.fitTo(roo_hist, rt.RooFit.Range(fit_range),  rt.RooFit.BatchMode("cpu"), Save=True, )
    # fit_result = sumExp.fitTo(roo_hist, rt.RooFit.Range(fit_range),  rt.RooFit.BatchMode("cpu"), Save=True, )
    
    # _ = sumExp.fitTo(roo_hist, rt.RooFit.Range(fit_range),  EvalBackend="cpu", Save=True, )
    # fit_result = sumExp.fitTo(roo_hist, rt.RooFit.Range(fit_range),  EvalBackend="cpu", Save=True, )
    _ = BWZ_Redux.fitTo(roo_hist, rt.RooFit.Range(fit_range),  EvalBackend="cpu", Save=True, )
    fit_result = BWZ_Redux.fitTo(roo_hist, rt.RooFit.Range(fit_range),  EvalBackend="cpu", Save=True, )
    

    #Core-PDF
    # _ = BWZxBern.fitTo(roo_hist, rt.RooFit.Range(fit_range),  rt.RooFit.BatchMode("cpu"), Save=True, )
    # _ = sumExp.fitTo(roo_hist, rt.RooFit.Range(fit_range),  rt.RooFit.BatchMode("cpu"), Save=True, )
    # fit_result = multipdf.fitTo(roo_hist, rt.RooFit.Range(fit_range),  rt.RooFit.BatchMode("cpu"), Save=True, )
    
    # _ = BWZxBern.fitTo(roo_hist, Save=True,  EvalBackend ="cpu")
    # fit_result = BWZxBern.fitTo(roo_hist, Save=True,  EvalBackend ="cpu")

    # draw on canvas
    frame = mass.frame()

    # apparently I have to plot invisible roo dataset for fit function plotting to work. Maybe this helps with normalization?
    roo_dataset.plotOn(frame, rt.RooFit.MarkerColor(0), rt.RooFit.LineColor(0) )
    # BWZxBern.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name="BWZxBern", LineColor=rt.kGreen)
    # sumExp.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name="Sum Exponential", LineColor=rt.kGreen)
    BWZ_Redux.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name="NWZ Redux", LineColor=rt.kGreen)
    # multipdf.plotOn(frame, rt.RooFit.NormRange(fit_range), rt.RooFit.Range("full"), Name="MultiPdf", LineColor=rt.kGreen)
    roo_dataset.plotOn(frame, rt.RooFit.CutRange(fit_range), DataError="SumW2", Name="data_hist")

    frame.Draw()
    canvas.Update()
    canvas.Draw()
    # canvas.SaveAs(f"./quick_plots/stage3_plot_test_BWZxBern.pdf")
    # canvas.SaveAs(f"./quick_plots/stage3_plot_test_sumExponent.pdf")
    canvas.SaveAs(f"./quick_plots/stage3_plot_test_BWZRedux.pdf")
    