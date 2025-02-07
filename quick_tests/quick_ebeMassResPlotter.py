import ROOT as rt
import dask_awkward as dak
import awkward as ak
import numpy as np
import json
import argparse
import os
from distributed import Client
import time    
import tqdm
import mplhep as hep
import matplotlib.pyplot as plt
import glob
import time
from dask_gateway import Gateway


def get_calib_categories(events):
    BB = ((abs(events["mu1_eta"])<=0.9) & (abs(events["mu2_eta"])<=0.9))
    BO = ((abs(events["mu1_eta"])<=0.9) & ((abs(events["mu2_eta"])>0.9) & (abs(events["mu2_eta"]) <=1.8)))
    BE = ((abs(events["mu1_eta"])<=0.9) & ((abs(events["mu2_eta"])>1.8) & (abs(events["mu2_eta"]) <=2.4)))
    OB = (((abs(events["mu1_eta"])>0.9) & (abs(events["mu1_eta"]) <=1.8)) & (abs(events["mu2_eta"])<=0.9))
    OO = (((abs(events["mu1_eta"])>0.9) & (abs(events["mu1_eta"]) <=1.8)) & ((abs(events["mu2_eta"])>0.9) & (abs(events["mu2_eta"]) <=1.8)))
    OE = (((abs(events["mu1_eta"])>0.9) & (abs(events["mu1_eta"]) <=1.8)) & ((abs(events["mu2_eta"])>1.8) & (abs(events["mu2_eta"]) <=2.4)))
    EB = (((abs(events["mu1_eta"])>1.8) & (abs(events["mu1_eta"]) <=2.4)) & (abs(events["mu2_eta"])<=0.9))
    EO = (((abs(events["mu1_eta"])>1.8) & (abs(events["mu1_eta"]) <=2.4)) & ((abs(events["mu2_eta"])>0.9) & (abs(events["mu2_eta"]) <=1.8)))
    EE = (((abs(events["mu1_eta"])>1.8) & (abs(events["mu1_eta"]) <=2.4)) & ((abs(events["mu2_eta"])>1.8) & (abs(events["mu2_eta"]) <=2.4)))
    categories = [((events["mu1_pt"]>30)&(events["mu1_pt"]<=45)&(BB | OB | EB)),
                          ((events["mu1_pt"]>30)&(events["mu1_pt"]<=45)&(BO | OO | EO)),
                          ((events["mu1_pt"]>30)&(events["mu1_pt"]<=45)&(BE | OE | EE)),
                          ((events["mu1_pt"]>45)&(events["mu1_pt"]<=52)&BB),
                          ((events["mu1_pt"]>45)&(events["mu1_pt"]<=52)&BO),
                          ((events["mu1_pt"]>45)&(events["mu1_pt"]<=52)&BE),
                          ((events["mu1_pt"]>45)&(events["mu1_pt"]<=52)&OB),
                          ((events["mu1_pt"]>45)&(events["mu1_pt"]<=52)&OO),
                          ((events["mu1_pt"]>45)&(events["mu1_pt"]<=52)&OE),
                          ((events["mu1_pt"]>45)&(events["mu1_pt"]<=52)&EB),
                          ((events["mu1_pt"]>45)&(events["mu1_pt"]<=52)&EO),
                          ((events["mu1_pt"]>45)&(events["mu1_pt"]<=52)&EE),
                  # voigtian start here onwards
                          ((events["mu1_pt"]>52)&(events["mu1_pt"]<=62)&BB),
                          ((events["mu1_pt"]>52)&(events["mu1_pt"]<=62)&BO),
                          ((events["mu1_pt"]>52)&(events["mu1_pt"]<=62)&BE),
                          ((events["mu1_pt"]>52)&(events["mu1_pt"]<=62)&OB),
                          ((events["mu1_pt"]>52)&(events["mu1_pt"]<=62)&OO),
                          ((events["mu1_pt"]>52)&(events["mu1_pt"]<=62)&OE),
                          ((events["mu1_pt"]>52)&(events["mu1_pt"]<=62)&EB),
                          ((events["mu1_pt"]>52)&(events["mu1_pt"]<=62)&EO),
                          ((events["mu1_pt"]>52)&(events["mu1_pt"]<=62)&EE),
                          ((events["mu1_pt"]>62)&(events["mu1_pt"]<=200)&BB),
                          ((events["mu1_pt"]>62)&(events["mu1_pt"]<=200)&BO),
                          ((events["mu1_pt"]>62)&(events["mu1_pt"]<=200)&BE),
                          ((events["mu1_pt"]>62)&(events["mu1_pt"]<=200)&OB),
                          ((events["mu1_pt"]>62)&(events["mu1_pt"]<=200)&OO),
                          ((events["mu1_pt"]>62)&(events["mu1_pt"]<=200)&OE),
                          ((events["mu1_pt"]>62)&(events["mu1_pt"]<=200)&EB),
                          ((events["mu1_pt"]>62)&(events["mu1_pt"]<=200)&EO),
                          ((events["mu1_pt"]>62)&(events["mu1_pt"]<=200)&EE),]
    return categories



def generateVoigtian_plot(mass_arr, cat_idx: int):
    """
    params
    mass_arr: numpy arrary of dimuon mass value to do calibration fit on
    cat_idx: int index of specific calibration category the mass_arr is from
    """
    canvas = rt.TCanvas(str(cat_idx),str(cat_idx)) # giving a specific name for each canvas prevents segfault
    canvas.cd()
    # workspace = rt.RooWorkspace("w", "w")
    mass =  rt.RooRealVar("dimuon_mass","mass (GeV)",100,np.min(mass_arr),np.max(mass_arr))
    mass.setBins(200)
    roo_dataset = rt.RooDataSet.from_numpy({"dimuon_mass": mass_arr}, [mass])
    # workspace.Import(mass)
    frame = mass.frame()

    # Voigtian --------------------------------------------------------------------------
    bwmZ = rt.RooRealVar("bwz_mZ" , "mZ", 91.1876, 91, 92)
    bwWidth = rt.RooRealVar("bwz_Width" , "widthZ", 2.4952, 1, 3)
    sigma = rt.RooRealVar("sigma" , "sigma", 2, 1.5, 2.5)
    bwWidth.setConstant(True)
    model1 = rt.RooVoigtian("signal" , "signal", mass, bwmZ, bwWidth, sigma)



    # # Exp x Erfc Background --------------------------------------------------------------------------
    # # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", 0.01, 0.00000001, 1) # positve coeff to get the peak shape we want 
    # exp_coeff = rt.RooRealVar("exp_coeff", "exp_coeff", -0.1, -1, -0.00000001) # negative coeff to get the peak shape we want 
    # shift = rt.RooRealVar("shift", "Offset", 85, 75, 105)
    # shifted_mass = rt.RooFormulaVar("shifted_mass", "@0-@1", rt.RooArgList(mass, shift))
    # model2_1 = rt.RooExponential("Exponential", "Exponential", shifted_mass,exp_coeff)
    
    # erfc_center = rt.RooRealVar("erfc_center" , "erfc_center", 91.2, 75, 105)
    # erfc_coeff = rt.RooRealVar("erfc_coeff" , "erfc_coeff", 0.1, 0, 1.5)
    # erfc_in = rt.RooFormulaVar("erfc_in", "(@0 - @2) * @1", rt.RooArgList(mass, erfc_coeff, erfc_center)) 
    # model2_2a = rt.RooFit.bindFunction("erfc", rt.TMath.Erfc, erfc_in) # turn TMath function to Roofit funciton
    # model2_2 = rt.RooWrapperPdf("erfc","erfc", model2_2a) # turn bound function to pdf
    # model2 = rt.RooProdPdf("bkg", "bkg", [model2_1, model2_2]) # generate ExpxErfc bkg  
    
    # Landau Background --------------------------------------------------------------------------
    mean = rt.RooRealVar("mean" , "mean", 95, 90, 100)
    sigma_landau = rt.RooRealVar("sigma_landau" , "sigma_landau", 2, 1.5, 2.5)
    model2 = rt.RooLandau("bkg", "bkg", mass, mean, sigma_landau) # generate Landau bkg  

    
    sigfrac = rt.RooRealVar("sigfrac", "sigfrac", 0.9, 0, 1.0)
    final_model = rt.RooAddPdf("final_model", "final_model", rt.RooArgList(model1, model2),rt.RooArgList(sigfrac))


    time_step = time.time()
    
    #fitting directly to unbinned dataset is slow, so first make a histogram
    roo_hist = rt.RooDataHist("data_hist","binned version of roo_dataset", rt.RooArgSet(mass), roo_dataset)  # copies binning from mass variable
    # do fitting
    rt.EnableImplicitMT()
    # final_model.fitTo(roo_hist, rt.RooFit.Save(), EvalBackend ="cpu")
    final_model.fitTo(roo_hist, rt.RooFit.PrefitDataFraction(0.8), Save=True,  EvalBackend ="cpu")
    print(f"fitting elapsed time: {time.time() - time_step}")
    #do plotting
    roo_dataset.plotOn(frame)
    final_model.plotOn(frame, Name="final_model", LineColor=rt.kGreen)
    final_model.plotOn(frame, Components="signal", LineColor=rt.kBlue)
    final_model.plotOn(frame, Components="bkg", LineColor=rt.kRed)
    
    frame.Draw()
    canvas.Update()
    # canvas.Draw()
    canvas.SaveAs(f"calibration_fitCat{cat_idx}.pdf")
    del canvas
    # consider script to wait a second for stability?
    time.sleep(1)

def generateBWxDCB_plot(mass_arr, cat_idx: int):
    """
    params
    mass_arr: numpy arrary of dimuon mass value to do calibration fit on
    cat_idx: int index of specific calibration category the mass_arr is from
    """
    canvas = rt.TCanvas(str(cat_idx),str(cat_idx)) # giving a specific name for each canvas prevents segfault
    canvas.cd()
    # workspace = rt.RooWorkspace("w", "w")
    mass =  rt.RooRealVar("dimuon_mass","mass (GeV)",100,np.min(mass_arr),np.max(mass_arr))
    mass.setBins(200)
    roo_dataset = rt.RooDataSet.from_numpy({"dimuon_mass": mass_arr}, [mass])
    # workspace.Import(mass)
    frame = mass.frame()

    # BWxDCB --------------------------------------------------------------------------
    bwmZ = rt.RooRealVar("bwz_mZ" , "mZ", 91.1876, 91, 92)
    bwWidth = rt.RooRealVar("bwz_Width" , "widthZ", 2.4952, 1, 3)
    bwmZ.setConstant(True)
    bwWidth.setConstant(True)
    
    
    model1_1 = rt.RooBreitWigner("bwz", "BWZ",mass, bwmZ, bwWidth)
    
    """
    Note from Jan: sometimes freeze n values in DCB to be frozen (ie 1, but could be other values)
    This is because alpha and n are highly correlated, so roofit can be really confused.
    Also, given that we care about the resolution, not the actual parameter values alpha and n, we can 
    put whatevere restrictions we want.
    """
    mean = rt.RooRealVar("mean" , "mean", 0, -10,10) # mean is mean relative to BW
    sigma = rt.RooRealVar("sigma" , "sigma", 2, .1, 4.0)
    alpha1 = rt.RooRealVar("alpha1" , "alpha1", 2, 0.01, 65)
    n1 = rt.RooRealVar("n1" , "n1", 10, 0.01, 185)
    alpha2 = rt.RooRealVar("alpha2" , "alpha2", 2.0, 0.01, 65)
    n2 = rt.RooRealVar("n2" , "n2", 25, 0.01, 385)
    n1.setConstant(True)
    n2.setConstant(True)
    model1_2 = rt.RooCrystalBall("dcb","dcb",mass, mean, sigma, alpha1, n1, alpha2, n2)
    
    # merge BW with DCB via convolution
    model1 = rt.RooFFTConvPdf("signal", "signal", mass, model1_1, model1_2) # BWxDCB
    
    
    mass.setBins(10000,"cache") # This nbins has nothing to do with actual nbins of mass. cache bins is representation of the variable only used in FFT
    mass.setMin("cache",50.5) 
    mass.setMax("cache",130.5)

    # Background --------------------------------------------------------------------------
    coeff = rt.RooRealVar("coeff", "coeff", 0.01, 0.00000001, 1)
    # shift = rt.RooRealVar("shift" , "shift", 92, -10, 1000)
    # coeff = rt.RooRealVar("coeff", "Slope", 0.0001, -1.5, 0)
    shift = rt.RooRealVar("shift", "Offset", 85, 75, 105)
    shifted_mass = rt.RooFormulaVar("shifted_mass", "@0-@1", rt.RooArgList(mass, shift))
    model2 = rt.RooExponential("bkg", "bkg", shifted_mass, coeff)
    
    sigfrac = rt.RooRealVar("sigfrac", "sigfrac", 0.9, 0, 1.0)
    # final_model = rt.RooAddPdf("final_model", "final_model", rt.RooArgList(model1, model2),sigfrac)
    final_model = rt.RooAddPdf("final_model", "final_model", [model1, model2],[sigfrac])
    


    time_step = time.time()
    
    #fitting directly to unbinned dataset is slow, so first make a histogram
    roo_hist = rt.RooDataHist("data_hist","binned version of roo_dataset", rt.RooArgSet(mass), roo_dataset)  # copies binning from mass variable
    # do fitting
    rt.EnableImplicitMT()
    # final_model.fitTo(roo_hist, rt.RooFit.Save(), EvalBackend ="cpu")
    final_model.fitTo(roo_hist,rt.RooFit.PrefitDataFraction(0.8), Save=True,  EvalBackend ="cpu") #, Minos=True
    print(f"fitting elapsed time: {time.time() - time_step}")
    #do plotting
    roo_dataset.plotOn(frame)
    final_model.plotOn(frame, Name="final_model", LineColor=rt.kGreen)
    final_model.plotOn(frame, Components="signal", LineColor=rt.kBlue)
    final_model.plotOn(frame, Components="bkg", LineColor=rt.kRed)
    
    frame.Draw()
    canvas.Update()
    # canvas.Draw()
    canvas.SaveAs(f"calibration_fitCat{cat_idx}.pdf")
    del canvas
    # consider script to wait a second for stability?
    time.sleep(1)

if __name__ == "__main__":
    client =  Client(n_workers=5,  threads_per_worker=1, processes=True, memory_limit='10 GiB') 
    common_load_path = "/work/users/yun79/stage1_output/Run2StorageTest/2018/f1_0"
    # data_load_path = common_load_path+"/data*/*/*.parquet"
    data_load_path = common_load_path+"/data_C/*/*.parquet"
    # data_load_path = common_load_path+"/data_D/*/*.parquet"
    
    data_events = dak.from_parquet(data_load_path) 
    # we're only interested in ZCR
    region_filter = ak.fill_none(data_events["z_peak"], value=False)
    data_events = data_events[region_filter]
    # only select specific fields to load to save run time
    fields_of_interest = ["mu1_pt", "mu1_eta", "mu2_eta","dimuon_mass"] # mu1,mu2 are needed to separate categories
    data_events = data_events[fields_of_interest]
    # load data to memory using compute()
    data_events = ak.zip({
        field : data_events[field] for field in data_events.fields
    }).compute()
    data_categories = get_calib_categories(data_events)
    # iterate over 30 different calibration categories
    # for idx in range(len(data_categories)):
    for idx in range(12, len(data_categories)):
        cat_selection = data_categories[idx]
        cat_dimuon_mass = ak.to_numpy(data_events.dimuon_mass[cat_selection])
        if idx < 12:
            generateBWxDCB_plot(cat_dimuon_mass, idx)
        else:
            generateVoigtian_plot(cat_dimuon_mass, idx)