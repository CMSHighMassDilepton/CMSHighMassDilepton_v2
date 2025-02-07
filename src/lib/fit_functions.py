from typing import Tuple, List, Dict
import ROOT as rt
import numpy as np
import awkward as ak
import ctypes

def getFEWZ_vals(FEWZ_histo):
    """
    Function from https://github.com/green-cabbage/copperhead_fork2/blob/Run3/stage3/fitter.py#L686-L705
    It's not the most elegant, but it works, so I am leaving it as it is
    """
    n_points = FEWZ_histo.GetNbinsX()
    x_vals = []
    y_vals = []
    for i in range(n_points):
        if i<0 or i >=42:
            continue
        if (FEWZ_histo.GetBinCenter(i)) < 110.0:
            x_vals.append(110.0)
            y_vals.append(FEWZ_histo.GetBinContent(i+1)*1.05) 
            continue
        if (FEWZ_histo.GetBinCenter(i)) >150:
            x_vals.append(150.0)
            y_vals.append(FEWZ_histo.GetBinContent(i)*0.95)  
            continue
        x_vals.append(FEWZ_histo.GetBinCenter(i))
        y_vals.append(FEWZ_histo.GetBinContent(i))
    return (np.array(x_vals), np.array(y_vals))
    


def MakeFEWZxBernDof3(
        name:str, 
        title:str, 
        mass: rt.RooRealVar, 
        c1: rt.RooRealVar, c2: rt.RooRealVar, c3: rt.RooRealVar
    ) ->Tuple[rt.RooProdPdf, Dict]:
    """
    params:
    mass = rt.RooRealVar that we will fitTo
    dof = degrees of freedom given to this model. Since the spline
    has no dof, all the dof is inserted to the Bernstein
    """
    # collect all variables that we don't want destroyed by Python once function ends
    out_dict = {}

    # make BernStein of order == dof
    n_coeffs = 3
    BernCoeff_list = [c1, c2, c3]
    name = f"BernsteinFast"
    bern_model = rt.RooBernsteinFast(n_coeffs)(name, name, mass, BernCoeff_list)
    out_dict[name] = bern_model # add model to make python remember


    
    # make the spline portion

    # this ROOT files has branches full_36fb, full_xsec, full_shape -> all three has the same shape (same hist once normalized)
    FEWZ_file = rt.TFile("./data/NNLO_Bourilkov_2017.root", "READ")
    # FEWZ_histo = FEWZ_file.Get("full_36fb")
    FEWZ_histo = FEWZ_file.Get("full_shape")
    
    x_arr, y_arr = getFEWZ_vals(FEWZ_histo)
    

    # Roospline start ---------------------------------    
    x_arr_vec = rt.vector("double")(x_arr)
    y_arr_vec = rt.vector("double")(y_arr)
    name = "fewz_roospline_func"
    roo_spline_func = rt.RooSpline(name, name, mass, x_arr_vec, y_arr_vec)
    out_dict[name] = roo_spline_func
    # Roospline end ------------------------------------

    # Roospline1D start ---------------------------------    
    # x0 = (ctypes.c_double * len(x_arr))(*x_arr)
    # y0 = (ctypes.c_double * len(y_arr))(*y_arr)
    # n = len(x0)
    # name = "fewz_roospline_func"
    # roo_spline_func = rt.RooSpline1D(name, name, mass, n,x0, y0)
    # out_dict[name] = roo_spline_func
    # Roospline1D end ------------------------------------
    

    # turn roo_spline_func into pdf
    # RooProdPDF seems to automatically normalize the pdfs on https://root.cern.ch/doc/master/classRooProdPdf.html,
    # so no need to "normalize" rooSpline into a PDF. Also, I tried both full_36fb and full_shape bracnhes of 
    # FEWZ histograms (they have same shape, but different values), and I saw no difference in fit function plot,
    # loosely suggesting automatic normalization
    roo_spline_pdf = roo_spline_func
    
    
    # name = "fewz_roospline_pdf"
    # roo_spline_pdf = rt.RooWrapperPdf(name, name, roo_spline_func)
    # out_dict[name] = roo_spline_pdf # add model to make python remember  

    # RooWrapperPdf doesn't seem to work well with FitTo. Just freezes for a long time
    # name = "fewz_roospline_pdf"
    # roo_spline_pdf = rt.RooGenericPdf(name, "@0", rt.RooArgList(roo_spline_func))      
    # out_dict[name] = roo_spline_pdf # add model to make python remember  

    final_model = rt.RooProdPdf(name, title, [bern_model, roo_spline_pdf]) 
   
    return (final_model, out_dict)

def MakeFEWZxBern(mass: rt.RooRealVar, dof: int) ->Tuple[rt.RooProdPdf, Dict]:
    """
    params:
    mass = rt.RooRealVar that we will fitTo
    dof = degrees of freedom given to this model. Since the spline
    has no dof, all the dof is inserted to the Bernstein
    """
    # collect all variables that we don't want destroyed by Python once function ends
    out_dict = {}

    c_start_val_map = {
        # 1 : 0.17,
        # 2 : 0.15,
        # 3 : 0.05,
        1 : 0.351,
        2 : 0.178,
        3 : 0.110,
    }
    # make BernStein of order == dof
    n_coeffs = dof 
    BernCoeff_list = []
    for ix in range(n_coeffs):
        name = f"FEWZxBern_Bernstein_c_{ix+1}"
        c_start_val = c_start_val_map[ix+1]
        coeff = rt.RooRealVar(name,name, c_start_val,-2.0, 2.0)
        out_dict[name] = coeff # add variable to make python remember 
        BernCoeff_list.append(coeff)
    name = f"FEWZxBern_BernsteinFast_n_coeffs_{n_coeffs}"
    bern_model = rt.RooBernsteinFast(n_coeffs)(name, name, mass, BernCoeff_list)
    out_dict[name] = bern_model # add model to make python remember


    
    # make the spline portion

    # this ROOT files has branches full_36fb, full_xsec, full_shape -> all three has the same shape (same hist once normalized)
    FEWZ_file = rt.TFile("./data/NNLO_Bourilkov_2017.root", "READ")
    # FEWZ_histo = FEWZ_file.Get("full_36fb")
    FEWZ_histo = FEWZ_file.Get("full_shape")
    
    x_arr, y_arr = getFEWZ_vals(FEWZ_histo)
    

    # Roospline start ---------------------------------    
    x_arr_vec = rt.vector("double")(x_arr)
    y_arr_vec = rt.vector("double")(y_arr)
    name = "fewz_roospline_func"
    roo_spline_func = rt.RooSpline(name, name, mass, x_arr_vec, y_arr_vec)
    out_dict[name] = roo_spline_func
    # Roospline end ------------------------------------

    # Roospline1D start ---------------------------------    
    # x0 = (ctypes.c_double * len(x_arr))(*x_arr)
    # y0 = (ctypes.c_double * len(y_arr))(*y_arr)
    # n = len(x0)
    # name = "fewz_roospline_func"
    # roo_spline_func = rt.RooSpline1D(name, name, mass, n,x0, y0)
    # out_dict[name] = roo_spline_func
    # Roospline1D end ------------------------------------
    

    # turn roo_spline_func into pdf
    # RooProdPDF seems to automatically normalize the pdfs on https://root.cern.ch/doc/master/classRooProdPdf.html,
    # so no need to "normalize" rooSpline into a PDF. Also, I tried both full_36fb and full_shape bracnhes of 
    # FEWZ histograms (they have same shape, but different values), and I saw no difference in fit function plot,
    # loosely suggesting automatic normalization
    roo_spline_pdf = roo_spline_func
    
    
    # name = "fewz_roospline_pdf"
    # roo_spline_pdf = rt.RooWrapperPdf(name, name, roo_spline_func)
    # out_dict[name] = roo_spline_pdf # add model to make python remember  

    # RooWrapperPdf doesn't seem to work well with FitTo. Just freezes for a long time
    # name = "fewz_roospline_pdf"
    # roo_spline_pdf = rt.RooGenericPdf(name, "@0", rt.RooArgList(roo_spline_func))      
    # out_dict[name] = roo_spline_pdf # add model to make python remember  

    name = f"FEWZxBern_dof_{dof}"
    final_model = rt.RooProdPdf(name, name, [bern_model, roo_spline_pdf]) 
   
    return (final_model, out_dict)


# def MakeFEWZxBernOld(mass: rt.RooRealVar, dof: int, mass_hist: rt.RooDataHist) ->Tuple[rt.RooProdPdf, Dict]:
#     """
#     params:
#     mass = rt.RooRealVar that we will fitTo
#     dof = degrees of freedom given to this model. Since the spline
#     has no dof, all the dof is inserted to the Bernstein
#     """
#     # collect all variables that we don't want destroyed by Python once function ends
#     out_dict = {}

#     c_start_val_map = {
#         0 : 0.25,
#         1 : 0.25,
#         2 : 0.25,
#         3 : 0.25,
#     }
#     # make BernStein of order == dof
#     n_coeffs = dof + 1 # you need to give in n+1 coefficients for degree of freedom n
#     BernCoeff_list = []
#     for ix in range(n_coeffs):
#         name = f"FEWZxBern_Bernstein_c_{ix}"
#         c_start_val = c_start_val_map[ix]
#         coeff = rt.RooRealVar(name,name, c_start_val,-2.0, 2.0)
#         out_dict[name] = coeff # add variable to make python remember 
#         BernCoeff_list.append(coeff)
#     name = f"FEWZxBern_Bernstein_model_n_coeffs_{n_coeffs}"
#     bern_model = rt.RooBernstein(name, name, mass, BernCoeff_list)
#     out_dict[name] = bern_model # add model to make python remember

#     # make the spline portion
#     x_arr = []
#     y_arr = []
#     nbins = mass.frame().GetXaxis().GetNbins()
#     for ix in range(nbins):
#         binCentre = mass_hist.get(ix)[mass.GetName()].getVal()
#         x_arr.append(binCentre)
#         binVal = mass_hist.weight(ix)
#         y_arr.append(binVal)

    
#     # x_arr_vec = rt.vector("double")(x_arr)
#     # y_arr_vec = rt.vector("double")(y_arr)
#     # name = "fewz_roospline_func"
#     # roo_spline_func = rt.RooSpline(name, name, mass, x_arr_vec, y_arr_vec)
        
#     x0 = [109.75,   110.25, 110.75, 111.25, 111.75, 112.25, 112.75, 113.25, 113.75, 114.25, 114.75, 115.25, 115.75, 116.25, 116.75, 117.25, 117.75, 118.25, 118.75, 119.25, 119.75, 120.25, 120.75, 121.25, 121.75, 122.25, 122.75, 123.25, 123.75, 124.25, 124.75, 125.25, 125.75, 126.25, 126.75, 127.25, 127.75, 128.25, 128.75, 129.25, 129.75, 130.25, 130.75, 131.25, 131.75, 132.25, 132.75, 133.25, 133.75, 134.25, 134.75, 135.25, 135.75, 136.25, 136.75, 137.25, 137.75, 138.25, 138.75, 139.25, 139.75, 140.25, 140.75, 141.25, 141.75, 142.25, 142.75, 143.25, 143.75, 144.25, 144.75, 145.25, 145.75, 146.25, 146.75, 147.25, 147.75, 148.25, 148.75, 149.25, 149.75, 150.25]
    
#     y0 = [1406.0, 1406.0, 1422.0, 1355.0, 1327.0, 1258.0, 1206.0, 1151.0, 1149.0, 1105.0, 1034.0, 1020.0, 1039.0, 965.0, 992.0, 955.0, 871.0, 914.0, 844.0, 847.0, 847.0, 838.0, 776.0, 780.0, 772.0, 776.0, 748.0, 777.0, 668.0, 678.0, 662.0, 665.0, 664.0, 642.0, 607.0, 604.0, 647.0, 585.0, 566.0, 578.0, 583.0, 574.0, 515.0, 528.0, 548.0, 513.0, 522.0, 494.0, 510.0, 482.0, 470.0, 476.0, 518.0, 426.0, 462.0, 477.0, 473.0, 462.0, 436.0, 409.0, 413.0, 436.0, 458.0, 402.0, 436.0, 424.0, 446.0, 444.0, 378.0, 366.0, 424.0, 338.0, 388.0, 321.0, 352.0, 349.0, 342.0, 352.0, 338.0, 336.0, 316.0, 316.0]

#     x_arr_ctype = (ctypes.c_double * len(x_arr))(*x_arr)
#     y_arr_ctype = (ctypes.c_double * len(y_arr))(*y_arr)
#     n = len(x_arr)
#     name = "fewz_roospline_func"
#     roo_spline_func = rt.RooSpline1D(name, name, mass, n, x_arr_ctype, y_arr_ctype, "CSPLINE")

#     out_dict[name] = roo_spline_func
#     name = "fewz_roospline_pdf"
#     roo_spline_pdf = rt.RooWrapperPdf(name, name, roo_spline_func)
#     out_dict[name] = roo_spline_pdf # add model to make python remember  
                          

#     name = f"FEWZxBern_dof_{dof}"
#     final_model = rt.RooProdPdf(name, name, [bern_model, roo_spline_pdf]) 
   
#     return (final_model, out_dict)

def MakeBWZ_Redux(mass: rt.RooRealVar, dof: int) ->Tuple[rt.RooProdPdf, Dict]:
    """
    params:
    mass = rt.RooRealVar that we will fitTo
    dof = degrees of freedom given to this model. This parameter is meaningless 
        as the actual dof(=3) is hard coded into the model's definition
    """
    # collect all variables that we don't want destroyed by Python once function ends
    out_dict = {}
    
    name = f"BWZ_Redux_a_coeff"
    # a_coeff = rt.RooRealVar(name,name, -0.001,-0.0025,0.0035)
    # a_coeff = rt.RooRealVar(name,name, -0.001,-0.015,0.015)
    a_coeff = rt.RooRealVar(name,name, -0.0146,-0.02,0.03)
    name = f"BWZ_Redux_b_coeff"
    b_coeff = rt.RooRealVar(name,name, -0.000111,-0.001,0.001)
    # b_coeff = rt.RooRealVar(name,name, -0.00022,-0.001,0.001)
    name = f"BWZ_Redux_c_coeff"
    # c_coeff = rt.RooRealVar(name,name, 1.5,-5.0,5.0)
    c_coeff = rt.RooRealVar(name,name, 0.462,-5.0,5.0)
    

    # add in the variables and models
    out_dict[a_coeff.GetName()] = a_coeff 
    out_dict[b_coeff.GetName()] = b_coeff
    out_dict[c_coeff.GetName()] = c_coeff 
    

    name = "BWZ_Redux_dof_3"
    final_model = rt.RooModZPdf(name, name, mass, a_coeff, b_coeff, c_coeff) 
    return (final_model, out_dict)



# def MakeBWZ_Redux_old(mass: rt.RooRealVar, dof: int) ->Tuple[rt.RooProdPdf, Dict]:
#     """
#     params:
#     mass = rt.RooRealVar that we will fitTo
#     dof = degrees of freedom given to this model. This parameter is meaningless 
#         as the actual dof(=3) is hard coded into the model's definition
#     """
#     # collect all variables that we don't want destroyed by Python once function ends
#     out_dict = {}
    
#     name = f"BWZ_Redux_a_coeff"
#     a_coeff = rt.RooRealVar(name,name, -0.001,-0.001,0.001)
#     name = "exp_model_mass"
#     exp_model_mass = rt.RooExponential(name, name, mass, a_coeff)
    
#     mass_sq = rt.RooFormulaVar("mass_sq", "@0*@0", rt.RooArgList(mass))
#     name = f"BWZ_Redux_b_coeff"
#     b_coeff = rt.RooRealVar(name,name, -0.00001,-0.001,0.001)
    
#     name = "exp_model_mass_sq"
#     exp_model_mass_sq = rt.RooExponential(name, name, mass_sq, b_coeff)

#     # add in the variables and models
#     out_dict[a_coeff.GetName()] = a_coeff 
#     out_dict[exp_model_mass.GetName()] = exp_model_mass
#     out_dict[mass_sq.GetName()] = mass_sq
#     out_dict[b_coeff.GetName()] = b_coeff
#     out_dict[exp_model_mass_sq.GetName()] = exp_model_mass_sq
    
#     # make Z boson related stuff
#     bwWidth = rt.RooRealVar("bwWidth", "bwWidth", 2.5, 0, 30)
#     bwmZ = rt.RooRealVar("bwmZ", "bwmZ", 91.2, 90, 92)
#     bwWidth.setConstant(True)
#     bwmZ.setConstant(True)

#     # start multiplying them all
#     name = f"BWZ_Redux_c_coeff"
#     c_coeff = rt.RooRealVar(name,name, 1.5,-5.0,5.0)
#     BWZ_redux_main = rt.RooGenericPdf(
#         "BWZ_redux_main", "@1/ ( pow((@0-@2), @3) + 0.25*pow(@1, @3) )", rt.RooArgList(mass, bwWidth, bwmZ, c_coeff)
#     )
#     # add in the variables and models
#     out_dict[bwWidth.GetName()] = bwWidth 
#     out_dict[bwmZ.GetName()] = bwmZ 
#     out_dict[c_coeff.GetName()] = c_coeff 
#     out_dict[BWZ_redux_main.GetName()] = BWZ_redux_main 

#     name = "BWZ_Redux_dof_3"
#     final_model = rt.RooProdPdf(name, name, [BWZ_redux_main, exp_model_mass, exp_model_mass_sq]) 
#     return (final_model, out_dict)

def MakeBWZxBernFast(mass: rt.RooRealVar, dof: int) ->Tuple[rt.RooProdPdf, Dict]:
    """
    params:
    mass = rt.RooRealVar that we will fitTo
    dof = degrees of freedom given to this model. We assume it to be >= 2
    """
    # collect all variables that we don't want destroyed by Python once function ends
    out_dict = {}



    c_start_val_map = {
        1 : 1.157,
        2 : 1.602,
    }
    # make BernStein
    bern_n_coeffs = dof-1
    # print(f"MakeBWZxBernFast bernFast_n_coeffs: {bern_n_coeffs}")
    BernCoeff_list = rt.RooArgList()

    for ix in range(bern_n_coeffs):
        name = f"BWZxBernFast_Bernstein_c_{ix+1}"
        c_start_val = c_start_val_map[ix+1]
        coeff = rt.RooRealVar(name,name, c_start_val,0, 2.0)
        out_dict[name] = coeff # add variable to make python remember 
        BernCoeff_list.add(coeff)
    
    #
    name = f"BWZxBernFast_Bernstein_model_n_coeffs_{bern_n_coeffs}"
    bern_model = rt.RooBernsteinFast(bern_n_coeffs)(name, name, mass, BernCoeff_list)
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
    expCoeff = rt.RooRealVar(name, name, -0.015, -1.0, 0.5)
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
    name = f"BWZxBern_dof_{dof}"
    # final_model = rt.RooProdPdf(name, name, [bern_model, full_BWZ]) 
    final_model = rt.RooProdPdf(name, name, [bern_model, BWZ, exp_model]) 
   
    return (final_model, out_dict)

def MakeBWZxBern(mass: rt.RooRealVar, dof: int) ->Tuple[rt.RooProdPdf, Dict]:
    """
    params:
    mass = rt.RooRealVar that we will fitTo
    dof = degrees of freedom given to this model. We assume it to be >= 2
    """
    # collect all variables that we don't want destroyed by Python once function ends
    out_dict = {}


    # c_start_val_map = {
    #     0 : 0.17,
    #     1 : 0.34, # 0.0025
    #     2 : 1.05,
    # }
    c_start_val_map = {
        0 : 1,
        1 : 1.671,
        2 : 2,
    }
    # make BernStein
    bern_dof = dof-1 # one dof is used for the RooExponenet
    bern_n_coeffs = bern_dof +1 # you need to give in n+1 coefficients for degree of freedom n
    print(f"bern_n_coeffs: {bern_n_coeffs}")
    # bern_n_coeffs = 2
    BernCoeff_list = []
    for ix in range(bern_n_coeffs):
        name = f"BWZxBern_Bernstein_c_{ix}"
        c_start_val = c_start_val_map[ix]
        coeff = rt.RooRealVar(name,name, c_start_val,-2.0, 2.0)
        if ix == 0 : # freeze the first coeff
            coeff.setConstant(True)
        out_dict[name] = coeff # add variable to make python remember 
        BernCoeff_list.append(coeff)
    name = f"BWZxBern_Bernstein_model_n_coeffs_{bern_n_coeffs}"
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
    expCoeff = rt.RooRealVar(name, name, -0.015, -1.0, 0.5)
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
    name = f"BWZxBern_dof_{dof}"
    # final_model = rt.RooProdPdf(name, name, [bern_model, full_BWZ]) 
    final_model = rt.RooProdPdf(name, name, [bern_model, BWZ, exp_model]) 
   
    return (final_model, out_dict)
    

def MakeSumExponential(mass: rt.RooRealVar, dof: int, fit_range="loSB,hiSB") ->Tuple[rt.RooAddPdf, Dict]:
    """
    params:
    mass = rt.RooRealVar that we will fitTo
    dof = degrees of freedom of the sum of exponential, that we assume to be >= 3. Must be an odd number
    fit_range = str representation of fit range from mass. We assume this has already been defined before this
        function is called. If no fit_range is specified, you can give an empty string
    returns:
    rt.RooAddPdf
    dictionary of variables with {variable name : rt.RooRealVar or rt.RooExponential} format mainly for keep python from
    destroying these variables, but also useful in debugging
    """
    order = int((dof+1)/2) # order is the number of expoenetial terms to sum up
    print(f"order: {order}")
    b_start_val_map = {
        # 0 : -0.2,
        # 1 : -0.02,
        0 : -0.0603,
        1 : -0.0450,
    }
    a_start_val_map = {
        # 1 : 0.1,
        1 : 0.742,
    }

    
    # TODO: make a dictionary list of starting values optimized for sumexponential for specific data, starting with data_* for ggH
    model_list = [] # list of RooExp models for RooAddPdf
    a_i_list = [] # list of RooExp coeffs for RooAddPdf
    b_i_list = [] # list of RooExp b_i variables to save it from being destroyed

    
    for ix in range(order):
        #hard code in starting values
        name = f"S_exp_b_{ix}"
        b_start_val = b_start_val_map[ix]
        b_i = rt.RooRealVar(name, name, b_start_val, -2.0, 1.0)
        b_i_list.append(b_i)
        
        name = f"S_exp_model_{ix}"
        model = rt.RooExponential(name, name, mass, b_i)
        model_list.append(model)
        
        if ix >0:
            name = f"S_exp_a_{ix}"
            a_start_val = a_start_val_map[ix]
            a_i = rt.RooRealVar(name, name, a_start_val, 0, 1.0)
            a_i_list.append(a_i)

    
            
    name = f"S_exp_dof_{dof}"
    recursiveFractions= True
    final_model = rt.RooAddPdf(name, name, model_list, a_i_list, recursiveFractions)
    # for good explnanation of recursiveFractions, visit https://root-forum.cern.ch/t/rooaddpdf-when-to-use-recursivefraction/33317
    # final_model = rt.RooAddPdf(name, name, model_list, a_i_list)
    if fit_range != "": # if empty string, skip
        final_model.fixCoefNormalization(rt.RooArgSet(mass))
        # final_model.fixCoefRange(fit_range)
        
    # collect all variables that we don't want destroyed by Python once function ends
    out_dict = {}
    for model in model_list:
        out_dict[model.GetName()] = model
    for a_i in a_i_list:
        out_dict[a_i.GetName()] = a_i
    for var in b_i_list:
        out_dict[var.GetName()] = var
    return (final_model, out_dict)


def MakePowerSum(mass: rt.RooRealVar, dof: int, fit_range="loSB,hiSB") ->Tuple[rt.RooAddPdf, Dict]:
    """
    params:
    mass = rt.RooRealVar that we will fitTo
    dof = degrees of freedom of the sum of exponential, that we assume to be >= 3. Must be an odd number
    fit_range = str representation of fit range from mass. We assume this has already been defined before this
        function is called. If no fit_range is specified, you can give an empty string
    returns:
    rt.RooAddPdf
    dictionary of variables with {variable name : rt.RooRealVar or rt.RooPowerSum} format mainly for keep python from
    destroying these variables, but also useful in debugging
    """
    order = int((dof+1)/2) # order is the number of power terms to sum up
    print(f"MakePowerSum order: {order}")
    b_start_val_map = {
        0 : -10,
        1 : -15,
    }
    a_start_val_map = {
        0 : 0.001,
        1 : 0.9,
    }
    out_dict = {}
    mass = rt.RooFormulaVar("mass_shift", "@0-100", rt.RooArgList(mass))
    out_dict[mass.GetName()] = mass

    
    # TODO: make a dictionary list of starting values optimized for sumexponential for specific data, starting with data_* for ggH
    a_i_list = [] # list of RooPowerSum coeffs 
    b_i_list = [] # list of RooPower exponents

    
    for ix in range(order):
        #hard code in starting values
        name = f"PowerSum_b_{ix}"
        b_start_val = b_start_val_map[ix]
        b_i = rt.RooRealVar(name, name, b_start_val, -2.0, 1.0)
        b_i_list.append(b_i)

        name = f"PowerSum_a_{ix}"
        a_start_val = a_start_val_map[ix]
        a_i = rt.RooRealVar(name, name, a_start_val, 0, 1.0)
        a_i_list.append(a_i)
        
    name = f"PowerSum_dof_{dof}"
    final_model = rt.RooPowerSum(name, name, mass, a_i_list, b_i_list)

    
    # collect all variables that we don't want destroyed by Python once function ends
    
    for a_i in a_i_list:
        out_dict[a_i.GetName()] = a_i
    for var in b_i_list:
        out_dict[var.GetName()] = var
    return (final_model, out_dict)