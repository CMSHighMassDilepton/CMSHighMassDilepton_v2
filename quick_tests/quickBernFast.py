import ROOT as rt
import numpy as np
import awkward as ak

if __name__ == "__main__":
    load_path = "/work/users/yun79/stage2_output/test/processed_events.parquet"
    processed_events = ak.from_parquet(load_path)
    print("events loaded!")

    subCat_filter = (processed_events["subCategory_idx"] == 0)
    mass_arr = ak.to_numpy(processed_events.dimuon_mass[subCat_filter])
    
    # name ="coeff"
    # BernCoeff_list = rt.RooArgList(
    #     rt.RooRealVar(name,name, 0.25,0, 2.0),
    #     rt.RooRealVar(name,name, 0.25,0, 2.0)
    # )
    
    # mass_name = "dimuon_mass"
    # mass =  rt.RooRealVar(mass_name,mass_name,0.5,0.0,1.0)
    # # mass =  rt.RooRealVar(mass_name,mass_name,125,110,150)
    # mass.setBins(80)
    # # mass.setRange("full", 110, 150 )
    # # generate exp decay data
    # mass_arr = np.random.uniform(size=100)
    # mass_arr = np.exp(-100 * mass_arr)
    # roo_dataset = rt.RooDataSet.from_numpy({mass_name: mass_arr}, [mass])
    # roo_hist = rt.RooDataHist("data_hist","data_hist", rt.RooArgSet(mass), roo_dataset)

    
    # name = "bern model"
    # # bern_model = rt.RooBernstein(name, name, mass, BernCoeff_list)
    # order = 3
    # bern_model =  rt.RooBernsteinFast(order)("pdf_bern_cat","bernstein",mass, BernCoeff_list)
    # rt.EnableImplicitMT()
    # bern_model.fitTo(roo_hist, EvalBackend="cpu", Save=True)


    
    
    name ="coeff"
    
    b1 = rt.RooRealVar(name + '_1',name, 0.25,0, 2.0)
    b2 = rt.RooRealVar(name + '_2',name, 0.25,0, 2.0)
    b3 = rt.RooRealVar(name + '_3',name, 0.25,0, 2.0)
    # b1 = rt.RooRealVar(name + '_dsfds',name, 0.25,0, 2.0)
    # b2 = rt.RooRealVar(name + '_dsdd',name, 0.25,0, 2.0)
    # b3 = rt.RooRealVar(name + '_ewrewrwerw',name, 0.25,0, 2.0)
    # BernCoeff_list = rt.RooArgList()
    # BernCoeff_list.add(b1)
    # BernCoeff_list.add(b2)
    # BernCoeff_list.add(b3)
    BernCoeff_list = rt.RooArgList(b1,b2,b3)
    
    mass_name = "dimuon_mass"
    # mass =  rt.RooRealVar(mass_name,mass_name,0.5,0.0,1.0)
    mass =  rt.RooRealVar(mass_name,mass_name,125,110,150)
    mass.setBins(80)
    mass.setRange("hiSB", 135, 150 )
    mass.setRange("loSB", 110, 115 )
    fit_range = "hiSB,loSB"
    
    # generate exp decay data
    # mass_arr = np.random.uniform(size=100)
    # mass_arr = np.exp(-100 * mass_arr)
    roo_dataset = rt.RooDataSet.from_numpy({mass_name: mass_arr}, [mass])
    roo_hist = rt.RooDataHist("data_hist","data_hist", rt.RooArgSet(mass), roo_dataset)
    
    
    name = "bern model"
    
    order = 3
    bern_model =  rt.RooBernsteinFast(order)("pdf_bern_cat","bernstein",mass, BernCoeff_list)
    rt.EnableImplicitMT()
    bern_model.fitTo(roo_hist, rt.RooFit.Range(fit_range), EvalBackend="cpu", Save=True)