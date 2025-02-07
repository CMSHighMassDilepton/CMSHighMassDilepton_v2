import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client
from omegaconf import OmegaConf

from typing import Tuple, List, Dict
import ROOT as rt
import glob, os

from quickSMFtest_functions import MakeBWZ_Redux, MakeBWZxBern, MakeSumExponential,prepare_features,evaluate_bdt



if __name__ == "__main__":
    client =  Client(n_workers=31,  threads_per_worker=1, processes=True, memory_limit='4 GiB') 
    load_path = "/depot/cms/users/yun79/results/stage1/test_VBF-filter_JECon_07June2024/2018/f1_0"
    # full_load_path = load_path+f"/data_C/*/*.parquet"
    # full_load_path = load_path+f"/data_D/*/*.parquet"
    # full_load_path = load_path+f"/data_*/*/*.parquet"
    # full_load_path = load_path+f"/data_A/*/*.parquet"
    full_load_path = load_path+f"/ggh_powheg/*/*.parquet"
    # full_load_path = load_path+f"/dy_M-100To200/*/*.parquet"
    
    events = dak.from_parquet(full_load_path)

    # load and obtain MVA outputs
    events["dimuon_dEta"] = np.abs(events.mu1_pt - events.mu2_pt)
    events["dimuon_pt_log"] = np.log(events.dimuon_pt)
    events["jj_mass_log"] = np.log(events.jj_mass)
    events["ll_zstar_log"] = np.log(events.ll_zstar)
    events["mu1_pt_over_mass"] = events.mu1_pt / events.dimuon_mass
    events["mu2_pt_over_mass"] = events.mu2_pt / events.dimuon_mass

    train_feat_dict = {
        "BDTperyear_2018" : [
            'dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_eta', 'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 
            'dimuon_pt_log', 'jet1_eta', 'jet1_phi', 'jet1_pt', 'jet1_qgl', 'jet2_eta', 'jet2_phi', 
            'jet2_pt', 'jet2_qgl', 'jj_dEta', 'jj_dPhi', 'jj_eta', 'jj_mass', 'jj_mass_log', 
            'jj_phi', 'jj_pt', 'll_zstar_log', 'mmj1_dEta', 'mmj1_dPhi', 
            'mmj_min_dEta', 'mmj_min_dPhi', 'mmjj_eta', 'mmjj_mass', 'mmjj_phi', 'mmjj_pt', 'mu1_eta', 'mu1_iso', 
            'mu1_phi', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt_over_mass', 'zeppenfeld'
        ],
        "phifixedBDT_2018" : [
                'dimuon_cos_theta_cs', 'dimuon_eta', 'dimuon_phi_cs', 'dimuon_pt', 'jet1_eta', 'jet1_pt', 'jet2_eta', 'jet2_pt', 'jj_dEta', 'jj_dPhi', 'jj_mass', 'mmj1_dEta', 'mmj1_dPhi',  'mmj_min_dEta', 'mmj_min_dPhi', 'mu1_eta', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_pt_over_mass', 'zeppenfeld' #, 'njets'
        ], # AN 19-124
        
    }
    
    # model_name = "BDTv12_2018"
    model_name = "phifixedBDT_2018"
    # model_name = "BDTperyear_2018"

    training_features = train_feat_dict[model_name]
    print(f"len(training_features): {len(training_features)}")

    # load training features from the ak.Record
    for training_feature in training_features:
        if training_feature not in events.fields:
            print(f"mssing feature: {training_feature}")
    
    fields2load = training_features + ["h_peak", "h_sidebands", "dimuon_mass", "wgt_nominal_total", "mmj2_dEta", "mmj2_dPhi"]
    events = events[fields2load]
    # load data to memory using compute()
    events = ak.zip({
        field : events[field] for field in events.fields
    }).compute()


    parameters = {
    "models_path" : "/depot/cms/hmm/vscheure/data/trained_models/"
    }
    
    
    processed_events = evaluate_bdt(events, "nominal", model_name, training_features, parameters) # this also only filters in h_peak and h_sidebands

    # load BDT score edges for subcategory divison
    BDTedges_load_path = "../configs/MVA/ggH/BDT_edges.yaml"
    edges = OmegaConf.load(BDTedges_load_path)
    year = "2018"
    edges = np.array(edges[year])
    print(f"subCat BDT edges: {edges}")

    # Calculate the subCategory index 
    BDT_score = processed_events["BDT_score"]
    n_edges = len(edges)
    BDT_score_repeat = ak.concatenate([BDT_score[:,np.newaxis] for i in range(n_edges)], axis=1)
    # BDT_score_repeat
    n_rows = len(BDT_score_repeat)
    edges_repeat = np.repeat(edges[np.newaxis,:],n_rows,axis=0)
    # edges_repeat.shape
    edge_idx = ak.sum( (BDT_score_repeat >= edges_repeat), axis=1)
    subCat_idx =  edge_idx - 1 # sub category index starts at zero
    processed_events["subCategory_idx"] = subCat_idx

    # filter in only the variables you need to do stage3
    fields2save = [
        "dimuon_mass",
        "BDT_score",
        "subCategory_idx",
        "wgt_nominal_total",
    ]
    processed_events = ak.zip({
        field : processed_events[field] for field in fields2save
    })
    # define save path and save
    save_path = "/work/users/yun79/stage2_output/test"
        # make save path if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if "data" in full_load_path:
        save_filename = save_path+"/processed_events_data.parquet"  
    elif "ggh_powheg" in full_load_path: # else, ggh powheg
        save_filename = save_path+"/processed_events_signalMC.parquet" 
    elif "dy_M-100To200" in full_load_path:
        save_filename = save_path+"/processed_events_dyMC.parquet" 
    print(f"save_filename: {save_filename}")

    # delete the file if there's already same save_filename
    try:
        os.remove(save_filename)
    except:
        pass
    ak.to_parquet(processed_events, save_filename)


    