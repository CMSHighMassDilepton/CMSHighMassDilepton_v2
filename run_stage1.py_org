from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from src.copperhead_processor import EventProcessor
# NanoAODSchema.warn_missing_crossrefs = False
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import dask
# from dask.distributed import Client
import sys
import time
import json
from distributed import LocalCluster, Client, progress
import pandas as pd
import os
import tqdm
import warnings
import dask_awkward as dak
import glob
from itertools import islice
import copy
import argparse
from dask.distributed import performance_report
from src.corrections.evaluator import nnlops_weights, qgl_weights
import os
from omegaconf import OmegaConf
from coffea.nanoevents.methods import vector


# dask.config.set({'logging.distributed': 'error'})
import logging
logger = logging.getLogger("distributed.utils_perf")
logger.setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

test_mode = False
np.set_printoptions(threshold=sys.maxsize)
import gc
import ctypes
from src.lib.get_parameters import getParametersForYr

def trim_memory() -> int:
     libc = ctypes.CDLL("libc.so.6")
     return libc.malloc_trim(0)
    
# # test code limiting memory leak -----------------------------------
# import gc
# client.run(gc.collect)  # collect garbage on all workers
# import ctypes
# def trim_memory() -> int:
#      libc = ctypes.CDLL("libc.so.6")
#      return libc.malloc_trim(0)
# client.run(trim_memory)

# #-------------------------------------------------------------------

def getSavePath(start_path: str, dataset_dict: dict, file_idx: int):
    """
    Small wrapper function that returns the directory path to save the parquets
    from stage1
    """
    fraction = round(dataset_dict["metadata"]["fraction"], 3)
    fraction_str = str(fraction).replace('.', '_')
    sample_name = dataset_dict['metadata']['dataset']
    save_path = start_path + f"/f{fraction_str}/{dataset_dict['metadata']['dataset']}/{file_idx}"
    return save_path

def dataset_loop(processor, dataset_dict, file_idx=0, test=False, save_path=None):
    if save_path is None:
        save_path = "/depot/cms/users/kaur214/results/stage1/test/" # default
        # save_path = "/depot/cms/hmm/yun79/copperheadV2/results/stage1/test/"
    # print(f"dataset_dict: {dataset_dict['files']}")
    events = NanoEventsFactory.from_root(
        dataset_dict["files"],
        schemaclass=NanoAODSchema,
        metadata= dataset_dict["metadata"],
        uproot_options={
            "timeout":2400,
            # "allow_read_errors_with_report": True, # this makes process skip over OSErrors
        },
    ).events()
    
    # save input events for CI testing start ---------------------------------------------
    # dir = f'./test/stage1_inputs/{dataset_dict["metadata"]["dataset"]}'
    # if not os.path.exists(dir):
    #     os.makedirs(dir)
    # # save dataset_dict as input
    # input_dataset = OmegaConf.create(dataset_dict)
    # filename = dir + '/dataset_dict.yaml'
    # try:
    #     os.remove(filename)
    # except OSError:
    #     pass
    # with open(filename, "w") as file:
    #     OmegaConf.save(config=input_dataset, f=file.name)
    # # now save output to compare as target
    # filename = f'./test/stage1_outputs/{dataset_dict["metadata"]["dataset"]}'
    # try:
    #     os.remove(filename)
    # except OSError:
    #     pass
    # out_collections = processor.process(events)
    # zip = ak.zip(out_collections, depth_limit=1)
    # zip.to_parquet(filename)
    # raise ValueError
    # save input events for CI testing end ---------------------------------------------
    print(f"n of partitions: {events.Muon.pt}")
    out_collections = processor.process(events)
    dataset_fraction = dataset_dict["metadata"]["fraction"]

    # ------------------------------------------
    skim_dict =  {
            
            'mu1_pt': (out_collections["mu1_pt"]),
            'mu2_pt': (out_collections["mu2_pt"]),
            'mu1_eta': (out_collections["mu1_eta"]),
            'mu2_eta': (out_collections["mu2_eta"]),
            'mu1_phi': (out_collections["mu1_phi"]),
            'mu2_phi': (out_collections["mu2_phi"]),
            'mu1_iso': (out_collections["mu1_iso"]),
            'mu2_iso': (out_collections["mu2_iso"]),
            "mu1_pt_over_mass" : (out_collections["mu1_pt"] / out_collections["dimuon_mass"]) ,
            "mu2_pt_over_mass" : (out_collections["mu2_pt"] / out_collections["dimuon_mass"]) ,
            'jet1_pt': (out_collections["jet1_pt"]),
            'jet2_pt': (out_collections["jet2_pt"]),
            'jet1_eta': (out_collections["jet1_eta"]),
            'jet2_eta': (out_collections["jet2_eta"]),
            'jet1_phi': (out_collections["jet1_phi"]),
            'jet2_phi': (out_collections["jet2_phi"]),
            'jet1_rapidity': (out_collections["jet1_rapidity"]),
            'jet2_rapidity': (out_collections["jet2_rapidity"]),
            'jet1_mass': (out_collections["jet1_mass"]),
            'jet2_mass': (out_collections["jet2_mass"]),
            'jet1_qgl': (out_collections["jet1_qgl"]),
            'jet2_qgl': (out_collections["jet2_qgl"]),
            'njets': (out_collections["njets"]),
            # jj variables------------------------------
            'jj_dEta': (out_collections["jj_dEta"]),
            'jj_dPhi': (out_collections["jj_dPhi"]),
            'jj_mass': (out_collections["jj_mass"]),
            'jj_mass_log': np.log(out_collections["jj_mass"]),
            'jj_pt': (out_collections["jj_pt"]),
            'jj_eta': (out_collections["jj_eta"]),
            'jj_phi': (out_collections["jj_phi"]),
            # weights -----------------------------------------
            # 'weights': (out_collections["weights"]),    
            # #dimuon variables-----------------------
            'dimuon_mass': (out_collections["dimuon_mass"]),
            'dimuon_ebe_mass_res': (out_collections["dimuon_ebe_mass_res"]),
            'dimuon_cos_theta_cs': (out_collections["dimuon_cos_theta_cs"]),
            'dimuon_phi_cs': (out_collections["dimuon_phi_cs"]),
            'dimuon_cos_theta_eta': (out_collections["dimuon_cos_theta_eta"]),
            'dimuon_phi_eta': (out_collections["dimuon_phi_eta"]),
            'dimuon_dPhi': (out_collections["dimuon_dPhi"]),
            'dimuon_dR': (out_collections["dimuon_dR"]),
            'dimuon_dEta': (out_collections["dimuon_dEta"]),
            'dimuon_eta': (out_collections["dimuon_eta"]),
            'dimuon_rapidity': (out_collections["dimuon_rapidity"]),
            'dimuon_phi': (out_collections["dimuon_phi"]),
            'dimuon_pt': (out_collections["dimuon_pt"]),
            'dimuon_pt_log': np.log(out_collections["dimuon_pt"]),

            # # mmj variables ------------------------------
            'mmj1_dEta': (out_collections["mmj1_dEta"]),
            'mmj1_dPhi': (out_collections["mmj1_dPhi"]),
            'mmj2_dEta': (out_collections["mmj2_dEta"]),
            'mmj2_dPhi': (out_collections["mmj2_dPhi"]),
            'mmj_min_dEta': (out_collections["mmj_min_dEta"]),
            'mmj_min_dPhi': (out_collections["mmj_min_dPhi"]),
            'mmjj_mass': (out_collections["mmjj_mass"]),
            'mmjj_pt': (out_collections["mmjj_pt"]),
            'mmjj_eta': (out_collections["mmjj_eta"]),
            'mmjj_phi': (out_collections["mmjj_phi"]),
            

            # 'jet1_pt_raw': (out_collections["jet1_pt_raw"]),
            # 'jet1_mass_raw': (out_collections["jet1_mass_raw"]),
            # 'jet1_rho': (out_collections["jet1_rho"]),
            # 'jet1_area': (out_collections["jet1_area"]),
            # 'jet1_pt_jec': (out_collections["jet1_pt_jec"]),
            # 'jet1_mass_jec': (out_collections["jet1_mass_jec"]),
            # 'jet2_pt_raw': (out_collections["jet2_pt_raw"]),
            # 'jet2_mass_raw': (out_collections["jet2_mass_raw"]),
            # 'jet2_rho': (out_collections["jet2_rho"]),
            # 'jet2_area': (out_collections["jet2_area"]),
            # 'jet2_pt_jec': (out_collections["jet2_pt_jec"]),
            # 'jet2_mass_jec': (out_collections["jet2_mass_jec"]),
        
            # fraction -------------------------------------
            "fraction" : dataset_fraction*(ak.ones_like(out_collections["njets"])), 
            # Btagging WPs ------------------------------------
            "nBtagLoose" : (out_collections["nBtagLoose"]),
            "nBtagMedium" : (out_collections["nBtagMedium"]),
            # regions -------------------------------------
            "z_peak" : (out_collections["z_peak"]),
            "h_sidebands" : (out_collections["h_sidebands"]),
            "h_peak" : (out_collections["h_peak"]),
            # vbf ?? ------------------------------------------------
            "vbf_cut" : (out_collections["vbf_cut"]),
            # "pass_leading_pt" : (out_collections["pass_leading_pt"]),
            "ll_zstar_log" : np.log(np.abs(out_collections["zeppenfeld"])),
            "zeppenfeld" : (out_collections["zeppenfeld"]),
            "event" : (out_collections["event"]),
            "rpt" : (out_collections["rpt"]),
            # temporary test start ------------------------------------
            # "M105to160normalizedWeight" : (out_collections["M105to160normalizedWeight"]),
            # temporary test end ------------------------------------
    }

    
    # add in weights
    weight_dict = {}
    for key, value in out_collections.items():
        if "wgt_nominal" in key:
            # print(f"wgt name: {key}")
            weight_dict[key] = value
    skim_dict.update(weight_dict)   
    
    # add in nsoftjets and htsoft variables
    softj_vars = {}
    for key, value in out_collections.items():
        if "nsoftjets" in key:
            softj_vars[key] = value
        elif "htsoft" in key:
            softj_vars[key] = value
    skim_dict.update(softj_vars)
    # print(f"softj_vars.keys(): {softj_vars.keys()}")
    # print(f"stage1 skim compute1: {ak.zip(skim_dict).compute()}")
    # gen jet variables start ------------------------------
    is_mc = dataset_dict["metadata"]["is_mc"]
    if is_mc:
        mc_dict = {
            "gjet1_pt":  (out_collections["gjet1_pt"]),
            "gjet1_eta":  (out_collections["gjet1_eta"]),
            "gjet1_phi":  (out_collections["gjet1_phi"]),
            "gjet1_mass":  (out_collections["gjet1_mass"]),
            "gjet2_pt":  (out_collections["gjet2_pt"]),
            "gjet2_eta":  (out_collections["gjet2_eta"]),
            "gjet2_phi":  (out_collections["gjet2_phi"]),
            "gjet2_mass":  (out_collections["gjet2_mass"]),
            "gjj_pt":  (out_collections["gjj_pt"]),
            "gjj_eta":  (out_collections["gjj_eta"]),
            "gjj_phi":  (out_collections["gjj_phi"]),
            "gjj_mass":  (out_collections["gjj_mass"]),
            "gjj_dEta":  (out_collections["gjj_dEta"]),
            "gjj_dPhi":  (out_collections["gjj_dPhi"]),
            "gjj_dR":  (out_collections["gjj_dR"]),
            
        }
        skim_dict.update(mc_dict)
    # gen jet variables end ------------------------------
    # print(f"stage1 skim compute2: {ak.zip(skim_dict).compute()}")
    # print(f"skim_dict.keys(): {skim_dict.keys()}")
    # # define save path
    # fraction = round(dataset_dict["metadata"]["fraction"], 3)
    # fraction_str = str(fraction).replace('.', '_')
    # sample_name = dataset_dict['metadata']['dataset']
    # save_path = save_path + f"/f{fraction_str}/{dataset_dict['metadata']['dataset']}/{file_idx}"
    # print(f"save_path: {save_path}")
    # # remove previously existing files
    # filelist = glob.glob(f"{save_path}/*.parquet")
    # print(f"len(filelist): {len(filelist)}")
    # for file in filelist:
    #     os.remove(file)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    
    #----------------------------------
    skim_zip = ak.zip(skim_dict, depth_limit=1)
    # print(f"skim_zip: {skim_zip}")
    # skim_zip.persist().to_parquet(save_path)
    # raise ValueError
    return skim_zip
    # return "Success!"
    # 
    # zip = dask.compute(skim_dict)
    # print(f"stage1 zip compute: {zip.compute()}")
    # zip.to_parquet(save_path, compute=True)
    # print("zip to parquet done!")
    # skim = dak.to_parquet(zip, save_path, compute=False)
    # print(f"stage1 skim compute4: {skim.compute()}")
    # print(f"stage1 skim persisted: {skim.persist()}")
    # return zip



def divide_chunks(data: dict, SIZE: int):
   it = iter(data)
   for i in range(0, len(data), SIZE):
      yield {k:data[k] for k in islice(it, SIZE)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-save",
    "--save_path",
    dest="save_path",
    default=None,
    action="store",
    help="save path to store stage1 output files",
    )
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="string value of year we are calculating",
    )
    parser.add_argument(
    "--use_gateway",
    dest="use_gateway",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses dask gateway client instead of local",
    )
    parser.add_argument(
    "-aod_v",
    "--NanoAODv",
    dest="NanoAODv",
    default="9",
    action="store",
    help="version number of NanoAOD samples we're working with. currently, only 9 and 12 are supported",
    )
    args = parser.parse_args()
    # make NanoAODv into an interger variable
    args.NanoAODv = int(args.NanoAODv)
    # check for NanoAOD versions
    allowed_nanoAODvs = [9, 12]
    if not (args.NanoAODv in allowed_nanoAODvs):
        print("wrong NanoAOD version is given!")
        raise ValueError
    time_step = time.time()
    
    warnings.filterwarnings('ignore')
    """
    Coffea Dask automatically uses the Dask Client that has been defined above
    """
    

    config = getParametersForYr("./configs/parameters/" , args.year)
    coffea_processor = EventProcessor(config, test_mode=test_mode)

    if not test_mode: # full scale implementation
        # # original ---------------------------------------------------------
        if args.use_gateway:
            from dask_gateway import Gateway
            gateway = Gateway(
                "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
                proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
            )
            cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
            client = gateway.connect(cluster_info.name).get_client()
            print(f"client: {client}")
            print("Gateway Client created")
        # # #-----------------------------------------------------------
        else:
            # client = Client(n_workers=1,  threads_per_worker=1, processes=True, memory_limit='15 GiB') 
            # client = Client(n_workers=15,  threads_per_worker=1, processes=True, memory_limit='6 GiB') 
            client = Client(n_workers=12,  threads_per_worker=1, processes=True, memory_limit='10 GiB') 
            print("Local scale Client created")
        #-------------------------------------------------------------------------------------
        #-----------------------------------------------------------
        # client = Client(n_workers=8,  threads_per_worker=1, processes=True, memory_limit='8 GiB') 
        #---------------------------------------------------------
        # print("cluster scale up")
        # sample_path = "./prestage_output/processor_samples.json"
        sample_path = "./prestage_output/fraction_processor_samples.json"
        with open(sample_path) as file:
            samples = json.loads(file.read())
        # add in NanoAODv info into samples metadata for coffea processor
        for dataset in samples.keys():
            samples[dataset]["metadata"]["NanoAODv"] = args.NanoAODv
        start_save_path = args.save_path + f"/{args.year}"
        print(f"start_save_path: {start_save_path}")
        # with performance_report(filename="dask-report.html"):
        # for dataset, sample in samples.items():
        # dask.config.set(scheduler='single-threaded')
        with performance_report(filename="dask-report.html"):
            for dataset, sample in tqdm.tqdm(samples.items()):
            # for dataset, sample in samples.items():
                sample_step = time.time()
                # max_file_len = 15
                # max_file_len = 50
                max_file_len = 100
                # max_file_len = 200
                # max_file_len = 25
                # max_file_len = 900
                # max_file_len = 10
                smaller_files = list(divide_chunks(sample["files"], max_file_len))
                # print(f"smaller_files: {smaller_files}")
                print(f"max_file_len: {max_file_len}")
                print(f"len(smaller_files): {len(smaller_files)}")
                # for idx in range(len(smaller_files)):
                # for idx in tqdm.tqdm(range(2, len(smaller_files)), leave=False):
                for idx in tqdm.tqdm(range(len(smaller_files)), leave=False):
                    # print("restarting workers!")
                    # client.restart(wait_for_workers = False)
                    smaller_sample = copy.deepcopy(sample)
                    smaller_sample["files"] = smaller_files[idx]
                    var_step = time.time()
                    # print(f"var_step: {var_step}")
                    to_persist = dataset_loop(coffea_processor, smaller_sample, file_idx=idx, test=test_mode, save_path=start_save_path)
                    save_path = getSavePath(start_save_path, smaller_sample, idx)
                    print(f"save_path: {save_path}")
                    # remove previously existing files and make path if doesn't exist
                    filelist = glob.glob(f"{save_path}/*.parquet")
                    print(f"len(filelist): {len(filelist)}")
                    for file in filelist:
                        os.remove(file)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    to_persist.persist().to_parquet(save_path)
                    
                    var_elapsed = round(time.time() - var_step, 3)
                    print(f"Finished file_idx {idx} in {var_elapsed} s.")
                sample_elapsed = round(time.time() - sample_step, 3)
                print(f"Finished sample {dataset} in {sample_elapsed} s.")
                
    else:
        # dataset_loop(coffea_processor, xrootd_path+fname, test=test_mode)

        sample_path = "./prestage_output/fraction_processor_samples.json"
        with open(sample_path) as file:
            samples = json.loads(file.read())
        # print(f"samples.keys(): {samples.keys()}")

        with performance_report(filename="dask-report.html"):
            # for dataset, sample in samples.items():
            for dataset, sample in tqdm.tqdm(samples.items()):
                
                # testing
                # if ("dy_M-100To200" not in dataset) and ("data_A" not in dataset):
                # if ("dy_M-100To200" not in dataset) :
                # if ("ggh_powheg" not in dataset) and ("vbf_powheg" not in dataset):
                #     continue
                # print(f"dataset: {dataset}")
                dataset_loop(coffea_processor, sample, test=test_mode)
        
    

    

    
    elapsed = round(time.time() - time_step, 3)
    print(f"Finished everything in {elapsed} s.")
