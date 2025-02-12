from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
#from src.dielectron_processor import EventProcessor
from src.copperhead_processor import EventProcessor
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
        save_path = "/depot/cms/users/kaur214/analysis_facility/outputs/elec/" # default
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

    
    out_collections = processor.process(events)
    dataset_fraction = dataset_dict["metadata"]["fraction"]


    skim_dict = out_collections
    skim_dict["fraction"] = dataset_fraction*(ak.ones_like(out_collections["event"]))

    skim_zip = ak.zip(skim_dict, depth_limit=1)
    # print(f"skim_zip: {skim_zip}")
    # skim_zip.persist().to_parquet(save_path)
    # raise ValueError
    return skim_zip

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
    default="/depot/cms/users/kaur214/analysis_facility/outputs/elec/",
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

    parser.add_argument(
    "-fl",
    "--flavor",
    dest="channel",
    default="mumu",
    action="store",
    help="specify which channel to run on.",
    )

    args = parser.parse_args()
    # make NanoAODv into an interger variable
    print(f"args.NanoAODv: {args.NanoAODv}")
    print(f"args.year: {args.year}")
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
    # print(f"stage1 config: {config}")
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
        sample_path = "/depot/cms/users/kaur214/analysis_facility/outputs/elec/prestage_output/processor_samples.json"
        #sample_path = "/depot/cms/users/kaur214/analysis_facility/outputs/prestage_output/processor_samples.json"
        #sample_path = "/depot/cms/users/kaur214/analysis_facility/outputs/prestage_output/fraction_processor_samples.json"
        with open(sample_path) as file:
            samples = json.loads(file.read())
        # add in NanoAODv info into samples metadata for coffea processor
        for dataset in samples.keys():
            samples[dataset]["metadata"]["NanoAODv"] = args.NanoAODv
        start_save_path = args.save_path + f"/stage1_output/{args.year}"
        print(f"start_save_path: {start_save_path}")
        # with performance_report(filename="dask-report.html"):
        # for dataset, sample in samples.items():
        # dask.config.set(scheduler='single-threaded')
        with performance_report(filename="dask-report.html"):
            for dataset, sample in tqdm.tqdm(samples.items()):
            # for dataset, sample in samples.items():
                sample_step = time.time()
                max_file_len = 10 ##number of files per job
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

        sample_path = "/depot/cms/users/kaur214/analysis_facility/outputs/elec/prestage_output/fraction_processor_samples.json"
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
