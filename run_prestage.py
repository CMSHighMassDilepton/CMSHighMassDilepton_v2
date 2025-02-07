import awkward as ak
from coffea.dataset_tools import rucio_utils
from coffea.dataset_tools.preprocess import preprocess
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
import json
import os
import argparse
import dask
dask.config.set({'logging.distributed': 'error'})
from distributed import LocalCluster, Client
import time
import copy
import tqdm
import uproot
import random
# random.seed(9002301)
import re
import glob
# import warnings
# warnings.filterwarnings("error", module="coffea.*")
from omegaconf import OmegaConf


def get_Xcache_filelist(fnames: list):
    new_fnames = []
    for fname in fnames:
        root_file = re.findall(r"/store.*", fname)[0]
        x_cache_fname = "root://cms-xcache.rcac.purdue.edu/" + root_file
        new_fnames.append(x_cache_fname)
    return new_fnames
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="year value. The options are: 2016preVFP, 2016postVFP, 2017, 2018",
    )
    parser.add_argument(
    "-ch",
    "--chunksize",
    dest="chunksize",
    default="10000",
    action="store",
    help="chunksize",
    )
    parser.add_argument(
    "-frac",
    "--change_fraction",
    dest="fraction",
    default=None,
    action="store",
    help="change fraction of steps of the data",
    )
    parser.add_argument(
    "-data",
    "--data",
    dest="data_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of data samples represented by alphabetical letters A-H",
    )
    parser.add_argument(
    "-bkg",
    "--background",
    dest="bkg_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of bkg samples represented by shorthands: DY, TT, ST, DB (diboson), EWK",
    )
    parser.add_argument(
    "-sig",
    "--signal",
    dest="sig_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of sig samples represented by shorthands: ggH, VBF",
    )
    parser.add_argument(
    "--use_gateway",
    dest="use_gateway",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses dask gateway client instead of local",
    )
    parser.add_argument(
    "--xcache",
    dest="xcache",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses xcache root file paths",
    )
    parser.add_argument(
    "--skipBadFiles",
    dest="skipBadFiles",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses skips bad files when calling preprocessing",
    )
    parser.add_argument(
    "-aod_v",
    "--NanoAODv",
    dest="NanoAODv",
    default="9",
    action="store",
    help="version number of NanoAOD samples we're working with. currently, only 9 and 12 are supported",
    )
    parser.add_argument( # temp flag to test the 2 percent data discrepancy in ggH cat between mine and official workspace
    "--run2_rereco",
    dest="run2_rereco",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses skips bad files when calling preprocessing",
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
    # print(f"args.bkg_samples: {args.bkg_samples}")
    os.environ['XRD_REQUESTTIMEOUT']="2400" # some root files via XRootD may timeout with default value
    if args.fraction is None: # do the normal prestage setup
        # allowlist_sites=["T2_US_Purdue"] # take data only from purdue for now
        allowlist_sites=["T2_US_Purdue", "T2_US_MIT","T2_US_FNAL"]
        total_events = 0
        # get dask client
        # turning off seperate client test start --------------------------------------------------------
        if args.use_gateway:
            from dask_gateway import Gateway
            gateway = Gateway(
                "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
                proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
            )
            cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
            client = gateway.connect(cluster_info.name).get_client()
            print("Gateway Client created")
        else: # use local cluster
            # cluster = LocalCluster(processes=True)
            # cluster.adapt(minimum=8, maximum=31) #min: 8 max: 32
            # client = Client(cluster)
            client = Client(n_workers=15,  threads_per_worker=1, processes=True, memory_limit='30 GiB')
            print("Local scale Client created")
        # turning off seperate client test end --------------------------------------------------------
        big_sample_info = {}
        year = args.year
        
        # load dataset sample paths from yaml files
        filelist = glob.glob("./configs/datasets" + "/*.yaml")
        dataset_confs = [OmegaConf.load(f) for f in filelist]
        datasets = OmegaConf.merge(*dataset_confs)
        if args.run2_rereco: # temp condition for RERECO data case
            dataset = datasets[f"{year}_RERECO"]
        else: # normal
            dataset = datasets[year]
            print("aman is here ", dataset)
        new_sample_list = []
       
        # take data
        data_l =  [sample_name for sample_name in dataset.keys() if "data" in sample_name]
        data_samples = args.data_samples
        # print(f"data_samples: {data_samples}")
        # print(f"data_l: {data_l}")
        if len(data_samples) >0:
            for data_letter in data_samples:
                for sample_name in data_l:
                    if data_letter in sample_name:
                        new_sample_list.append(sample_name)
        # take bkg
        bkg_samples = args.bkg_samples
        print(f"bkg_samples: {bkg_samples}")
        if len(bkg_samples) >0:
            for bkg_sample in bkg_samples:
                if bkg_sample.upper() == "DY": # enforce upper case to prevent confusion
                    new_sample_list.append("dy_M-50")
                    # new_sample_list.append("dy_M-100To200")
                    # new_sample_list.append("dy_VBF_filter")
                    # new_sample_list.append("dy_m105_160_vbf_amc")
                    # new_sample_list.append("dy_VBF_filter_customJMEoff")
                    # new_sample_list.append("dy_M-50To120")
                    # new_sample_list.append("dy_M-120To200")
                    # new_sample_list.append("dy_VBF_filter_Amandeep10_6_26")
                    # new_sample_list.append("dy_VBF_filter_Amandeep10_6_32")
                    # new_sample_list.append("dy_VBF_filter_Amandeep_fromGenSim")
                    #new_sample_list.append("dy_VBF_filter_fromGridpack")
                    # Run3 specific datasets:
                    # new_sample_list.append("dy_M-50To120")
                    # new_sample_list.append("dy_M-120To200")
                

                
                elif bkg_sample.upper() == "TT": # enforce upper case to prevent confusion
                    new_sample_list.append("ttjets_dl")
                    new_sample_list.append("ttjets_sl")
                elif bkg_sample.upper() == "ST": # enforce upper case to prevent confusion
                    new_sample_list.append("st_tw_top")
                    new_sample_list.append("st_tw_antitop")
                elif bkg_sample.upper() == "VV": # enforce upper case to prevent confusion
                    new_sample_list.append("ww_2l2nu")
                    new_sample_list.append("wz_3lnu")
                    new_sample_list.append("wz_2l2q")
                    new_sample_list.append("wz_1l1nu2q")
                    new_sample_list.append("zz")
                elif bkg_sample.upper() == "EWK": # enforce upper case to prevent confusion
                    new_sample_list.append("ewk_lljj_mll50_mjj120")
                else:
                    print(f"unknown background {bkg_sample} was given!")
        print(new_sample_list) 
        # take sig
        sig_samples = args.sig_samples
        if len(sig_samples) >0:
            for sig_sample in sig_samples:
                if sig_sample.upper() == "GGH": # enforce upper case to prevent confusion
                    new_sample_list.append("ggh_powheg")
                    # new_sample_list.append("ggh_amcPS")
                elif sig_sample.upper() == "VBF": # enforce upper case to prevent confusion
                    new_sample_list.append("vbf_powheg")
                else:
                    print(f"unknown signal {sig_sample} was given!")
        
        dataset_dict = {}
        for sample_name in new_sample_list:
            try:
                dataset_dict[sample_name] = dataset[sample_name]
                
            except:
                print(f"Sample {sample_name} doesn't exist. Skipping")
                continue
        dataset = dataset_dict # overwrite dataset bc we don't need it anymore
        print(f"dataset: {dataset}")
        print(f"new dataset: {dataset.keys()}")
        print(f"year: {year}")
        print(f"type(year): {type(year)}")
        print(f"len(year): {len(year)}")
        print(f"args.run2_rereco: {args.run2_rereco}")
        for sample_name in tqdm.tqdm(dataset.keys()):
            is_data =  ("data" in sample_name)
            
            if sample_name == "dy_VBF_filter":
                """
                temporary condition bc IDK a way to integrate prod/phys03 CMSDAS datasets into rucio utils
                """
                # test start -----------------------------------------------------------
                load_path = "/eos/purdue/store/user/vscheure/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/UL18_Nano/240514_124107/"
                fnames = glob.glob(f"{load_path}/*/*.root")

            elif sample_name == "dy_m105_160_vbf_amc":
                """
                load directly from local files
                """
                # test start -----------------------------------------------------------
                load_path = "/eos/purdue/store/mc/RunIIAutumn18NanoAODv6/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/NANOAODSIM"
                fnames = glob.glob(f"{load_path}/*/*/*.root")
                
            elif sample_name == "dy_VBF_filter_fromGridpack":
                """
                load directly from local files
                """
                # test start -----------------------------------------------------------
                load_path = "/eos/purdue/store/user/hyeonseo/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/Flat_NanoAODSIMv9_CMSSW_10_6_26_BigRun/240904_151935/0000/"
                fnames = glob.glob(f"{load_path}/*.root")

            # override the the data path if doing rereco data test
            
            # elif args.run2_rereco:
            #     if year == "2018":
            #         if sample_name == "data_A":
            #             load_path = "/eos/purdue/store/group/local/hmm/FSRnano18ABC_NANOV10b/SingleMuon/RunIISummer16MiniAODv3_FSRnano18ABC_NANOV10b_un2018A-17Sep2018-v2/"
            #             fnames = glob.glob(f"{load_path}/*/*/*.root")
            #         elif sample_name == "data_B":
            #             load_path = "/eos/purdue/store/group/local/hmm/FSRnano18ABC_NANOV10b/SingleMuon/RunIISummer16MiniAODv3_FSRnano18ABC_NANOV10b_un2018B-17Sep2018-v1/"
            #             fnames = glob.glob(f"{load_path}/*/*/*.root")
            #         elif sample_name == "data_C":
            #             load_path = "/eos/purdue/store/group/local/hmm/FSRnano18ABC_NANOV10b/SingleMuon/RunIISummer16MiniAODv3_FSRnano18ABC_NANOV10b_un2018C-17Sep2018-v1/"
            #             fnames = glob.glob(f"{load_path}/*/*/*.root")
            #         elif sample_name == "data_D":
            #             load_path = "/eos/purdue/store/group/local/hmm/FSRnano18D_NANOV10b/SingleMuon/RunIISummer16MiniAODv3_FSRnano18D_NANOV10b_un2018D-22Jan2019-v2/"
            #             fnames = glob.glob(f"{load_path}/*/*/*.root")
            #     elif year == "2017":
            #         if sample_name == "data_B":
            #             load_path = "/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017B-31Mar2018-v1/"
            #             fnames = glob.glob(f"{load_path}/*/*/*.root")
            #         elif sample_name == "data_C":
            #             load_path = "/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017C-31Mar2018-v1/"
            #             fnames = glob.glob(f"{load_path}/*/*/*.root")
            #         elif sample_name == "data_D":
            #             load_path = "/eos/purdue//store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017D-31Mar2018-v1/"
            #             fnames = glob.glob(f"{load_path}/*/*/*.root")
            #         elif sample_name == "data_E":
            #             load_path = "/eos/purdue/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017E-31Mar2018-v1/"
            #             fnames = glob.glob(f"{load_path}/*/*/*.root")
            #         elif sample_name == "data_F":
            #             load_path = "/eos/purdue//store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017F-31Mar2018-v1/"
            #             fnames = glob.glob(f"{load_path}/*/*/*.root")
            #     elif year == "2016":
            #         if sample_name == "data_B":
            #             load_path = "/eos/purdue/store/group/local/hmm/FSRNANO2016DATAV8a/SingleMuon/RunIIData17_FSRNANO2016DATAV8a_un2016B-17Jul2018_ver2-v1/"
            #             fnames = glob.glob(f"{load_path}/*/*/*.root")
            #         elif sample_name == "data_C":
            #             load_path = "/eos/purdue/store/group/local/hmm/FSRNANO2016DATAV8a/SingleMuon/RunIIData17_FSRNANO2016DATAV8a_Run2016C-17Jul2018-v1/"
            #             fnames = glob.glob(f"{load_path}/*/*/*.root")
            #         elif sample_name == "data_D":
            #             load_path = "/eos/purdue/store/group/local/hmm/FSRNANO2016DATAV8a/SingleMuon/RunIIData17_FSRNANO2016DATAV8a_Run2016D-17Jul2018-v1/"
            #             fnames = glob.glob(f"{load_path}/*/*/*.root")
            #         elif sample_name == "data_E":
            #             load_path = "/eos/purdue/store/group/local/hmm/FSRNANO2016DATAV8a/SingleMuon/RunIIData17_FSRNANO2016DATAV8a_Run2016E-17Jul2018-v1/"
            #             fnames = glob.glob(f"{load_path}/*/*/*.root")
            #         elif sample_name == "data_F":
            #             load_path = "/eos/purdue/store/group/local/hmm/FSRNANO2016DATAV8a/SingleMuon/RunIIData17_FSRNANO2016DATAV8a_Run2016F-17Jul2018-v1/"
            #             fnames = glob.glob(f"{load_path}/*/*/*.root")
            #         elif sample_name == "data_G":
            #             load_path = "/eos/purdue/store/group/local/hmm/FSRNANO2016DATAV8a/SingleMuon/RunIIData17_FSRNANO2016DATAV8a_Run2016G-17Jul2018-v1/"
            #             fnames = glob.glob(f"{load_path}/*/*/*.root")
            #         elif sample_name == "data_H":
            #             load_path = "/eos/purdue/store/group/local/hmm/FSRNANO2016DATAV8a/SingleMuon/RunIIData17_FSRNANO2016DATAV8a_Run2016H-17Jul2018-v1/"
            #             fnames = glob.glob(f"{load_path}/*/*/*.root")
            #     else:
            #         print("unknown sample case!")
            #         raise ValueError
                
                
            # comment out the NanoAODv >= 12 condition bc it was for a test, but this code could still be useful, so I am keep it for now
                # elif (args.NanoAODv >= 12) and is_data :
                #     """
                #     temp condition for privately produced Run2 data in NanoAOD v12 format
                #     """
                #     if year == "2016preVFP":
                #         if sample_name == "data_B":
                #             load_path = "/eos/purdue/store/user/vscheure/SingleMuon/UL16preVFP_NanoAODv12/240518_151941/*/*.root"
                #         elif sample_name == "data_C":
                #             load_path = "/eos/purdue/store/user/vscheure/SingleMuon/UL16preVFP_NanoAODv12/240518_152121/*/*.root"
                #         elif sample_name == "data_D":
                #             load_path = "/eos/purdue/store/user/vscheure/SingleMuon/UL16preVFP_NanoAODv12/240518_152208/*/*.root"
                #         elif sample_name == "data_E":
                #             load_path = "/eos/purdue/store/user/vscheure/SingleMuon/UL16preVFP_NanoAODv12/240518_152315/*/*.root"
                #         elif sample_name == "data_F":
                #             load_path = "/eos/purdue/store/user/vscheure/SingleMuon/UL16preVFP_NanoAODv12/240518_152432/*/*.root"
                #         else:
                #             print("unknown v12 Data run!")
                #             raise ValueError
                #     elif year == "2016postVFP":
                #         if sample_name == "data_F":
                #             load_path = "/eos/purdue/store/user/vscheure/SingleMuon/UL16postVFP_NanoAODv12/240517_201444/*/*.root"
                #         elif sample_name == "data_G":
                #             load_path = "/eos/purdue/store/user/vscheure/SingleMuon/UL16postVFP_NanoAODv12/240517_201522/*/*.root"
                #         elif sample_name == "data_H":
                #             load_path = "/eos/purdue/store/user/vscheure/SingleMuon/UL16postVFP_NanoAODv12/240517_201559/*/*.root"
                #         else:
                #             print("unknown v12 Data run!")
                #     elif year == "2017":
                #         if sample_name == "data_B":
                #             load_path = "/eos/purdue/store/user/vscheure/SingleMuon/UL17_NanoAODv12_2/240520_095548/*/*.root"
                #         elif sample_name == "data_C":
                #             load_path = "/eos/purdue/store/user/vscheure/SingleMuon/UL17_NanoAODv12_2/240520_095646/*/*.root"
                #         elif sample_name == "data_D":
                #             load_path = "/eos/purdue/store/user/vscheure/SingleMuon/UL17_NanoAODv12_2/240520_095712/*/*.root"
                #         elif sample_name == "data_E":
                #             load_path = "/eos/purdue/store/user/vscheure/SingleMuon/UL17_NanoAODv12_2/240520_095505/*/*.root"
                #         elif sample_name == "data_F":
                #             load_path = "/eos/purdue/store/user/vscheure/SingleMuon/UL17_NanoAODv12_2/240520_095808/*/*.root"
                #         else:
                #             print("unknown v12 Data run!")
                #             raise ValueError
                #     else:
                #         print("Uncompatible year for privately produced data nanoaodV12!")
                #         raise ValueError
                #     print(f"fnames load_path: {load_path}")
                #     fnames = glob.glob(f"{load_path}")
            else:
                das_query = dataset[sample_name]
                print(f"das_query: {das_query}")
                
                rucio_client = rucio_utils.get_rucio_client()
                outlist, outtree = rucio_utils.query_dataset(
                    das_query,
                    client=rucio_client,
                    tree=True,
                    scope="cms",
                )
                outfiles,outsites,sites_counts =rucio_utils.get_dataset_files_replicas(
                    outlist[0],
                    allowlist_sites=allowlist_sites,
                    mode="full",
                    client=rucio_client,
                    # partial_allowed=True
                )
                fnames = [file[0] for file in outfiles if file != []]
                fnames = [fname.replace("root://eos.cms.rcac.purdue.edu/", "/eos/purdue") for fname in fnames] # replace xrootd prefix bc it's causing file not found error
               
                
                # random.shuffle(fnames)
                if args.xcache:
                    fnames = get_Xcache_filelist(fnames)
            
            print(f"sample_name: {sample_name}")
            print(f"len(fnames): {len(fnames)}")
            fnames = [fname.replace("/eos/purdue", "root://eos.cms.rcac.purdue.edu/") for fname in fnames] # replace xrootd prefix bc it's causing file not found error
            # print(f"fnames: {fnames}")

            
            """
            run through each file and collect total number of 
            """
            preprocess_metadata = {
                "sumGenWgts" : None,
                "nGenEvts" : None,
                "data_entries" : None,
            }
            if is_data: # data sample
                file_input = {fname : {"object_path": "Events"} for fname in fnames}
                events = NanoEventsFactory.from_root(
                        file_input,
                        metadata={},
                        schemaclass=NanoAODSchema,
                        uproot_options={"timeout":2400},
                ).events()
                preprocess_metadata["data_entries"] = int(ak.num(events.Muon.pt, axis=0).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                total_events += preprocess_metadata["data_entries"] 
            else: # if MC
                file_input = {fname : {"object_path": "Runs"} for fname in fnames}
                # print(f"file_input: {file_input}")
                # print(f"file_input: {file_input}")
                # print(len(file_input.keys()))
                runs = NanoEventsFactory.from_root(
                        file_input,
                        metadata={},
                        schemaclass=BaseSchema,
                        uproot_options={"timeout":2400},
                ).events()               
                # print(f"runs.fields: {runs.fields}")
                if sample_name == "dy_m105_160_vbf_amc": # nanoAODv6
                    preprocess_metadata["sumGenWgts"] = float(ak.sum(runs.genEventSumw_).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                    preprocess_metadata["nGenEvts"] = int(ak.sum(runs.genEventCount_).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                else:
                    preprocess_metadata["sumGenWgts"] = float(ak.sum(runs.genEventSumw).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                    preprocess_metadata["nGenEvts"] = int(ak.sum(runs.genEventCount).compute()) # convert into 32bit precision as 64 bit precision isn't json serializable
                total_events += preprocess_metadata["nGenEvts"] 

    
            
            # test start -------------------------------
            if sample_name == "dy_VBF_filter":
                """
                Starting from coffea 2024.4.1, this if statement is technically as obsolite preprocess
                can now handle thousands of root files no problem, but this "manual" is at least three 
                times faster than preprocess, so keeping this if statement for now
                """
                runs = NanoEventsFactory.from_root(
                        file_input,
                        metadata={},
                        schemaclass=BaseSchema,
                        uproot_options={"timeout":2400},
                ).events()  
                genEventCount = runs.genEventCount.compute()
                # print(f"(genEventCount): {(genEventCount)}")
                # print(f"len(genEventCount): {len(genEventCount)}")
                
                assert len(fnames) == len(genEventCount)
                file_dict = {}
                for idx in range(len(fnames)):
                    step_start = 0
                    step_end = int(genEventCount[idx]) # convert into 32bit precision as 64 bit precision isn't json serializable
                    file = fnames[idx]
                    file_dict[file] = {
                        "object_path": "Events",
                        "steps" : [[step_start,step_end]],
                        "num_entries" : step_end, 
                    }
                final_output = {
                    sample_name :{"files" :file_dict}
                }
                # print(f"final_output: {final_output}")
                pre_stage_data = final_output
            else:
                """
                everything else other than VBF filter, but starting from coffea 2024.4.1 is condition could
                techincally work on VBF filter files, just slower
                """
                val = "Events"
                file_dict = {}
                for file in fnames:
                    file_dict[file] = val
                final_output = {
                    sample_name :{"files" :file_dict}
                }
                # print(f"final_output: {final_output}")
                step_size = int(args.chunksize)
                files_available, files_total = preprocess(
                    final_output,
                    step_size=step_size,
                    align_clusters=False,
                    skip_bad_files=args.skipBadFiles,
                )
                # print(f"files_available: {files_available}")
                pre_stage_data = files_available

            # test end2  --------------------------------------------------------------
            # add in metadata
            pre_stage_data[sample_name]['metadata'] = preprocess_metadata
            # add in faction -> for later use
            pre_stage_data[sample_name]['metadata']['fraction'] = 1.0
            pre_stage_data[sample_name]['metadata']['original_fraction'] = 1.0
            # if preprocess_metadata["data_entries"] is not None: # Data
            if "data" in sample_name: # data sample
                pre_stage_data[sample_name]['metadata']["is_mc"] = False
            else: # MC
                pre_stage_data[sample_name]['metadata']["is_mc"] = True
            pre_stage_data[sample_name]['metadata']["dataset"] = sample_name
            big_sample_info.update(pre_stage_data)
        
        #save the sample info
        directory = "./prestage_output"
        filename = directory+"/processor_samples.json"
        dupli_fname = directory+"/fraction_processor_samples.json" # duplicated fname in case you want to skip fractioning
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, "w") as file:
                json.dump(big_sample_info, file)
        with open(dupli_fname, "w") as file:
                json.dump(big_sample_info, file)
    
        elapsed = round(time.time() - time_step, 3)
        print(f"Finished everything in {elapsed} s.")
        print(f"Total Events in files {total_events}.")
        
    else: # take the pre existing samples.json and prune off files we don't need
        fraction = float(args.fraction)
        directory = "./prestage_output"
        sample_path = directory + "/processor_samples.json"
        with open(sample_path) as file:
            samples = json.loads(file.read())
        new_samples = copy.deepcopy(samples) # copy old sample, overwrite it later
        if fraction < 1.0: # else, just save the original samples and new samples
            for sample_name, sample in tqdm.tqdm(samples.items()):
                is_data = "data" in sample_name
                tot_N_evnts = sample['metadata']["data_entries"] if is_data else sample['metadata']["nGenEvts"]
                new_N_evnts = int(tot_N_evnts*fraction)
                old_N_evnts = new_samples[sample_name]['metadata']["data_entries"] if is_data else new_samples[sample_name]['metadata']["nGenEvts"]
                if is_data:
                    print("data!")
                    new_samples[sample_name]['metadata']["data_entries"] = new_N_evnts
                else:
                    new_samples[sample_name]['metadata']["nGenEvts"] = new_N_evnts
                    new_samples[sample_name]['metadata']["sumGenWgts"] *= new_N_evnts/old_N_evnts # just directly multiply by fraction for this since this is already float and this is much faster
                # new_samples[sample_name]['metadata']["fraction"] = fraction
                # state new fraction
                new_samples[sample_name]['metadata']['fraction'] = new_N_evnts/old_N_evnts
                print(f"new_samples[sample_name]['metadata']['fraction']: {new_samples[sample_name]['metadata']['fraction']}")
                # new_samples[sample_name]['metadata']["original_fraction"] = fraction
                
                # loop through the files to correct the steps
                event_counter = 0 # keeps track of events of multiple root files
                stop_flag = False
                new_files = {}
                for file, file_dict in sample["files"].items():
                    if stop_flag:
                        del new_samples[sample_name]["files"][file] # delete the exess files
                        continue
                    new_steps = []
                    # loop through step sizes to correct fractions
                    for step_iteration in file_dict["steps"]:
                        new_step_lim = new_N_evnts-event_counter
                        if step_iteration[1] < new_step_lim:
                            new_steps.append(step_iteration)
                        else:  # change the upper limit
                            new_steps.append([
                                step_iteration[0],
                                new_step_lim
                            ])
                            stop_flag = True
                            break
                    new_samples[sample_name]["files"][file]["steps"] = new_steps # overwrite new steps
                    # add the end step val to the event_counter
                    if not stop_flag: # update variables and move to next file
                        end_idx = len(file_dict["steps"])-1
                        event_counter += file_dict["steps"][end_idx][1]

        #save the sample info
        
        filename = directory+"/fraction_processor_samples.json"
        with open(filename, "w") as file:
                json.dump(new_samples, file)
    
        elapsed = round(time.time() - time_step, 3)
        print(f"Finished everything in {elapsed} s.")

