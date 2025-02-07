from coffea.jetmet_tools import CorrectedJetsFactory, JECStack
from coffea.lookup_tools import extractor
import numpy as np
import awkward as ak
import os


# def jec_names_and_sources_yaml(jec_pars, year):
#     localdir = os.path.dirname(os.path.abspath("__file__"))
#     print(f"localdir: {localdir}")
#     OmegaConf.register_new_resolver("default_params_dir", lambda: localdir+"/data", replace=True)
#     jec_dict = OmegaConf.load(localdir+'/parameters/jets_calibration.yaml')
#     jec_fnames = jec_dict['default_jets_calibration']["factory_configuration"]["AK4PFchs"]["JES_JER_Syst"][year]
#     names = {}
#     suffix = {
#         "jec_names": [f"_{level}_AK4PFchs" for level in jec_pars["jec_levels_mc"]],
#         "junc_names": ["_Uncertainty_AK4PFchs"],
#         "junc_sources": ["_UncertaintySources_AK4PFchs"],
#         "jer_names": ["_PtResolution_AK4PFchs"],
#         "jersf_names": ["_SF_AK4PFchs"],
#     }

#     for key, suff in suffix.items():
#         if "data" in key:
#             names[key] = {}
#             for run in jec_pars["runs"]:
#                 for tag, iruns in jec_pars["jec_data_tags"].items():
#                     if run in iruns:
#                         names[key].update({run: [f"{tag}{s}" for s in suff]})
#         else:
#             tag = jec_pars["jer_tags"] if "jer" in key else jec_pars["jec_tags"]
#             names[key] = [f"{tag}{s}" for s in suff]

#     return names

def jec_names_and_sources(jec_pars):
    names = {}
    suffix = {
        "jec_names": [f"_{level}_AK4PFchs" for level in jec_pars["jec_levels_mc"]],
        "jec_names_data": [
            f"_{level}_AK4PFchs" for level in jec_pars["jec_levels_data"]
        ],
        "junc_names": ["_Uncertainty_AK4PFchs"],
        "junc_names_data": ["_Uncertainty_AK4PFchs"],
        "junc_sources": ["_UncertaintySources_AK4PFchs"],
        "junc_sources_data": ["_UncertaintySources_AK4PFchs"],
        "jer_names": ["_PtResolution_AK4PFchs"],
        "jersf_names": ["_SF_AK4PFchs"],
    }

    for key, suff in suffix.items():
        if "data" in key:
            names[key] = {}
            for run in jec_pars["runs"]:
                for tag, iruns in jec_pars["jec_data_tags"].items():
                    if run in iruns:
                        names[key].update({run: [f"{tag}{s}" for s in suff]})
        else:
            tag = jec_pars["jer_tags"] if "jer" in key else jec_pars["jec_tags"]
            names[key] = [f"{tag}{s}" for s in suff]

    return names


def jec_weight_sets(jec_pars, year):
    weight_sets = {}
    names = jec_names_and_sources(jec_pars)

    extensions = {
        "jec_names": "jec",
        "jer_names": "jr",
        "jersf_names": "jersf",
        "junc_names": "junc",
        "junc_sources": "junc",
    }

    weight_sets["jec_weight_sets_mc"] = []
    weight_sets["jec_weight_sets_data"] = []

    for opt, ext in extensions.items():
        # MC
        weight_sets["jec_weight_sets_mc"].extend(
            [f"* * data/jec/{name}.{ext}.txt" for name in names[opt]]
        )
        # Data
        if "jer" in opt:
            continue
        data = []
        for run, items in names[f"{opt}_data"].items():
            data.extend(items)
        data = list(set(data))
        weight_sets["jec_weight_sets_data"].extend(
            [f"* * data/jec/{name}.{ext}.txt" for name in data]
        )

    # return weight_sets
    return (weight_sets, names)


def get_name_map(stack):
    name_map = stack.blank_name_map
    name_map["JetPt"] = "pt"
    name_map["JetMass"] = "mass"
    name_map["JetEta"] = "eta"
    name_map["JetA"] = "area"
    name_map["ptGenJet"] = "pt_gen"
    name_map["ptRaw"] = "pt_raw"
    name_map["massRaw"] = "mass_raw"
    name_map["Rho"] = "PU_rho"
    # print(f"name_map: {name_map}")
    return name_map

def get_jec_factories(jec_parameters: dict, year):
    # jec_pars = {k: v[year] for k, v in jec_parameters.items()}
    # jec_pars = {k: v for k, v in jec_parameters.items()}
    
    jec_pars = jec_parameters


    weight_sets, names = jec_weight_sets(jec_pars, year)

    jec_factories = {}
    jec_factories_data = {}

    # Prepare evaluators for JEC, JER and their systematics
    jetext = extractor()
    jetext.add_weight_sets(weight_sets["jec_weight_sets_mc"])
    jetext.add_weight_sets(weight_sets["jec_weight_sets_data"])
    jetext.finalize()
    jet_evaluator = jetext.make_evaluator()

    stacks_def = {
        "jec_stack": ["jec_names"],
        "jer_stack": ["jer_names", "jersf_names"],
        "junc_stack": ["junc_names"],
    }

    stacks = {}
    for key, vals in stacks_def.items():
        stacks[key] = []
        for v in vals:
            stacks[key].extend(names[v])


    jec_input_options = {}
    jet_variations = ["jec", "junc", "jer"]
    
    for variation in jet_variations:
        # jec_input_options[variation] = {
        #     name: jet_evaluator[name] for name in stacks[f"{variation}_stack"]
        # }
        """
        matches names specific for jet variation with the appropriate jet evaluator
        """
        # jec_input_options[opt] = {
        #     name: jet_evaluator[name] for name in stacks[f"{opt}_stack"]
        # }
        jec_input_options[variation] ={}
        for name in stacks[f"{variation}_stack"]:
            jec_input_options[variation][name] =jet_evaluator[name] 
        
    # print(f"jec_factories jec_input_options: \n {jec_input_options}")
    for src in names["junc_sources"]:
        for key in jet_evaluator.keys():
            if src in key:
                jec_input_options["junc"][key] = jet_evaluator[key]

    # Create separate factories for JEC, JER, JEC variations
    for variation in jet_variations:

        
        stack = JECStack(jec_input_options[variation])
        # print(f"jec_factories JECStack: {stack}")
        # print(f"jec_factories get_name_map(stack): {get_name_map(stack)}")
        jec_factories[variation] = CorrectedJetsFactory(get_name_map(stack), stack)

    # Create a separate factory for each data run
    for run in jec_pars["runs"]:
        jec_inputs_data = {}
        for opt in ["jec", "junc"]:
            jec_inputs_data.update(
                {name: jet_evaluator[name] for name in names[f"{opt}_names_data"][run]}
            )
        for src in names["junc_sources_data"][run]:
            for key in jet_evaluator.keys():
                if src in key:
                    jec_inputs_data[key] = jet_evaluator[key]

        jec_stack_data = JECStack(jec_inputs_data)
        jec_factories_data[run] = CorrectedJetsFactory(
            get_name_map(jec_stack_data), jec_stack_data
        )

    return jec_factories, jec_factories_data


    
def jet_id(jets, config):
    # print(f"jets parameters: {parameters}")
    pass_jet_id = ak.ones_like(jets.jetId, dtype=bool)
    if "loose" in config["jet_id"]:
        pass_jet_id = jets.jetId >= 1
    elif "tight" in config["jet_id"]: # according to https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD#NanoAOD_format , jet Id is same for UL 2016,2017 and 2018
        pass_jet_id = jets.jetId >= 2
            
    return pass_jet_id


def jet_puid(jets, config):
    jet_puid_opt = config["jet_puid"]
    year = config["year"]
    puId = jets.puId
    # jet puid for standard wps are different for 2016 vs 2017,2018 as shown in https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL#Working_Points
    # only apply jet puid to jets with pt < 50, else, pass
    # as stated in https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL
    if "2016" in year:
        jet_puid_wps = {
            "loose": (puId >= 1) | (jets.pt > 50),
            "medium": (puId >= 3) | (jets.pt > 50),
            "tight": (puId >= 7) | (jets.pt > 50),
        }
    else: # 2017 and 2018
        jet_puid_wps = {
            "loose": (puId >= 4) | (jets.pt > 50),
            "medium": (puId >= 6) | (jets.pt > 50),
            "tight": (puId >= 7) | (jets.pt > 50),
        }
    pass_jet_puid = ak.ones_like(jets.jetId, dtype=bool)
    
    if "2017" in year: # for misreco due ot ECAL endcap noise
        eta_window = (abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0)
        pass_jet_puid = (eta_window & jet_puid_wps["tight"]) | ( # tight puid in the noisy eta window, else loose
            (~eta_window) & jet_puid_wps[jet_puid_opt]
        )
    else:
        pass_jet_puid = jet_puid_wps[jet_puid_opt]
    return pass_jet_puid


def fill_softjets(events, jets, mu1, mu2, nmuons, cutoff, test_mode=False):
    if test_mode:
        print(f"jets events.SoftActivityJet.fields: {events.SoftActivityJet.fields}")
        print(f"jets cutoff: {cutoff}")
    events["SoftActivityJet","mass"] = 0
    saj = events.SoftActivityJet
    saj_Njets = events[f"SoftActivityJetNjets{cutoff}"]
    saj_HT = events[f"SoftActivityJetHT{cutoff}"]
    

    njets = ak.num(jets, axis=1)
    padded_jets = ak.pad_none(jets, 2)
    jet1 = padded_jets[:,0]
    jet2 = padded_jets[:,1]
    
    # nmuons = ak.num(muons, axis=1)
    # mu1 = muons[:,0]
    # mu2 = muons[:,1]
    if test_mode:
        print(f"jets njets: {njets}")
        print(f"jets saj.pt: {saj.pt}")
        print(f"jets jet1.pt: {jet1.pt}")
        print(f"jets jet2.pt: {jet2.pt}")
        print(f"jets mu1.pt: {mu1.pt}")
        print(f"jets mu2.pt: {mu2.pt}")
    
    dR_m1 = saj.delta_r(mu1)
    dR_m2 = saj.delta_r(mu2)
    dR_j1 = saj.delta_r(jet1)
    dR_j2 = saj.delta_r(jet2)
    dR_m1_filter = ak.fill_none((dR_m1 < 0.4), value=False, axis=None)
    dR_m2_filter = ak.fill_none((dR_m2 < 0.4), value=False, axis=None)
    dR_j1_filter = ak.fill_none((dR_j1 < 0.4), value=False, axis=None)
    dR_j2_filter = ak.fill_none((dR_j2 < 0.4), value=False, axis=None)
    if test_mode:
        print(f"jets dR_m1_filter: {dR_m1_filter}")
        print(f"jets dR_m2_filter: {dR_m2_filter}")
        print(f"jets dR_j1_filter: {dR_j1_filter}")
        print(f"jets dR_j2_filter: {dR_j2_filter}")
    saj_to_remove = dR_m1_filter | dR_m2_filter | dR_j1_filter | dR_j2_filter
    saj_to_remove = ak.fill_none(saj_to_remove, value=False)
    
    
    footprint = saj[(saj_to_remove) & (saj.pt > cutoff)]
    footprint_sumPt = ak.sum(footprint.pt, axis=1)
    if test_mode:
        print(f"jets saj_to_remove: {saj_to_remove}")
        print(f"jets footprint_sumPt: {ak.to_numpy(footprint_sumPt)}")
    ht_corrected = saj_HT - footprint_sumPt
    footprint_njets = ak.num(footprint, axis=1)
    corrected_njets = saj_Njets - footprint_njets
    
    if test_mode:
        print(f"jets footprint_njets: {ak.to_numpy(footprint_njets)}")
        print(f"jets corrected_njets: {ak.to_numpy(corrected_njets)}")
        print(f"jets saj_Njets: {saj_Njets}")

    evnts_to_correct = (nmuons==2) |(njets > 0) 
    if test_mode:
        print(f"jets evnts_to_correct: {evnts_to_correct}")
        print(f"jets footprint_njets b4: {ak.to_numpy(saj_Njets)}")
        print(f"jets corrected_njets b4: {ak.to_numpy(saj_HT)}")
    
    saj_Njets = ak.where(evnts_to_correct,corrected_njets,saj_Njets)
    saj_HT = ak.where(evnts_to_correct,ht_corrected,saj_HT)

    if test_mode:
        print(f"jets footprint_njets after: {ak.to_numpy(saj_Njets)}")
        print(f"jets corrected_njets after: {ak.to_numpy(saj_HT)}")
    out_dict = {
        f"nsoftjets{cutoff}" : saj_Njets,
        f"htsoft{cutoff}" : saj_HT
    }
    return out_dict