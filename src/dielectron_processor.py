import coffea.processor as processor
from coffea.lookup_tools import extractor
import awkward as ak
import numpy as np
from typing import Union, TypeVar, Tuple
import correctionlib
from src.corrections.rochester import apply_roccor
from src.corrections.fsr_recovery import fsr_recovery, fsr_recoveryV1
from src.corrections.geofit import apply_geofit
from src.corrections.jet import get_jec_factories, jet_id, jet_puid, fill_softjets
# from src.corrections.weight import Weights
from src.corrections.evaluator import pu_evaluator, nnlops_weights, musf_evaluator, get_musf_lookup, lhe_weights, stxs_lookups, add_stxs_variations, add_pdf_variations, qgl_weights, qgl_weights_eager, qgl_weights_keepDim, btag_weights_json, btag_weights_jsonKeepDim, get_jetpuid_weights
import json
from coffea.lumi_tools import LumiMask
import pandas as pd # just for debugging
import dask_awkward as dak
import dask
from coffea.analysis_tools import Weights
import copy
from coffea.nanoevents.methods import vector
import sys

coffea_nanoevent = TypeVar('coffea_nanoevent') 
ak_array = TypeVar('ak_array')

save_path = "/depot/cms/users/yun79/results/stage1/DNN_test//2018/f0_1/data_B/0" # for debugging

def getRapidity(obj):
    px = obj.pt * np.cos(obj.phi)
    py = obj.pt * np.sin(obj.phi)
    pz = obj.pt * np.sinh(obj.eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + obj.mass**2)
    rap = 0.5 * np.log((e + pz) / (e - pz))
    return rap


def _mass2_kernel(t, x, y, z):
    return t * t - x * x - y * y - z * z

def testJetVector(jets):
    """
    This is a helper function in debugging observed inconsistiency in Jet variables after
    migration from coffea native vectors to hep native vectors
    params:
    jets -> nanoevent vector of Jet. IE: events.Jet
    """
    padded_jets = ak.pad_none(jets, target=2)
    # print(f"type padded_jets: {type(padded_jets.compute())}")
    jet1 = padded_jets[:, 0]
    jet2 = padded_jets[:, 1]
    normal_dijet =  jet1 + jet2
    print(f"type normal_dijet: {type(normal_dijet.compute())}")
    # explicitly reinitialize the jets
    jet1_4D_vec = ak.zip({"pt":jet1.pt, "eta":jet1.eta, "phi":jet1.phi, "mass":jet1.mass}, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
    jet2_4D_vec = ak.zip({"pt":jet2.pt, "eta":jet2.eta, "phi":jet2.phi, "mass":jet2.mass}, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
    new_dijet = jet1_4D_vec + jet2_4D_vec
    target_arr = ak.fill_none(new_dijet.mass.compute(), value=-99.0)
    out_arr = ak.fill_none(normal_dijet.mass.compute(), value=-99.0)
    rel_err = np.abs((target_arr-out_arr)/target_arr)
    print(f"max rel_err: {ak.max(rel_err)}")

# Dmitry's implementation of delta_r
def delta_r_V1(eta1, eta2, phi1, phi2):
    deta = abs(eta1 - eta2)
    dphi = abs(np.mod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi)
    dr = np.sqrt(deta**2 + dphi**2)
    return deta, dphi, dr



class EventProcessor(processor.ProcessorABC):
    # def __init__(self, config_path: str,**kwargs):
    def __init__(self, config: dict, test_mode=False, **kwargs):
        """
        TODO: replace all of these with self.config dict variable which is taken from a
        pre-made json file
        """
        self.config = config

        self.test_mode = test_mode
        dict_update = {
            "apply_LHE_Filter" : False,
            "do_trigger_match" : False, # False
            "do_roccor" : False,# True
            "do_fsr" : False, # True
            "do_geofit" : False, # True
            "do_beamConstraint": False, # if True, override do_geofit
            "do_nnlops" : False,
            "do_pdf" : False,
        }
        self.config.update(dict_update)
        

        # --- Evaluator
        extractor_instance = extractor()
        ##---specify which channel to run on----##
        #channel = self.config["channel"]

        #Aman edits

        year = self.config["year"]

        # PU ID weights
        jetpuid_filename = self.config["jetpuid_sf_file"]
        extractor_instance.add_weight_sets([f"* * {jetpuid_filename}"])
        
        extractor_instance.finalize()
        self.evaluator = extractor_instance.make_evaluator()

    def process(self, events: coffea_nanoevent):
        year = self.config["year"]
        """
        TODO: Once you're done with testing and validation, do LHE cut after HLT and trigger match event filtering to save computation
        """
    


        """
        Apply LHE cuts for DY sample stitching
        Basically remove events that has dilepton mass between 100 and 200 GeV
        """
        # print("testJetVector right as process starts")
        # testJetVector(events.Jet)
        
        #event_filter = ak.ones_like(events.HLT.IsoMu24) # 1D boolean array to be used to filter out bad events
        event_filter = ak.ones_like(events.event) # 1D boolean array to be used to filter out bad events
        dataset = events.metadata['dataset']
        print(f"dataset: {dataset}")
        #print(f"events.metadata: {events.metadata}")
        NanoAODv = events.metadata['NanoAODv']
        is_mc = events.metadata['is_mc']
        #print(f"NanoAODv: {NanoAODv}")
        # LHE cut original start -----------------------------------------------------------------------------
        if ((self.config["apply_LHE_Filter"] == True ) and (dataset == 'dy_M-50')): # if dy_M-50, apply LHE cut
            print("doing dy_M-50 LHE cut!")
            LHE_particles = events.LHEPart #has unique pdgIDs of [ 1,  2,  3,  4,  5, 11, 13, 15, 21]
            bool_filter = (abs(LHE_particles.pdgId) == 11) | (abs(LHE_particles.pdgId) == 13) | (abs(LHE_particles.pdgId) == 15)
            LHE_leptons = LHE_particles[bool_filter]


            """
            TODO: maybe we can get faster by just indexing first and second, instead of argmax and argmins
            When I had a quick look, all LHE_leptons had either two or zero leptons per event, never one, 
            so just indexing first and second could work
            """
            max_idxs = ak.argmax(LHE_leptons.pdgId , axis=1,keepdims=True) # get idx for normal lepton
            min_idxs = ak.argmin(LHE_leptons.pdgId , axis=1,keepdims=True) # get idx for anti lepton
            LHE_lepton_barless = LHE_leptons[max_idxs]
            LHE_lepton_bar = LHE_leptons[min_idxs]
            LHE_dilepton_mass =  (LHE_lepton_barless +LHE_lepton_bar).mass

            # LHE_filter = ak.flatten(((LHE_dilepton_mass > 100) & (LHE_dilepton_mass < 200)))
            LHE_filter = (((LHE_dilepton_mass > 100) & (LHE_dilepton_mass < 200)))[:,0]
            # print(f"LHE_filter: {LHE_filter.compute()}")
            LHE_filter = ak.fill_none(LHE_filter, value=False) 
            LHE_filter = (LHE_filter== False) # we want True to indicate that we want to keep the event
            # print(f"copperhead2 EventProcessor LHE_filter[32]: \n{ak.to_numpy(LHE_filter[32])}")

            event_filter = event_filter & LHE_filter
        # LHE cut original end -----------------------------------------------------------------------------
                    
            

# just reading test start --------------------------------------------------------------------------        

            
        # # Apply HLT to both Data and MC. NOTE: this would probably be superfluous if you already do trigger matching
        HLT_filter = ak.zeros_like(event_filter, dtype="bool")  # start with 1D of Falses

        for HLT_str in self.config["el_hlt"]:
            HLT_filter = HLT_filter | events.HLT[HLT_str]
            
        event_filter = event_filter & HLT_filter

        # ------------------------------------------------------------#
        # Skimming end, filter out events and prepare for pre-selection
        # Edit: NVM; doing it this stage breaks fsr recovery
        # ------------------------------------------------------------#

        
        if is_mc:
            lumi_mask = ak.ones_like(event_filter)

        
        else:
            lumi_info = LumiMask(self.config["lumimask"])
            lumi_mask = lumi_info(events.run, events.luminosityBlock)


        do_pu_wgt = True
        if self.test_mode is True: # this override should prob be replaced with something more robust in the future, or just be removed
            do_pu_wgt = False # basic override bc PU due to slight differences in implementation copperheadV1 and copperheadV2 implementation

        if do_pu_wgt:
            print("doing PU re-wgt!")
            # obtain PU reweighting b4 event filtering, and apply it after we finalize event_filter
            print(f"year: {year}")
            if ("22" in year) or ("23" in year) or ("24" in year):
                run_campaign = 3
            else:
                run_campaign = 2
            print(f"run_campaign: {run_campaign}")
            if is_mc:
                pu_wgts = pu_evaluator(
                            self.config,
                            events.Pileup.nTrueInt,
                            onTheSpot=False, # use locally saved true PU dist
                            Run = run_campaign
                    )
       
        # # Save raw variables before computing any corrections
        # # rochester and geofit corrects pt only, but fsr_recovery changes all vals below
        # attempt at fixing fsr issue start -------------------------------------------------------------------
        events["Electron", "pt_raw"] = ak.ones_like(events.Electron.pt) * events.Electron.pt
        events["Electron", "eta_raw"] = ak.ones_like(events.Electron.eta) * events.Electron.eta
        events["Electron", "phi_raw"] = ak.ones_like(events.Electron.phi) * events.Electron.phi
        



        # --------------------------------------------------------#
        # Select electrons that pass pT, eta,
        # Select events with 2 good electrons, no muons,
        # passing quality cuts and at least one good PV
        # --------------------------------------------------------#

        # Apply event quality flags
        evnt_qual_flg_selection = ak.ones_like(event_filter)
        for evt_qual_flg in self.config["event_flags"]:
            evnt_qual_flg_selection = evnt_qual_flg_selection & events.Flag[evt_qual_flg]

        
        electron_selection = (
            (events.Electron.pt_raw >= self.config["electron_pt_cut"])
            & (abs(events.Electron.eta_raw) < self.config["electron_eta_cut"])
            & events.Electron[self.config[f"electron_id_v{NanoAODv}"]]
        )


        electrons = events.Electron[electron_selection]
        print(electrons) 
        
        # count electrons that pass the electron selection
        nelectrons = ak.num(electrons, axis=1)
        # Find opposite-sign electrons, but in the analysis we apply no charge selection on electrons
        ee_charge = ak.prod(electrons.charge, axis=1)
        
        muon_id = "looseId"
        # Veto events with good quality muon; 
        muon_selection = (
            (events.Muon.pt > 10.)
            & (abs(events.Muon.eta) < 2.4)
            & events.Muon[muon_id]
        )
        
        muon_veto = (ak.num(events.Muon[muon_selection], axis=1) == 0) 

        
        event_filter = (
                event_filter
                & lumi_mask
                & (evnt_qual_flg_selection > 0)
                & (nelectrons == 2)
                & muon_veto 
                & (events.PV.npvsGood > 0) # number of good primary vertex cut

        )


        # --------------------------------------------------------#
        # Select events with electrons passing leading pT cut
        # --------------------------------------------------------#

        electrons_padded = ak.pad_none(electrons, target=2)
        sorted_args = ak.argsort(electrons_padded.pt, ascending=False) # leadinig pt is ordered by pt
        electrons_sorted = (electrons_padded[sorted_args])
        e1 = electrons_sorted[:,0]
        pass_leading_pt = e1.pt_raw > self.config["electron_leading_pt"]
        pass_leading_pt = ak.fill_none(pass_leading_pt, value=False) 


        event_filter = event_filter & pass_leading_pt
        # test end -----------------------------------------------------------------------

        
        # calculate sum of gen weight b4 skimming off bad events
        if is_mc:
            if self.test_mode: # for small files local testing
                sumWeights = ak.sum(events.genWeight, axis=0) # for testing
                print(f"small file test sumWeights: {(sumWeights.compute())}") # for testing
            else:
                sumWeights = events.metadata['sumGenWgts']
                print(f"sumWeights: {(sumWeights)}")


        # to_packed testing -----------------------------------------------
        events = events[event_filter==True]
        electrons = electrons[event_filter==True]
        nelectrons = ak.to_packed(nelectrons[event_filter==True])

        
        # turn off pu weights test start ---------------------------------
        if is_mc and do_pu_wgt:
            for variation in pu_wgts.keys():
                pu_wgts[variation] = ak.to_packed(pu_wgts[variation][event_filter==True])
        pass_leading_pt = ak.to_packed(pass_leading_pt[event_filter==True])

        
        # --------------------------------------------------------#
        # Fill dielectron and electron variables
        # --------------------------------------------------------#

        # ---------------------------------------------------------
        # TODO: find out why we don't filter out bad events right now via
        # even_selection column, since fill electron is computationally exp
        # Last time I checked there was some errors on LHE correction shape mismatch
        # ---------------------------------------------------------

        electrons_padded = ak.pad_none(electrons, target=2)
        sorted_args = ak.argsort(electrons_padded.pt, ascending=False)
        electrons_sorted = (electrons_padded[sorted_args])
        e1 = electrons_sorted[:,0]
        e2 = electrons_sorted[:,1]
        
        dielectron_dR = e1.delta_r(e2)
        dielectron_dEta = abs(e1.eta - e2.eta)
        dielectron_dPhi = abs(e1.delta_phi(e2))
        dielectron = e1+e2
        
        # #fill genjets
        
        #if is_mc:
        #    gjets = events.GenJet
        #    gleptons = events.GenPart[
        #        (
        #            (abs(events.GenPart.pdgId) == 13)
        #            | (abs(events.GenPart.pdgId) == 11)
        #            | (abs(events.GenPart.pdgId) == 15)
        #        )
        #        & events.GenPart.hasFlags('isHardProcess')
        #    ]
        #    gl_pair = ak.cartesian({"jet": gjets, "lepton": gleptons}, axis=1, nested=True)
        #    dr_gl = gl_pair["jet"].delta_r(gl_pair["lepton"])
        #    isolated = ak.all((dr_gl > 0.3), axis=-1) # this also returns true if there's no leptons near the gjet

        #    # same order sorting algorithm as reco jet start -----------------
        #    gjets = ak.to_packed(gjets[isolated])
        #    # print(f"gjets.pt: {gjets.pt.compute()}")
        #    sorted_args = ak.argsort(gjets.pt, ascending=False)
        #    sorted_gjets = (gjets[sorted_args])
        #    gjets_sorted = ak.pad_none(sorted_gjets, target=2) 
        #    # same order sorting algorithm as reco jet end -----------------
        #    
        #    # print(f"gjets_sorted: {gjets_sorted.compute()}")
        #    gjet1 = gjets_sorted[:,0]
        #    gjet2 = gjets_sorted[:,1] 
        #    # original start -----------------------------------------------
        #    gjj = gjet1 + gjet2
        #    
        #    gjj_dEta = abs(gjet1.eta - gjet2.eta)
        #    gjj_dPhi = abs(gjet1.delta_phi(gjet2))
        #    gjj_dR = gjet1.delta_r(gjet2)


        #self.prepare_jets(events, NanoAODv=NanoAODv)
        # print("test ject vector right after prepare_jets")
        # testJetVector(events.Jet)

        # ------------------------------------------------------------#
        # Apply JEC, get JEC and JER variations
        # ------------------------------------------------------------#
        year = self.config["year"]
        jets = events.Jet
        #self.jec_factories_mc, self.jec_factories_data = get_jec_factories(
        #    self.config["jec_parameters"], 
        #    year
        #)   
        
        do_jec = False # True       
        # do_jecunc = self.config["do_jecunc"]
        # do_jerunc = self.config["do_jerunc"]
        #testing 
        do_jecunc = False
        do_jerunc = False
        # cache = events.caches[0]
        factory = None
        if do_jec:
            if is_mc:
                factory = self.jec_factories_mc["jec"]
            else:
                for run in self.config["jec_parameters"]["runs"]:
                    # print(f"run: {run}")
                    if run in dataset:
                        factory = self.jec_factories_data[run]
                if factory == None:
                    print("JEC factory not recognized!")
                    raise ValueError
                
            print("do jec!")
            print("test ject vector b4 JEC")
            # testJetVector(jets)
            jets = factory.build(jets)
            print("test ject vector after JEC")
            # testJetVector(jets)

        else:
            jets["mass_jec"] = jets.mass
            jets["pt_jec"] = jets.pt

        

        # # ------------------------------------------------------------#
        # # Apply genweights, PU weights
        # # and L1 prefiring weights
        # # ------------------------------------------------------------#
        weights = Weights(None, storeIndividual=True) # none for dask awkward
        if is_mc:
            weights.add("genWeight", weight=events.genWeight)
            # original initial weight start ----------------
            weights.add("genWeight_normalization", weight=ak.ones_like(events.genWeight)/sumWeights)
            #temporary lhe filter end -----------------
            cross_section = self.config["cross_sections"][dataset]
            #print(f"cross_section: {cross_section}")
            integrated_lumi = self.config["integrated_lumis"]
            weights.add("xsec", weight=ak.ones_like(events.genWeight)*cross_section)
            weights.add("lumi", weight=ak.ones_like(events.genWeight)*integrated_lumi)
            # original initial weight end ----------------
            
            if do_pu_wgt:
                print("adding PU wgts!")
                weights.add("pu", weight=pu_wgts["nom"],weightUp=pu_wgts["up"],weightDown=pu_wgts["down"])
            # L1 prefiring weights
            if self.config["do_l1prefiring_wgts"] and ("L1PreFiringWeight" in events.fields):
                L1_nom = events.L1PreFiringWeight.Nom
                L1_up = events.L1PreFiringWeight.Up
                L1_down = events.L1PreFiringWeight.Dn
                weights.add("l1prefiring", 
                    weight=L1_nom,
                    weightUp=L1_up,
                    weightDown=L1_down
                )
        else: # data-> just add in ak ones for consistency
            weights.add("ones", weight=ak.values_astype(ak.ones_like(events.HLT.IsoMu24), "float32"))
        
          

        
        # ------------------------------------------------------------#
        # Calculate other event weights
        # ------------------------------------------------------------#
        pt_variations = (
            ["nominal"]
            # + jec_pars["jec_variations"]
            # + jec_pars["jer_variations"]
        )
        if is_mc:
            # moved nnlops reweighting outside of dak process and to run_stage1-----------------
            do_nnlops = self.config["do_nnlops"] and ("ggh" in events.metadata["dataset"])
            if do_nnlops:
                print("doing NNLOPS!")
                nnlopsw = nnlops_weights(events.HTXS.Higgs_pt, events.HTXS.njets30, self.config, events.metadata["dataset"])
                weights.add("nnlops", weight=nnlopsw)
            


            
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            do_lhe = (
                ("LHEScaleWeight" in events.fields)
                and ("LHEPdfWeight" in events.fields)
                and ("nominal" in pt_variations)
            )
            if do_lhe:
                print("doing LHE!")
                lhe_ren, lhe_fac = lhe_weights(events, events.metadata["dataset"], self.config["year"])
                weights.add("LHERen", 
                    weight=ak.ones_like(lhe_ren["up"]),
                    weightUp=lhe_ren["up"],
                    weightDown=lhe_ren["down"]
                )
                weights.add("LHEFac", 
                    weight=ak.ones_like(lhe_fac["up"]),
                    weightUp=lhe_fac["up"],
                    weightDown=lhe_fac["down"]
                )
            
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            dataset = events.metadata["dataset"]

            do_thu = False
            if do_thu:
                print("doing THU!")
                add_stxs_variations(
                    events,
                    weights,
                    self.config,
                )
                

            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            do_pdf = (
                self.config["do_pdf"]
                and ("nominal" in pt_variations)
                and (
                    "dy" in dataset
                    or "ewk" in dataset
                    or "ggh" in dataset
                    or "vbf" in dataset
                )
                and ("mg" not in dataset)
            )
            if do_pdf:
                print("doing pdf!")
                # add_pdf_variations(events, self.weight_collection, self.config, dataset)
                pdf_vars = add_pdf_variations(events, self.config, dataset)
                weights.add("pdf_2rms", 
                    weight=ak.ones_like(pdf_vars["up"]),
                    weightUp=pdf_vars["up"],
                    weightDown=pdf_vars["down"]
                )

        # ------------------------------------------------------------#
        # Fill Electron variables and gjet variables
        # ------------------------------------------------------------#
        out_dict = {
            "event" : events.event,
            #"HLT_filter" : HLT_filter, 
            "e1_pt" : e1.pt,
            "e2_pt" : e2.pt,
            "e1_eta" : e1.eta,
            "e2_eta" : e2.eta,
            #"mu1_phi" : mu1.phi,
            #"mu2_phi" : mu2.phi,
            #"mu1_charge" : mu1.charge,
            #"mu2_charge" : mu2.charge,
            #"mu1_iso" : mu1.Iso_raw,
            #"mu2_iso" : mu2.Iso_raw,
            #"nmuons" : nmuons,
            "dielectron_mass" : dielectron.mass,
            #"dimuon_pt" : dimuon.pt,
            #"dimuon_eta" : dimuon.eta,
            #"dimuon_rapidity" : getRapidity(dimuon),
            #"dimuon_phi" : dimuon.phi,
            #"dimuon_dEta" : dimuon_dEta,
            #"dimuon_dPhi" : dimuon_dPhi,
            #"dimuon_dR" : dimuon_dR,
            #"dimuon_ebe_mass_res" : dimuon_ebe_mass_res,
            #"dimuon_cos_theta_cs" : dimuon_cos_theta_cs,
            #"dimuon_phi_cs" : dimuon_phi_cs,
            #"dimuon_cos_theta_eta" : dimuon_cos_theta_eta,
            #"dimuon_phi_eta" : dimuon_phi_eta,
            #"mu1_pt_raw" : mu1.pt_raw,
            #"mu2_pt_raw" : mu2.pt_raw,
            #"mu1_pt_fsr" : mu1.pt_fsr,
            #"mu2_pt_fsr" : mu2.pt_fsr,
            #"pass_leading_pt" : pass_leading_pt,
        }
        wgt_nominal = weights.weight()
        weight_dict = {"wgt_nominal_total" : wgt_nominal}
        print("weight_dict ", weight_dict)
        #out_dict.update(weight_dict)
        print(" i think the problem is here ")
        return out_dict
        
    def postprocess(self, accumulator):
        """
        Arbitrary postprocess function that's required to run the processor
        """
        pass

    
