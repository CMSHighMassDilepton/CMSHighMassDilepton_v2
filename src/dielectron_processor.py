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
from src.corrections.evaluator import pu_evaluator, nnlops_weights, musf_evaluator, musf_iso_evaluator, musf_hlt_evaluator, lhe_weights, stxs_lookups, add_stxs_variations, add_pdf_variations, qgl_weights, qgl_weights_eager, qgl_weights_keepDim, btag_weights_json, btag_weights_jsonKeepDim, get_jetpuid_weights
import json
from coffea.lumi_tools import LumiMask
import pandas as pd # just for debugging
import dask_awkward as dak
import dask
from coffea.analysis_tools import Weights
import copy
from coffea.nanoevents.methods import vector
import time

import sys
sys.path.append("/depot/cms/users/kaur214/analysis_facility/CMSHighMassDilepton_v2/CMSHighMassDilepton_v2")

coffea_nanoevent = TypeVar('coffea_nanoevent')
ak_array = TypeVar('ak_array')

def getRapidity(obj):
    px = obj.pt * np.cos(obj.phi)
    py = obj.pt * np.sin(obj.phi)
    pz = obj.pt * np.sinh(obj.eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + obj.mass**2)
    rap = 0.5 * np.log((e + pz) / (e - pz))
    return rap


def _mass2_kernel(t, x, y, z):
    return t * t - x * x - y * y - z * z

def delta_r_V1(eta1, eta2, phi1, phi2):
    deta = abs(eta1 - eta2)
    dphi = abs(np.mod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi)
    dr = np.sqrt(deta**2 + dphi**2)
    return deta, dphi, dr


class EventProcessor(processor.ProcessorABC):
    def __init__(self, config: dict, test_mode=False, **kwargs):
        """
        TODO: replace all of these with self.config dict variable which is taken from a
        pre-made json file
        """
        self.config = config

        self.test_mode = test_mode
        dict_update = {
            "apply_LHE_Filter" : False,
            "do_pdf" : False,
            "apply_lhe" : False,
            "do_musf" : False,
            "applyTopPtWeight": False,
        }
        self.config.update(dict_update)


        # --- Evaluator
        extractor_instance = extractor()


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

        event_filter = ak.ones_like(events.event) # 1D boolean array to be used to filter out bad events
        dataset = events.metadata['dataset']
        NanoAODv = events.metadata['NanoAODv']
        is_mc = events.metadata['is_mc']
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

            LHE_filter = (((LHE_dilepton_mass > 100) & (LHE_dilepton_mass < 200)))[:,0]
            LHE_filter = ak.fill_none(LHE_filter, value=False)
            LHE_filter = (LHE_filter== False) # we want True to indicate that we want to keep the event

            event_filter = event_filter & LHE_filter
        # LHE cut original end -----------------------------------------------------------------------------

        # # Apply HLT to both Data and MC.
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
            # obtain PU reweighting b4 event filtering, and apply it after we finalize event_filter
            if ("22" in year) or ("23" in year) or ("24" in year):
                run_campaign = 3
            else:
                run_campaign = 2
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
            & ((abs(events.Electron.eta_raw) < 1.442)
            | (abs(events.Electron.eta_raw) > 1.566))
            & events.Electron[self.config[f"electron_id_v{NanoAODv}"]]
        )

        electrons = events.Electron[electron_selection]
       
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
        pass_leading_pt = ak.to_packed(pass_leading_pt[event_filter==True])

        
        if is_mc and do_pu_wgt:
            for variation in pu_wgts.keys():
                pu_wgts[variation] = ak.to_packed(pu_wgts[variation][event_filter==True])

        
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
        
        if is_mc:
            gjets = events.GenJet
            gleptons = events.GenPart[
                (
                    (abs(events.GenPart.pdgId) == 13)
                    | (abs(events.GenPart.pdgId) == 11)
                    | (abs(events.GenPart.pdgId) == 15)
                )
                & events.GenPart.hasFlags('isHardProcess')
            ]
            gl_pair = ak.cartesian({"jet": gjets, "lepton": gleptons}, axis=1, nested=True)
            dr_gl = gl_pair["jet"].delta_r(gl_pair["lepton"])
            isolated = ak.all((dr_gl > 0.3), axis=-1) # this also returns true if there's no leptons near the gjet

            # same order sorting algorithm as reco jet start -----------------
            gjets = ak.to_packed(gjets[isolated])
            # print(f"gjets.pt: {gjets.pt.compute()}")
            sorted_args = ak.argsort(gjets.pt, ascending=False)
            sorted_gjets = (gjets[sorted_args])
            gjets_sorted = ak.pad_none(sorted_gjets, target=2)
            # same order sorting algorithm as reco jet end -----------------

            # print(f"gjets_sorted: {gjets_sorted.compute()}")
            gjet1 = gjets_sorted[:,0]
            gjet2 = gjets_sorted[:,1]
            # original start -----------------------------------------------
            gjj = gjet1 + gjet2

            gjj_dEta = abs(gjet1.eta - gjet2.eta)
            gjj_dPhi = abs(gjet1.delta_phi(gjet2))
            gjj_dR = gjet1.delta_r(gjet2)



        self.prepare_jets(events, NanoAODv=NanoAODv)



        # ------------------------------------------------------------#
        # Apply JEC, get JEC and JER variations
        # ------------------------------------------------------------#
        year = self.config["year"]
        jets = events.Jet
        self.jec_factories_mc, self.jec_factories_data = get_jec_factories(
            self.config["jec_parameters"], 
            year
        )   
        
        do_jec = True # True       
        # do_jecunc = self.config["do_jecunc"]
        # do_jerunc = self.config["do_jerunc"]
        #testing 
        do_jecunc = False
        do_jerunc = False
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
                
            jets = factory.build(jets)

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
            weights.add("genWeight_normalization", weight=ak.ones_like(events.genWeight)/sumWeights)
            cross_section = self.config["cross_sections"][dataset]
            integrated_lumi = self.config["integrated_lumis"]
            weights.add("xsec", weight=ak.ones_like(events.genWeight)*cross_section)
            weights.add("lumi", weight=ak.ones_like(events.genWeight)*integrated_lumi)

            if do_pu_wgt:
                print("adding PU wgts!")
                weights.add("pu", weight=pu_wgts["nom"],weightUp=pu_wgts["up"],weightDown=pu_wgts["down"])


            # L1 prefiring weights
            #if self.config["do_l1prefiring_wgts"] and ("L1PreFiringWeight" in events.fields):
            #    L1_nom = events.L1PreFiringWeight.Nom
            #    L1_up = events.L1PreFiringWeight.Up
            #    L1_down = events.L1PreFiringWeight.Dn
            #    weights.add("l1prefiring", 
            #        weight=L1_nom,
            #        weightUp=L1_up,
            #        weightDown=L1_down
            #    )
        else: # data-> just add in ak ones for consistency
            weights.add("ones", weight=ak.values_astype(ak.ones_like(events.HLT.Ele30_WPTight_Gsf), "float32"))
        
          

        
        # ------------------------------------------------------------#
        # Calculate other event weights
        # ------------------------------------------------------------#
        pt_variations = (
            ["nominal"]
            # + jec_pars["jec_variations"]
            # + jec_pars["jer_variations"]
        )
        if is_mc and self.config["do_musf"]:
            #do mu SF start -------------------------------------
            print("doing musf!")
            muID =  musf_evaluator(
                self.config, self.config["year"], mu1, mu2
            )
            muIso = musf_iso_evaluator(
                self.config, self.config["year"], mu1, mu2
            )

            muHLT = musf_hlt_evaluator(
                self.config, self.config["year"], mu1, mu2
            )

            weights.add("muID",
                    weight=muID["nom"],
                    weightUp=muID["up"],
                    weightDown=muID["down"]
            )
            weights.add("muIso",
                    weight=muIso["nom"],
                    weightUp=muIso["up"],
                    weightDown=muIso["down"]
            )
            weights.add("muHLT",
                    weight=muHLT["nom"],
                    weightUp=muHLT["up"],
                    weightDown=muHLT["down"]
            )

            #do mu SF end -------------------------------------
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
        if is_mc and self.config["apply_lhe"]:
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

        #    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
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
            "e1_pt" : e1.pt,
            "e2_pt" : e2.pt,
            "e1_eta" : e1.eta,
            "e2_eta" : e2.eta,
            "e1_phi" : e1.phi,
            "e2_phi" : e2.phi,
            "dielectron_mass" : dielectron.mass,
            "dielectron_pt" :   dielectron.pt,
            "dielectron_eta" :  dielectron.eta,
        }
        if is_mc:
            mc_dict = {
                #"gjj_dR" : gjj_dR,
            }
            out_dict.update(mc_dict)

        # ------------------------------------------------------------#
        # Loop over JEC variations and fill jet variables
        # ------------------------------------------------------------#

        for variation in pt_variations:
            jet_loop_dict = self.jet_loop(
                events,
                jets,
                dielectron,
                e1,
                e2,
                variation,
                weights,
                NanoAODv = NanoAODv,
                do_jec = do_jec,
                do_jecunc = do_jecunc,
                do_jerunc = do_jerunc,
            )

            out_dict.update(jet_loop_dict)


        njets = out_dict["njets"]

        dataset = events.metadata["dataset"]

        print(f"weight statistics: {weights.weightStatistics.keys()}")
        wgt_nominal = weights.weight()
        if "wgt_nominal_btag_wgt" in out_dict.keys():
            # btag is seperated due to requiring information of other weights, and adding it directly to the weights varibles
            # screws up with the values
            print("adding btag wgts!")
            wgt_nominal = wgt_nominal*out_dict["wgt_nominal_btag_wgt"]

        # add in weights
        weight_dict = {"wgt_nominal_total" : wgt_nominal}
        for weight_type in list(weights.weightStatistics.keys()):
            wgt_name = "wgt_nominal_" + weight_type
            # print(f"wgt_name: {wgt_name}")
            weight_dict[wgt_name] = weights.partial_weight(include=[weight_type])
        out_dict.update(weight_dict)

        return out_dict

    def postprocess(self, accumulator):
        """
        Arbitrary postprocess function that's required to run the processor
        """
        pass

    def prepare_jets(self, events, NanoAODv=9):

        events["Jet", "pt_raw"] = (1 - events.Jet.rawFactor) * events.Jet.pt
        events["Jet", "mass_raw"] = (1 - events.Jet.rawFactor) * events.Jet.mass
        if NanoAODv >= 12:
            fixedGridRhoFastjetAll = events.Rho.fixedGridRhoFastjetAll
        else: # if v9
            fixedGridRhoFastjetAll = events.fixedGridRhoFastjetAll
        events["Jet", "PU_rho"] = ak.broadcast_arrays(fixedGridRhoFastjetAll, events.Jet.pt)[0]

        if events.metadata["is_mc"]:
            # pt_gen is used for JEC (one of the factory name map values)
            events["Jet", "pt_gen"] =  ak.values_astype(
                ak.fill_none(events.Jet.matched_gen.pt, value=0.0),
                "float32"
            )
            events["Jet", "has_matched_gen"] = events.Jet.genJetIdx > 0
        else:
            events["Jet", "has_matched_gen"] = False

        return

    def jet_loop(
        self,
        events,
        jets,
        dielectron,
        e1,
        e2,
        variation,
        weights,
        NanoAODv = 9,
        do_jec = False,
        do_jecunc = False,
        do_jerunc = False,
    ):
        is_mc = events.metadata["is_mc"]
        dataset = events.metadata["dataset"]
        year = self.config["year"]
        if (not is_mc) and variation != "nominal":
            return

        # Find jets that have selected muons within dR<0.4 from them

        matched_el_pt = jets.matched_electrons.pt_fsr if "pt_fsr" in jets.matched_electrons.fields else jets.matched_electrons.pt

        matched_el_id = jets.matched_electrons[self.config[f"electron_id_v{NanoAODv}"]]
        matched_el_pass = (
            (matched_el_pt > self.config["electron_pt_cut"])
            & matched_el_id
        )
        matched_el_pass = ak.sum(matched_el_pass, axis=2) > 0 # there's at least one matched electron that passes the electron selection
        clean = ~(ak.fill_none(matched_el_pass, value=False))

        # # ------------------------------------------------------------#
        # # Apply jetID and PUID
        # # ------------------------------------------------------------#

        pass_jet_id = jet_id(jets, self.config)

        print(f"jet loop NanoAODv: {NanoAODv}")
        if NanoAODv == 9 :
            pass_jet_puid = jet_puid(jets, self.config)
            # Jet PUID scale factors, which also takes pt < 50 into account within the function
            if is_mc:
                print("doing jet puid weights!")
                jet_puid_opt = self.config["jet_puid"]
                pt_name = "pt"
                puId = jets.puId
                jetpuid_weight = get_jetpuid_weights(
                    self.evaluator, year, jets, pt_name,
                    jet_puid_opt, pass_jet_puid
                )
                weights.add("jetpuid_wgt",
                        weight=jetpuid_weight,
                )
        else: # NanoAODv12 doesn't have Jet_PuID yet
            pass_jet_puid = ak.ones_like(pass_jet_id, dtype="bool")
        # ------------------------------------------------------------#
        # Select jets
        # ------------------------------------------------------------#
        # apply HEM Veto, written in "HEM effect in 2018" appendix K of the main long AN
        HEMVeto = ak.ones_like(clean) == 1 # 1D array saying True
        if year == "2018":
            HEMVeto_filter = (
                (jets.pt >= 20.0)
                & (jets.eta >= -3.0)
                & (jets.eta <= -1.3)
                & (jets.phi >= -1.57)
                & (jets.phi <= -0.87)
            )
            false_arr = ak.ones_like(HEMVeto) < 0
            HEMVeto = ak.where(HEMVeto_filter, false_arr, HEMVeto)
            # print(f"HEMVeto : {HEMVeto.compute()}")

        # original jet_selection-----------------------------------------------
        jet_selection = (
            pass_jet_id
            & clean
            & (jets.pt > self.config["jet_pt_cut"])
            & (abs(jets.eta) < self.config["jet_eta_cut"])
            & HEMVeto
        )
        # original jet_selection end ----------------------------------------------

        jets = ak.to_packed(jets[jet_selection])
        njets = ak.num(jets, axis=1)

        # ------------------------------------------------------------#
        # Fill jet-related variables
        # ------------------------------------------------------------#

        sorted_args = ak.argsort(jets.pt, ascending=False)
        sorted_jets = (jets[sorted_args])
        jets = sorted_jets
        paddedSorted_jets = ak.pad_none(sorted_jets, target=2)
        jet1 = paddedSorted_jets[:,0]
        jet2 = paddedSorted_jets[:,1]

        dijet = jet1+jet2

        jj_dEta = abs(jet1.eta - jet2.eta)
        jj_dPhi = abs(jet1.delta_phi(jet2))
        mmj1_dEta = abs(dielectron.eta - jet1.eta)
        mmj2_dEta = abs(dielectron.eta - jet2.eta)

        min_dEta_filter  = ak.fill_none((mmj1_dEta < mmj2_dEta), value=True)
        mmj_min_dEta = ak.where(
            min_dEta_filter,
            mmj1_dEta,
            mmj2_dEta,
        )
        # print(f"mmj_min_dEta: {mmj_min_dEta.compute()}")
        mmj1_dPhi = abs(dielectron.delta_phi(jet1))
        mmj2_dPhi = abs(dielectron.delta_phi(jet2))
        mmj1_dR = dielectron.delta_r(jet1)
        mmj2_dR = dielectron.delta_r(jet2)

        min_dPhi_filter = ak.fill_none((mmj1_dPhi < mmj2_dPhi), value=True)
        mmj_min_dPhi = ak.where(
            min_dPhi_filter,
            mmj1_dPhi,
            mmj2_dPhi,
        )

        jet_loop_out_dict = {
            "jet1_pt" : jet1.pt,
            "jet1_eta" : jet1.eta,
            "jet1_phi" : jet1.phi,
            "jet2_pt" : jet2.pt,
            "jet2_eta" : jet2.eta,
            "jet2_phi" : jet1.phi,
            "njets" : njets,

        }
        if is_mc:
            mc_dict = {
                #"jet1_pt_gen" : jet1.pt_gen,
                #"jet2_pt_gen" : jet2.pt_gen,
            }
            jet_loop_out_dict.update(mc_dict)

        # ------------------------------------------------------------#
        # Apply remaining cuts
        # ------------------------------------------------------------#


        #     # # --- Btag weights  start--- #
            do_btag_wgt = False # True
            if NanoAODv ==12:
                do_btag_wgt = False # temporary condition
            if do_btag_wgt:
                print("doing btag wgt!")
                btag_systs = self.config["btag_systs"] #if do_btag_syst else []
                btag_json =  correctionlib.CorrectionSet.from_file(self.config["btag_sf_json"],)
                # original start -------------------------------------
                # btag_wgt, btag_syst = btag_weights_json(
                #     self, btag_systs, jets, weights, bjet_sel_mask, btag_json
                # )
                # original end -------------------------------------

                # keep dims start -------------------------------------
                btag_wgt, btag_syst = btag_weights_jsonKeepDim(
                            self, btag_systs, jets, weights, bjet_sel_mask, btag_json
                )
                # keep dims end -------------------------------------
                # print(f"btag_wgt: {ak.to_numpy(btag_wgt.compute())}")
                # print(f"btag_syst['jes_up']: {ak.to_numpy(btag_syst['jes']['up'].compute())}")
                # print(f"btag_syst['jes_down']: {ak.to_numpy(btag_syst['jes']['down'].compute())}")
            # # --- Btag weights end --- #



        btagLoose_filter = (jets.btagDeepFlavB > self.config["btag_loose_wp"]) & (abs(jets.eta) < 2.5)
        nBtagLoose = ak.num(ak.to_packed(jets[btagLoose_filter]), axis=1)
        nBtagLoose = ak.fill_none(nBtagLoose, value=0)


        btagMedium_filter = (jets.btagDeepFlavB > self.config["btag_medium_wp"]) & (abs(jets.eta) < 2.5)
        nBtagMedium = ak.num(ak.to_packed(jets[btagMedium_filter]), axis=1)
        nBtagMedium = ak.fill_none(nBtagMedium, value=0)

        #print(f"nBtagLoose: {ak.to_numpy(nBtagLoose.compute())}")
        # print(f"njets: {ak.to_numpy(njets.compute())}")
        temp_out_dict = {
            "nBtagLoose": nBtagLoose,
            "nBtagMedium": nBtagMedium,
        }
        jet_loop_out_dict.update(temp_out_dict)
        if is_mc and do_btag_wgt:
            jet_loop_out_dict.update({
                "wgt_nominal_btag_wgt": btag_wgt
            })


        # --------------------------------------------------------------#
        # Fill outputs
        # --------------------------------------------------------------#

    #     variables.update({"wgt_nominal": weights.get_weight("nominal")})

    #     # All variables are affected by jet pT because of jet selections:
    #     # a jet may or may not be selected depending on pT variation.

    #     for key, val in variables.items():
    #         output.loc[:, pd.IndexSlice[key, variation]] = val

        return jet_loop_out_dict

