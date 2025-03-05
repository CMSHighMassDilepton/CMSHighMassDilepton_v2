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


def etaFrame_variables(
        mu1: coffea_nanoevent,
        mu2: coffea_nanoevent
    ) -> Tuple[ak_array]: 
    """
    Obtain eta frame cos(theta) and phi as specified in:
    https://link.springer.com/article/10.1140/epjc/s10052-011-1600-y and
    This Eta frame values supposedly plays a similar role to CS frame in terms of physics
    sensitivity, but with better resolution. Not nocessarily believe this claim 
    however.
    """
    # divide muons in terms of negative and positive charges instead of leading pT
    mu_neg = ak.where((mu1.charge<0), mu1,mu2)
    mu_pos = ak.where((mu1.charge>0), mu1,mu2)
    dphi = abs(mu_neg.delta_phi(mu_pos))
    theta_eta = np.arccos(np.tanh((mu_neg.eta - mu_pos.eta) / 2))
    phi_eta = np.tan((np.pi - np.abs(dphi)) / 2) * np.sin(theta_eta)
    return np.cos(theta_eta), phi_eta

def cs_variables(
        mu1: coffea_nanoevent,
        mu2: coffea_nanoevent
    ) -> Tuple[ak_array]: 
    """
    return cos(theta) and phi in collins-soper frame
    """
    dimuon = mu1 + mu2
    cos_theta_cs = getCosThetaCS(mu1, mu2, dimuon)
    phi_cs = getPhiCS(mu1, mu2, dimuon)
    return cos_theta_cs, phi_cs

def getCosThetaCS(
    mu1: coffea_nanoevent,
    mu2: coffea_nanoevent,
    dimuon: coffea_nanoevent,
    ) -> ak_array :
    """
    return cos(theta) in collins-soper frame
    the formula for cos(theta) is given in Eqn 1. of https://www.ciemat.es/portal.do?TR=A&IDR=1&identificador=813
    """
    dimuon_pt = dimuon.pt
    dimuon_mass = dimuon.mass
    nominator = 2*(mu1.pz*mu2.energy - mu2.pz*mu1.energy)
    demoninator = dimuon_mass * (dimuon_mass**2 + dimuon_pt**2)**(0.5)
    cos_theta_cs = -(nominator/demoninator) # add negative sign to match the sign on pisa implementation at https://github.com/green-cabbage/copperhead_fork2/blob/Run3/python/math_tools.py#L152-L223
    return cos_theta_cs

def getPhiCS(
    mu1: coffea_nanoevent,
    mu2: coffea_nanoevent,
    dimuon: coffea_nanoevent,
    ) -> ak_array :
    """
    return phi in collins-soper frame
    the formula for phi is given in Eqn F.8 of https://people.na.infn.it/~elly/TesiAtlas/SpinCP/TestIpotesi/CollinSoperDefinition.pdf
    the implementation is heavily inspired from https://github.com/JanFSchulte/SUSYBSMAnalysis-Zprime2muAnalysis/blob/mini-AOD-2018/src/AsymFunctions.C#L1549-L1603
    """
    mu_neg = ak.where((mu1.charge<0), mu1,mu2)
    mu_pos = ak.where((mu1.charge>0), mu1,mu2)
    dimuon_pz = dimuon.pz
    dimuon_pt = dimuon.pt
    dimuon_mass = dimuon.mass
    beam_vec_z = ak.where((dimuon_pz>0), ak.ones_like(dimuon_pz), -ak.ones_like(dimuon_pz))
    # intialize beam vector as threevector to do cross product
    # beam_vec =  ak.zip(
    #     {
    #         "x": ak.zeros_like(dimuon_pz),
    #         "y": ak.zeros_like(dimuon_pz),
    #         "z": beam_vec_z,
    #     },
    #     with_name="ThreeVector",
    #     behavior=vector.behavior,
    # )
    # print(f"vector.__file__: {vector.__file__}")
    beam_vec =  ak.zip(
        {
            "x": ak.zeros_like(dimuon_pz),
            "y": ak.zeros_like(dimuon_pz),
            "z": beam_vec_z,
        },
        with_name="Momentum3D",
        behavior=vector.behavior
    )
    # apply cross product. note x,y,z of dimuon refers to its momentum, NOT its location
    # mu.px == mu.x, mu.py == mu.y and so on
    dimuon3D_vec = ak.zip({"x":dimuon.x, "y":dimuon.y, "z":dimuon.z}, with_name="Momentum3D", behavior=vector.behavior)
    R_T = beam_vec.cross(dimuon3D_vec) # direct cross product with dimuon doesn't work bc it's a 5D vector with x,y,z,t and charge
    
    R_T = R_T.unit() # make it a unit vector
    Q_T = dimuon
    Q_coeff = ( ((dimuon_mass*dimuon_mass + (dimuon_pt*dimuon_pt)))**(0.5) )/dimuon_mass
    delta_T_dot_R_T = (mu_neg.px-mu_pos.px)*R_T.x + (mu_neg.py-mu_pos.py)*R_T.y 
    delta_R_term = delta_T_dot_R_T
    delta_R_term = -delta_R_term # add negative sign to match the sign on pisa implementation at https://github.com/green-cabbage/copperhead_fork2/blob/Run3/python/math_tools.py#L152-L223
    delta_T_dot_Q_T = (mu_neg.px-mu_pos.px)*Q_T.px + (mu_neg.py-mu_pos.py)*Q_T.py
    delta_T_dot_Q_T = -delta_T_dot_Q_T # add negative sign to match the sign on pisa implementation at https://github.com/green-cabbage/copperhead_fork2/blob/Run3/python/math_tools.py#L152-L223
    delta_Q_term = delta_T_dot_Q_T
    delta_Q_term = delta_Q_term / dimuon_pt # normalize since Q_T should techincally be a unit vector
    phi_cs = np.arctan2(Q_coeff*delta_R_term, delta_Q_term)
    return phi_cs
    

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
            "do_roccor" : True,# True
            "do_pdf" : False,
            "do_musf" : True,
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

        for HLT_str in self.config["mu_hlt"]:
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
        events["Muon", "pt_raw"] = ak.ones_like(events.Muon.pt) * events.Muon.pt
        events["Muon", "eta_raw"] = ak.ones_like(events.Muon.eta) * events.Muon.eta
        events["Muon", "phi_raw"] = ak.ones_like(events.Muon.phi) * events.Muon.phi
        
        events["Muon", "Iso_raw"] = ak.ones_like(events.Muon.pfRelIso04_all) * events.Muon.pfRelIso04_all
        events["Muon", "tunepRelPt"] = ak.ones_like(events.Muon.tunepRelPt) * events.Muon.tunepRelPt 
        events["Muon", "tunePpt"]    = ak.ones_like(events.Muon.pt) * events.Muon.tunepRelPt * events.Muon.pt

        

        # --------------------------------------------------------#
        # Select muons that pass pT, eta, isolation cuts,
        # muon ID and quality flags
        # Select events with 2 good muons, no electrons,
        # passing quality cuts and at least one good PV
        # --------------------------------------------------------#

        # Apply event quality flags
        evnt_qual_flg_selection = ak.ones_like(event_filter)
        for evt_qual_flg in self.config["event_flags"]:
            evnt_qual_flg_selection = evnt_qual_flg_selection & events.Flag[evt_qual_flg]

        
        # muon_id = "mediumId" if "medium" in self.config["muon_id"] else "looseId"
        # print(f"copperhead2 EventProcessor muon_id: {muon_id}")
        # original muon selection ------------------------------------------------
        muon_selection = (
            (events.Muon.pt_raw >= self.config["muon_pt_cut"])
            & (abs(events.Muon.eta_raw) < self.config["muon_eta_cut"])
            & (events.Muon.Iso_raw < self.config["muon_iso_cut"])
            & events.Muon[self.config["muon_id"]]
            #& (abs(events.Muon.dxy) < self.config["muon_dxy"])
            #& (abs(events.Muon.dz) < self.config["muon_dz"])
            #& (
            #        (events.Muon.ptErr / events.Muon.pt_raw)
            #        < self.config["muon_ptErr/pt"]
            #    )
        )
        # original muon selection end ------------------------------------------------

        if self.config["do_roccor"]:
            print("doing rochester!")
            # print(f"df.Muon.pt b4 roccor: {events.Muon.pt.compute()}")
            apply_roccor(events, self.config["roccor_file"], is_mc)
            events["Muon", "pt"] = events.Muon.pt_roch

        muons = events.Muon[muon_selection]
        
        # count muons that pass the muon selection
        nmuons = ak.num(muons, axis=1)
        # Find opposite-sign muons
        mm_charge = ak.prod(muons.charge, axis=1)
        
        electron_id = self.config[f"electron_id_v{NanoAODv}"]
        # Veto events with good quality electrons; 
        electron_selection = (
            (events.Electron.pt > self.config["electron_veto_pt_cut"])
            & (abs(events.Electron.eta) < self.config["electron_eta_cut"])
            & events.Electron[electron_id]
        )
        
        electron_veto = (ak.num(events.Electron[electron_selection], axis=1) == 0) 

        
        event_filter = (
                event_filter
                & lumi_mask
                & (evnt_qual_flg_selection > 0)
                & (nmuons == 2)
                & (mm_charge == -1)
                & electron_veto 
                & (events.PV.npvsGood > 0) # number of good primary vertex cut

        )
        # --------------------------------------------------------#
        # Select events with muons passing leading pT cut
        # --------------------------------------------------------#

        muons_padded = ak.pad_none(muons, target=2)
        sorted_args = ak.argsort(muons_padded.pt, ascending=False) # leadinig pt is ordered by pt
        muons_sorted = (muons_padded[sorted_args])
        mu1 = muons_sorted[:,0]
        pass_leading_pt = mu1.pt_raw > self.config["muon_leading_pt"]
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
        muons = muons[event_filter==True]
        nmuons = ak.to_packed(nmuons[event_filter==True])
        pass_leading_pt = ak.to_packed(pass_leading_pt[event_filter==True])

        
        if is_mc and do_pu_wgt:
            for variation in pu_wgts.keys():
                pu_wgts[variation] = ak.to_packed(pu_wgts[variation][event_filter==True])
                
        

        
        # --------------------------------------------------------#
        # Fill dimuon and muon variables
        # --------------------------------------------------------#

        # ---------------------------------------------------------
        # TODO: find out why we don't filter out bad events right now via
        # even_selection column, since fill muon is computationally exp
        # Last time I checked there was some errors on LHE correction shape mismatch
        # ---------------------------------------------------------

        muons_padded = ak.pad_none(muons, target=2)
        sorted_args = ak.argsort(muons_padded.pt, ascending=False)
        muons_sorted = (muons_padded[sorted_args])
        mu1 = muons_sorted[:,0]
        mu2 = muons_sorted[:,1]
        
        dimuon_dR = mu1.delta_r(mu2)
        dimuon_dEta = abs(mu1.eta - mu2.eta)
        dimuon_dPhi = abs(mu1.delta_phi(mu2))
        dimuon = mu1+mu2
        
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
            weights.add("ones", weight=ak.values_astype(ak.ones_like(events.HLT.IsoMu24), "float32"))
        
          

        
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
        if is_mc:
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
                

            ## --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
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
        # Fill Muon variables and gjet variables
        # ------------------------------------------------------------#
        out_dict = {
            "event" : events.event,
            "mu1_pt" : mu1.pt,
            "mu2_pt" : mu2.pt,
            "mu1_eta" : mu1.eta,
            "mu2_eta" : mu2.eta,
            "mu1_phi" : mu1.phi,
            "mu2_phi" : mu2.phi,
            "dimuon_mass" : dimuon.mass,
            "dimuon_pt" :   dimuon.pt,
            "dimuon_eta" :  dimuon.eta,
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
                dimuon,
                mu1,
                mu2,
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
        dimuon,
        mu1,
        mu2,
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

        # matched_mu_pt = jets.matched_muons.pt_fsr
        matched_mu_pt = jets.matched_muons.pt_fsr if "pt_fsr" in jets.matched_muons.fields else jets.matched_muons.pt
        matched_mu_iso = jets.matched_muons.pfRelIso04_all

        matched_mu_id = jets.matched_muons[self.config["muon_id"]]
        matched_mu_pass = (
            (matched_mu_pt > self.config["muon_pt_cut"])
            & (matched_mu_iso < self.config["muon_iso_cut"])
            & matched_mu_id
        )
        matched_mu_pass = ak.sum(matched_mu_pass, axis=2) > 0 # there's at least one matched mu that passes the muon selection
        clean = ~(ak.fill_none(matched_mu_pass, value=False))
        

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
        mmj1_dEta = abs(dimuon.eta - jet1.eta)
        mmj2_dEta = abs(dimuon.eta - jet2.eta)
        
        min_dEta_filter  = ak.fill_none((mmj1_dEta < mmj2_dEta), value=True)
        mmj_min_dEta = ak.where(
            min_dEta_filter,
            mmj1_dEta,
            mmj2_dEta,
        )
        # print(f"mmj_min_dEta: {mmj_min_dEta.compute()}")
        mmj1_dPhi = abs(dimuon.delta_phi(jet1))
        mmj2_dPhi = abs(dimuon.delta_phi(jet2))
        mmj1_dR = dimuon.delta_r(jet1)
        mmj2_dR = dimuon.delta_r(jet2)
        
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
            "jet2_phi" : jet2.phi,
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
    
