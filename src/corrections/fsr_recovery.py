import numpy as np
import awkward as ak
from typing import TypeVar, Tuple
ak_array = TypeVar('ak_array')
coffea_nanoevent = TypeVar('coffea_nanoevent') 


def fsr_recovery(events: coffea_nanoevent) -> ak_array:

    # # testing better fsrPhotonsToRecover ---------------------------------------------------------------
    # mu_fsr_delta_r = events.Muon.delta_r(events.Muon.matched_fsrPhoton)
    # fsr_abs_eta = abs(events.Muon.matched_fsrPhoton.eta)
    # fsrPhotonsToRecover = (
    #     (events.Muon.fsrPhotonIdx >= 0) # pick rows and cols that have values (-1 means no photon found)
    #     & (events.Muon.matched_fsrPhoton.relIso03 < 1.8)
    #     & (events.Muon.matched_fsrPhoton.dROverEt2 < 0.012)
    #     & (events.Muon.matched_fsrPhoton.pt / events.Muon.pt < 0.4) # suppress Zgamma -> mumu contanmination
    #     & ( ((fsr_abs_eta > 0)&(fsr_abs_eta < 1.4442)) | ((fsr_abs_eta > 1.566)&(fsr_abs_eta < 2.5)) )
    #     & ((mu_fsr_delta_r > 0.0001) & (mu_fsr_delta_r < 0.5))
    # ) 
    # # # testing better fsrPhotonsToRecover end ---------------------------------------------------------------
    
    # original fsrPhotonsToRecover ---------------------------------------------------------------
    fsrPhotonsToRecover = (
        (events.Muon.fsrPhotonIdx >= 0) # pick rows and cols that have values (-1 means no photon found)
        & (events.Muon.matched_fsrPhoton.relIso03 < 1.8)
        & (events.Muon.matched_fsrPhoton.dROverEt2 < 0.012)
        & (events.Muon.matched_fsrPhoton.pt / events.Muon.pt < 0.4) # suppress Zgamma -> mumu contanmination
        & (abs(events.Muon.matched_fsrPhoton.eta) < 2.4)
    ) 
    # original fsrPhotonsToRecover end ---------------------------------------------------------------

    
    fsrPhotonsToRecover = ak.fill_none(fsrPhotonsToRecover, False) 
    
    # add mass and charge as otherwise you can't add two lorentzvectors
    events["FsrPhoton", "mass"] = 0 
    events["FsrPhoton", "charge"] = 0 
    # print(f"events.Muon.matched_fsrPhoton: {events.Muon.matched_fsrPhoton.compute()}")
    fsr_muons = events.Muon.matched_fsrPhoton + events.Muon # None means there weren't matched fsrphotons
    # fsr_muons =  ak.values_astype(events.Muon.matched_fsrPhoton,  "float64")  + ak.values_astype(events.Muon,  "float64") 
    
    fsr_iso = (events.Muon.pfRelIso04_all * events.Muon.pt - events.Muon.matched_fsrPhoton.pt) / fsr_muons.pt
   
    events["Muon", "pt_fsr"] = ak.where(fsrPhotonsToRecover, fsr_muons.pt, events.Muon.pt)
    events["Muon", "eta_fsr"] = ak.where(fsrPhotonsToRecover, fsr_muons.eta, events.Muon.eta)
    events["Muon", "phi_fsr"] = ak.where(fsrPhotonsToRecover, fsr_muons.phi, events.Muon.phi)
    events["Muon", "mass_fsr"] = ak.where(fsrPhotonsToRecover, fsr_muons.mass, events.Muon.mass)
    events["Muon", "iso_fsr"] = ak.where(fsrPhotonsToRecover, fsr_iso, events.Muon.pfRelIso04_all)


    
    # fsr_filter = (ak.sum(fsrPhotonsToRecover, axis=1) > 0)
    # fsr_events = events[fsr_filter]
    # argmax = ak.argmax(fsr_events.Muon.pt_fsr, axis=1)
    # argmax_not_leading = argmax != 0
    # argmax_not_leading_events = fsr_events[argmax_not_leading]
    return fsrPhotonsToRecover # return boolean filter for geofit


def fsr_recoveryV1(df):
    mask = (
        (df.Muon.fsrPhotonIdx >= 0)
        & (df.Muon.matched_fsrPhoton.relIso03 < 1.8)
        & (df.Muon.matched_fsrPhoton.dROverEt2 < 0.012)
        & (df.Muon.matched_fsrPhoton.pt / df.Muon.pt < 0.4)
        & (abs(df.Muon.matched_fsrPhoton.eta) < 2.4)
    )
    mask = ak.fill_none(mask, False)

    # px = ak.zeros_like(df.Muon.pt, dtype=np.float64)
    # py = ak.zeros_like(df.Muon.pt, dtype=np.float64)
    # pz = ak.zeros_like(df.Muon.pt, dtype=np.float64)
    # e = ak.zeros_like(df.Muon.pt, dtype=np.float64)
    px = ak.zeros_like(df.Muon.pt)
    py = ak.zeros_like(df.Muon.pt)
    pz = ak.zeros_like(df.Muon.pt)
    e = ak.zeros_like(df.Muon.pt)

    # print(f"fsr recovery py: {py.compute()}")
    # print(f"fsr recovery px: {px.compute()}")
    fsr = {
        "pt": df.Muon.matched_fsrPhoton.pt,
        "eta": df.Muon.matched_fsrPhoton.eta,
        "phi": df.Muon.matched_fsrPhoton.phi,
        "mass": 0.0,
    }

    for obj in [df.Muon, fsr]:
        px_ = obj["pt"] * np.cos(obj["phi"])
        py_ = obj["pt"] * np.sin(obj["phi"])
        pz_ = obj["pt"] * np.sinh(obj["eta"])
        e_ = np.sqrt(px_**2 + py_**2 + pz_**2 + obj["mass"] ** 2)

        px = px + px_
        py = py + py_
        pz = pz + pz_
        e = e + e_

    pt = np.sqrt(px**2 + py**2)
    # print(f"type(pt): {(pt.type)}")
    # print(f"total nmuons applied with fsrPhotons: {ak.sum(mask,axis=None)}")
    eta = np.arcsinh(pz / pt)
    phi = np.arctan2(py, px)
    # print(f"fsr recovery py: {py.compute()}")
    # print(f"fsr recovery px: {px.compute()}")
    # print(f"fsr recovery phi: {phi.compute()}")
    mass = np.sqrt(e**2 - px**2 - py**2 - pz**2)
    # print(f"type(eta): {(eta.type)}")
    # print(f"type(phi): {(phi.type)}")
    # print(f"type(mass): {(mass.type)}")
    iso = (df.Muon.pfRelIso04_all * df.Muon.pt - df.Muon.matched_fsrPhoton.pt) / pt

    df["Muon", "pt_fsr"] = ak.where(mask, pt, df.Muon.pt)
    df["Muon", "eta_fsr"] = ak.where(mask, eta, df.Muon.eta)
    df["Muon", "phi_fsr"] = ak.where(mask, phi, df.Muon.phi)
    df["Muon", "mass_fsr"] = ak.where(mask, mass, df.Muon.mass)
    # df["Muon", "mass_fsr"] = df.Muon.mass
    df["Muon", "iso_fsr"] = ak.where(mask, iso, df.Muon.pfRelIso04_all)
    fsr_event_mask = ak.sum(mask, axis=1) > 0
    # print(f"fsr_event_mask cases being true : {ak.sum(fsr_event_mask).compute()}")
    # print(f"df.Muon.phi_fsr : {df.Muon.phi_fsr.compute()}")
    # print(f"df.Muon.phi_fsr is none sum: {ak.sum(ak.is_none(df.Muon.phi_fsr)).compute()}")
    # print(f"df.Muon.iso_fsr is none sum: {ak.sum(ak.is_none(df.Muon.iso_fsr)).compute()}")
    
    # print(f"df[mask].Muon.pt_fsr: {df[fsr_event_mask].Muon.pt_fsr}")
    # print(f"df[mask].Muon.pt: {df[fsr_event_mask].Muon.pt}")
    # print(f"fsr[pt][mask]: \n {fsr['pt'][fsr_event_mask]}")
    return mask



    