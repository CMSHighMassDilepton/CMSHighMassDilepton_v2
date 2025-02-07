import numpy as np
import awkward as ak
from coffea.lookup_tools import txt_converters, rochester_lookup
import pandas as pd
import correctionlib.schemav2 as cs


def apply_roccor(events, roccor_file_path: str, is_mc:bool, test_mode=False):
    rochester_data = txt_converters.convert_rochester_file(
        roccor_file_path, loaduncs=True, 
    )
    rochester = rochester_lookup.rochester_lookup(rochester_data)
    if is_mc:
        hasgen = ak.is_none(events.Muon.matched_gen.pt, axis=1) == False


        # testing start -----------------------------------------------------------
        resrng = cs.Correction(
            name="resrng",
            description="Deterministic smearing value generator",
            version=1,
            inputs=[
                cs.Variable(name="pt", type="real", description="Unsmeared jet pt"),
                cs.Variable(name="eta", type="real", description="Jet pseudorapdity"),
                cs.Variable(name="phi", type="real", description="Jet phi"),
                cs.Variable(name="charge", type="real", description="Muon charge"),
                # cs.Variable(name="event", type="real", description="Event number"),
            ],
            output=cs.Variable(name="rng", type="real"),
            data=cs.HashPRNG(
                nodetype="hashprng",
                inputs=["pt", "eta", "phi", "charge"],
                distribution="stdflat",
            )
        )
        mc_rand = resrng.to_evaluator().evaluate(
            events.Muon.pt,
            events.Muon.eta,
            events.Muon.phi,
            events.Muon.charge,
            # events.event,
        )
        # print(f"mc_rand: {ak.to_numpy(ak.flatten(mc_rand.compute()))}")
        # mc_rand = ak.unflatten(mc_rand, ak.num(events.Muon.pt, axis=1))
        
        # testing end --------------------------------------------------------------


        mc_kspread = rochester.kSpreadMC(
            events.Muon.charge,
            events.Muon.pt,
            events.Muon.eta,
            events.Muon.phi,
            events.Muon.matched_gen.pt,
        )

        mc_ksmear = rochester.kSmearMC(
            events.Muon.charge,
            events.Muon.pt,
            events.Muon.eta,
            events.Muon.phi,
            events.Muon.nTrackerLayers,
            mc_rand,
        )
        errspread = rochester.kSpreadMCerror(
            events.Muon.charge,
            events.Muon.pt,
            events.Muon.eta,
            events.Muon.phi,
            events.Muon.matched_gen.pt,
        )
        errsmear = rochester.kSmearMCerror(
            events.Muon.charge,
            events.Muon.pt,
            events.Muon.eta,
            events.Muon.phi,
            events.Muon.nTrackerLayers,
            mc_rand,
        )
        # hasgen_flat = np.array(ak.flatten(hasgen))
        corrections = ak.ones_like(events.Muon.pt)
        corrections = ak.where(hasgen, mc_kspread, corrections)
        corrections = ak.where((~hasgen), mc_ksmear, corrections)
        # corrections[hasgen] = mc_kspread
        # corrections[~hasgen] = mc_ksmear
        
        errors = ak.ones_like(events.Muon.pt)
        errors = ak.where(hasgen, errspread, errors)
        errors = ak.where((~hasgen), errsmear, errors)
        # errors[hasgen] = np.array(ak.flatten(errspread))
        # errors[~hasgen] = np.array(ak.flatten(errsmear))

        # corrections = ak.unflatten(corrections, ak.num(events.Muon.pt, axis=1))
        # errors = ak.unflatten(errors, ak.num(events.Muon.pt, axis=1))

    else:
        corrections = rochester.kScaleDT(
            events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi
        )
        errors = rochester.kScaleDTerror(
            events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi
        )
    # print(f"corrections: {ak.to_numpy(ak.flatten(corrections.compute()))}")
    events["Muon", "pt_roch"] = events.Muon.pt * corrections
    # uncommenting these lines below add really significant more compute time. not reccommended unless necessary
    # events["Muon", "pt_roch_up"] = events.Muon.pt_roch + events.Muon.pt * errors
    # events["Muon", "pt_roch_down"] = events.Muon.pt_roch - events.Muon.pt * errors
