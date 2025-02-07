# import pandas as pd
import numpy as np
import awkward as ak
from copy import deepcopy

class Weights(object):
    """
    NOTE: only supports keeping track of nominal weights for now. 
    No ..off or up or down due to issues with deepcopying ak arrays 
    """
    def __init__(self, ones):
        """
        ones -> 1D array representing the eventwise of weights.
        We assume that the event filtering has been done
        """
        self.weights = {
            "nominal" : ones
        }
        self.wgts = {}
        self.variations = []

    def add_weight(self, name, wgt=None, how="nom"):
        if (wgt is None) and ("dummy" not in how):
            return
        # print(f" add_weight name: {name}") #btag_wgt_lfstats1

        # print(f" add_weight wgt: \n {wgt}") #  {'up': pd df, 'down': pd df}
        # print(f"weights add_weight how: {how}")
        if how == "nom":
            # add only nominal weight
            self.add_nom_weight(name, wgt)
            
        elif how == "all":
            # add weight with up and down variations
            nom = wgt["nom"]
            up = wgt["up"]
            down = wgt["down"]
            # print(f"add_weight all how nom: \n {nom}")
            self.add_weight_with_variations(name, nom, up, down)
        elif how == "only_vars":
            # add only variations
            up = wgt["up"]
            down = wgt["down"]
            self.add_only_variations(name, up, down)

        # print(f"add_weight sself.wgts : \n {self.wgts}")
    
    def add_nom_weight(self, name, wgt):
        w_names = list(self.weights.keys()) # get a static list, not a dict_key pointer
        # print(f'w_names b4: {w_names}')

        
        self.weights[f"{name}_off"] = self.weights["nominal"]
        # print(f'self.weights {name}_off \n : {ak.to_numpy(self.weights[f"{name}_off"])}')
        # print(f'w_names after: {w_names}')
        for w_name in w_names:
            # print(f'w_name: {w_name}')
            # print(f'self.weights {w_name} \n : {ak.to_numpy(self.weights[w_name])}')
            self.weights[w_name] = self.weights[w_name]*wgt

        # print(f"wgt \n : {ak.to_numpy(wgt)}")
        self.variations.append(name)
        self.wgts[name] = wgt 

    

    def add_weight_with_variations(self, name, nom_wgt, up, down):
        w_names = list(self.weights.keys()) # get a static list, not a dict_key pointer
        self.wgts[name] = nom_wgt
        prev_nom = self.weights["nominal"]
        self.weights[f"{name}_off"] = prev_nom
        self.weights[f"{name}_up"] = ak.values_astype(prev_nom * up, "float64")
        self.weights[f"{name}_down"] = ak.values_astype(prev_nom * down, "float64")
        for w_name in w_names:
            self.weights[w_name] = self.weights[w_name]*nom_wgt
        self.variations.append(name)

    def add_only_variations(self, name, up, down):
        prev_nom = self.weights["nominal"]
        self.weights[f"{name}_up"] = ak.values_astype(prev_nom * up, "float64")
        self.weights[f"{name}_down"] = ak.values_astype(prev_nom * down, "float64")
        self.variations.append(name)


    def get_weight(self, name: str):
        if name in self.weights.keys():
            # print(f"{name} in keys")
            return self.weights[name]
        else:
            return ak.array([])

    """
    I don't use this code, it's just here from copperheadV1 in case I need to use it
    def effect_on_normalization(self, mask=np.array([])):
        if len(mask) == 0:
            mask = np.ones(self.df.shape[0], dtype=int)
        for var in self.variations:
            if f"{var}_off" not in self.df.columns:
                continue
            wgt_off = self.df[f"{var}_off"].dropna().to_numpy().sum()
            wgt_on = self.df["nominal"].dropna().to_numpy().sum()
            effect = (wgt_on - wgt_off) / wgt_on * 100
            if effect < 0:
                ef = round(-effect, 2)
                print(f"Enabling {var} decreases yield by {ef}%")
            else:
                ef = round(effect, 2)
                print(f"Enabling {var} increases yield by {ef}%")
    """

    def get_info(self):
        info_str = ""
        for w_name in self.weights.keys():
            info_str += f"{w_name} : \n {ak.to_numpy(self.weights[w_name])} \n"
        return info_str