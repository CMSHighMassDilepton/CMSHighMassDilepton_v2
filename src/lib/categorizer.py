import dask_awkward as dak
import awkward as ak
import numpy as np

class CategorizerBase():
    def __init__(self):
        pass
    def categorize(stage1_data):
        pass

class CategorizerCutSelect(CategorizerBase):
    def categorize(stage1_data):
        cat_dict = {}
        btag_cut =(stage1_data.nBtagLoose >= 2) | (stage1_data.nBtagMedium >= 1)
        vbf_cut = stage1_data.vbf_cut
        category_selection = (
            # ~stage1_data.vbf_cut & # we're interested in ggH category
            ~btag_cut # btag cut is for VH and ttH categories
        )
        cat_dict["VBF"] = category_selection & vbf_cut
        cat_dict["ggH"] = category_selection & (~vbf_cut)

        # turn bool arrays to filtered stage1_data
        for key, val in cat_dict:
            cat_dict[key] = stage1_data[val]
        return cat_dict

class CategorizerMVA(CategorizerBase):
    def __init__(self, MVA, scorebin_edges):
        self.MVA = MVA
        self.edges = scorebin_edges
    def runMVA(self):
    def categorize(stage1_data):
        input_data = stage1_data
        scores = self.MVA(input_data)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    year = str(2018)
    load_path = "/work/users/yun79/stage1_output/copperheadV2/stage2_test"
    full_load_path = load_path + f"/{yeawr}/f1_0/ggh_powheg/*/*.parquet"
    stage1_data = dak.from_parquet(full_load_path)
    # convert to pd df for now