"""
All classes must has "Custom" in their names to ensure only those are read by the category
wrapper
"""

import awkward as ak


# class CustomCat_VBF:
#     name = "Cat_VBF"
#     def __init__(self):
#         pass
#     def filterCategory(events): # apparently self is not needed
#         production_cat_cut = ak.fill_none(events.vbf_cut, value=False) # this require jj_mass > 400 AND jj_dEta > 2.5
#         btag_cut =(events.nBtagLoose >= 2) | (events.nBtagMedium >= 1)
#         vbfCat_selection = (
#             production_cat_cut  
#             & ~btag_cut # btag cut is for VH and ttH categories
#         )
#         return vbfCat_selection

# class CustomCat_ggH:
#     name = "Cat_ggH"
#     def __init__(self):
#         pass
#     def filterCategory(events): # apparently self is not needed
#         vbf_cut = ak.fill_none(events.vbf_cut, value=False) # this require jj_mass > 400 AND jj_dEta > 2.5
#         production_cat_cut = ~vbf_cut
#         btag_cut =(events.nBtagLoose >= 2) | (events.nBtagMedium >= 1)
#         ggHCat_selection = (
#             production_cat_cut  
#             & ~btag_cut # btag cut is for VH and ttH categories
#         )
#         return ggHCat_selection


class CustomInvBtagCut:
    name = "InvBtagCut"
    def __init__(self):
        pass
    def filterCategory(events): # apparently self is not needed
        btag_cut = (events.nBtagLoose >= 2) | (events.nBtagMedium >= 1)
        btag_cut = ak.fill_none(btag_cut, value=False)
        return ~btag_cut

class CustomVbfCut: # specified in line 827 of AN-19-124
    name = "VbfCut"
    def __init__(self):
        pass
    def filterCategory(events): # apparently self is not needed
        # vbf_cut = ak.fill_none(events.vbf_cut, value=False) # this require jj_mass > 400 AND jj_dEta > 2.5
        vbf_cut = (events.jj_mass > 400) & (events.jj_dEta > 2.5) 
        vbf_cut = ak.fill_none(vbf_cut, value=False)
        return vbf_cut

class CustomJet1PtCut: # specified in line 827 of AN-19-124
    name = "Jet1PtCut"
    def __init__(self):
        pass
    def filterCategory(events): # apparently self is not needed
        jet1_ptCut = ak.fill_none(events.jet1_pt > 35, value=False) 
        return jet1_ptCut

class CustomInvVbfCut: # specified in line 830 of AN-19-124
    name = "InvVbfCut"
    def __init__(self):
        pass
    def filterCategory(events): # apparently self is not needed
        # vbf_cut = ak.fill_none(events.vbf_cut, value=False) # this require jj_mass > 400 AND jj_dEta > 2.5
        vbf_cut = (events.jj_mass > 400) & (events.jj_dEta > 2.5) 
        vbf_cut = ak.fill_none(vbf_cut, value=False)
        return ~vbf_cut