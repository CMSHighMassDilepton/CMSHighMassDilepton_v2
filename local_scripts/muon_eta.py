import numpy as np
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import glob
import math
from itertools import repeat
import mplhep as hep
import time
import matplotlib.pyplot as plt
import copy
import random as rand
import pandas as pd
from functools import reduce

from ROOT import TCanvas, gStyle, TH1F,TH2F, TLegend, TFile, RooRealVar, RooCBShape, gROOT, RooFit, RooAddPdf, RooDataHist, RooDataSet
from ROOT import RooArgList

from array import array
import sys
import argparse

from dask_gateway import Gateway
gateway = Gateway(
    "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
    proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
)
cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
client = gateway.connect(cluster_info.name).get_client()
print("Gateway Client created")

parser = argparse.ArgumentParser()


load_fields = [
        "dimuon_mass",
        "mu1_pt",
        "mu1_eta",
        "wgt_nominal_total",
    ]


paths = "/depot/cms/users/kaur214/analysis_facility/outputs/stage1_output/2018/f1_0/dy_M-50/*/*parquet"

paths_data = "/depot/cms/users/kaur214/analysis_facility/outputs/stage1_output/2018/f1_0/data_A/*/*parquet"

sig_files = glob.glob(paths)
df_temp = dd.read_parquet(sig_files)

data_files = glob.glob(paths_data)
df_data_temp = dd.read_parquet(data_files)


df_dy   = df_temp[load_fields]

df_data = df_data_temp[load_fields]


print("computation complete")

df_dy   = df_dy[(df_dy["dimuon_mass"] > 60.) & (df_dy["dimuon_mass"] <120.)]
df_data   = df_data[(df_data["dimuon_mass"] > 60.) & (df_data["dimuon_mass"] <120.)]


massBinningMuMu = (
    [j for j in range(-3, 3, 30)]
    + [3]
)


print("starting .. ")

dy_mass = df_dy["mu1_eta"].compute().values
data_mass =  df_data["mu1_eta"].compute().values 

wgt_dy = df_dy["wgt_nominal_total"].compute().values
wgt_data = df_data["wgt_nominal_total"].compute().values

print("done complete")

h_dy = TH1F("h_dy", "h_dy", len(massBinningMuMu)-1, array('d', massBinningMuMu))
h_data = TH1F("h_data", "h_data", len(massBinningMuMu)-1, array('d', massBinningMuMu))

for i in range(len(dy_mass)):
    h_dy.Fill(dy_mass[i], wgt_dy[i])

for i in range(len(data_mass)):
    h_data.Fill(data_mass[i], wgt_data[i])


file2 = TFile("new_dy_2018A_eta.root","RECREATE")
file2.cd()
h_dy.Write()
h_data.Write()
file2.Close()








