from typing import Tuple, List, Dict
import ROOT as rt
import numpy as np
import pickle
import awkward as ak
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt





# functions for MVA related stuff start --------------------------------------------
def prepare_features(events: ak.Record, training_features, variation="nominal", add_year=False):
    """
    This basically looks over the events and find the relevant features specified in training_features.
    Basically, training_features don't specify variation (ie nominal), and if there's variation specific 
    features in events, you return the variation spefific key
    """
    #global training_features
    if add_year:
        features = training_features + ["year"]
    else:
        features = training_features
    features_var = []
    #print(features)
    for trf in features:
        if f"{trf}_{variation}" in events.fields:
            features_var.append(f"{trf}_{variation}")
        elif trf in events.fields:
            features_var.append(trf)
        else:
            print(f"Variable {trf} not found in training dataframe!")
    return features_var

    
# -----------------------------------------------------------------------------------------------------------------
# ggH BDT 
# -----------------------------------------------------------------------------------------------------------------
def evaluate_bdt(df: ak.Record, variation, model, training_features: List[str], parameters) -> ak.Record :
    """
    
    """
    # print(f"sum df.h_peak: {ak.sum(df.h_peak)}")
    # overwrite dimuon mass for regions not in h_peak

    

    # temporary definition of event bc I don't have it, and we need it for 4-fold method to work
    if "event" not in df.fields:
        print("Events not in Fields, using np arange instead")
        df["event"] = np.arange(len(df.dimuon_pt))
    
    features = prepare_features(df,training_features, variation=variation, add_year=False)
    # features = training_features
    #model = f"{model}_{parameters['years'][0]}"
    # score_name = f"score_{model}_{variation}"
    score_name = "BDT_score" # this is evaluation BDT score
    year = parameters["year"]

    # df.loc[:, score_name] = 0
    score_total = np.zeros(len(df['dimuon_pt']))
    
    nfolds = 4
    
    for i in range(nfolds):
        # eval_folds are the list of test dataset chunks that each bdt is trained to evaluate
        eval_folds = [(i + f) % nfolds for f in [3]]
        eval_filter = (df.event % nfolds ) == (np.array(eval_folds) * ak.ones_like(df.event))
        # eval_filter = df.event.mod(nfolds).isin(eval_folds)
        print(f"eval_folds: {eval_folds}")
        # print(f"eval_filter: {eval_filter}")
        # scalers_path = f"{parameters['models_path']}/{model}/scalers_{model}_{i}.npy"
        scalers_path = f"{parameters['models_path']}/scalers_{model}_{year}_{i}.npy"
        print(f"scalers_path: {scalers_path}")
        scalers = np.load(scalers_path, allow_pickle=True)
        # model_path = f"{parameters['models_path']}/{model}/{model}_{i}.pkl"
        model_path = f"{parameters['models_path']}/{model}_{year}_{i}.pkl"

        bdt_model = pickle.load(open(model_path, "rb"))
        df_i = df[eval_filter]
        # print(f"df_i: {len(df_i)}")
        # print(len
        if len(df_i) == 0:
            continue
        # print(f"scalers: {scalers.shape}")
        # print(f"df_i: {df_i}")
        df_i_feat = df_i[features]
        df_i_feat = ak.concatenate([df_i_feat[field][:, np.newaxis] for field in df_i_feat.fields], axis=1)
        # print(f"df_i_feat: {df_i_feat.shape}")
        df_i_feat = ak.Array(df_i_feat)
        df_i_feat = (df_i_feat - scalers[0]) / scalers[1]
        if len(df_i_feat) > 0:
            print(f"model: {model}")
            prediction = np.array(
                bdt_model.predict_proba(df_i_feat)[:, 1]
            ).ravel()
            # print(f"prediction: {prediction}")
            score_total[eval_filter] = prediction

    df[score_name] = score_total

    # do the same for validation score
    score_name_val = "BDT_score_val"
    score_total_val = np.zeros(len(df['dimuon_pt']))

    for i in range(nfolds):
        # val_folds are the list of test dataset chunks that each bdt is trained to evaluate
        val_folds = [(i+f)%nfolds for f in [2]]
        # val_filter = df.event.mod(nfolds).isin(val_folds)
        val_filter = (df.event % nfolds ) == (np.array(val_folds) * ak.ones_like(df.event))
        print(f"val_folds: {val_folds}")
        # print(f"val_filter: {val_filter}")
        # scalers_path = f"{parameters['models_path']}/{model}/scalers_{model}_{i}.npy"
        scalers_path = f"{parameters['models_path']}/scalers_{model}_{year}_{i}.npy"
        scalers = np.load(scalers_path, allow_pickle=True)
        # model_path = f"{parameters['models_path']}/{model}/{model}_{i}.pkl"
        model_path = f"{parameters['models_path']}/{model}_{year}_{i}.pkl"

        bdt_model = pickle.load(open(model_path, "rb"))
        df_i = df[val_filter]
        # print(f"df_i: {len(df_i)}")
        # print(len
        if len(df_i) == 0:
            continue
        # print(f"scalers: {scalers.shape}")
        # print(f"df_i: {df_i}")
        df_i_feat = df_i[features]
        df_i_feat = ak.concatenate([df_i_feat[field][:, np.newaxis] for field in df_i_feat.fields], axis=1)
        # print(f"type df_i_feat: {type(df_i_feat)}")
        # print(f"df_i_feat: {df_i_feat.shape}")
        df_i_feat = ak.Array(df_i_feat)
        df_i_feat = (df_i_feat - scalers[0]) / scalers[1]
        if len(df_i_feat) > 0:
            print(f"model: {model}")
            prediction = np.array(
                bdt_model.predict_proba(df_i_feat)[:, 1]
            ).ravel()
            # print(f"prediction: {prediction}")
            score_total_val[val_filter] = prediction

    df[score_name_val] = score_total_val
    
    # do the same for training score
    score_name_train = "BDT_score_train"
    score_total_train = np.zeros(len(df['dimuon_pt']))

    for i in range(nfolds):
        train_folds = [(i+f)%nfolds for f in [0,1]]
        event = ak.to_dataframe(df.event) # convert to pd.df to apply mod() and isin() function
        train_filter = event.mod(nfolds).isin(train_folds).values.ravel()
        print(f"train_folds: {train_folds}")
        # print(f"train_filter: {train_filter}")
        # scalers_path = f"{parameters['models_path']}/{model}/scalers_{model}_{i}.npy"
        scalers_path = f"{parameters['models_path']}/scalers_{model}_{year}_{i}.npy"
        scalers = np.load(scalers_path, allow_pickle=True)
        # model_path = f"{parameters['models_path']}/{model}/{model}_{i}.pkl"
        model_path = f"{parameters['models_path']}/{model}_{year}_{i}.pkl"

        bdt_model = pickle.load(open(model_path, "rb"))
        df_i = df[train_filter]
        # print(f"df_i: {len(df_i)}")
        # print(len
        if len(df_i) == 0:
            continue
        # print(f"scalers: {scalers.shape}")
        # print(f"df_i: {df_i}")
        df_i_feat = df_i[features]
        df_i_feat = ak.concatenate([df_i_feat[field][:, np.newaxis] for field in df_i_feat.fields], axis=1)
        # print(f"type df_i_feat: {type(df_i_feat)}")
        # print(f"df_i_feat: {df_i_feat.shape}")
        df_i_feat = ak.Array(df_i_feat)
        df_i_feat = (df_i_feat - scalers[0]) / scalers[1]
        if len(df_i_feat) > 0:
            print(f"model: {model}")
            prediction = np.array(
                bdt_model.predict_proba(df_i_feat)[:, 1]
            ).ravel()
            # print(f"prediction: {prediction}")
            score_total_train[train_filter] = prediction

    df[score_name_train] = score_total_train
    
    return df

# -----------------------------------------------------------------------------------------------------------------
# VBF DNN 
# -----------------------------------------------------------------------------------------------------------------
class DnnVBF(nn.Module):
    """
    this class is a copy of "Net" class from https://github.com/green-cabbage/copperhead_fork2/blob/Run3/stage2/mva_models.py#L6
    """
    def __init__(self, input_shape):
        super(DnnVBF, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.tanh(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.tanh(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.tanh(x)
        x = self.dropout3(x)

        x = self.output(x)
        output = F.sigmoid(x)
        return output

def evaluate_dnn(df: ak.Record, variation: str, model: str, features: List[str], parameters: dict) -> ak.Record :
    """
    
    """
    print(f"sum df.h_peak: {ak.sum(df.h_peak)}")

    # # temporary fix to due to some bug according to Valerie
    # df['mmj_min_dEta'] = df["mmj2_dEta"]
    # df['mmj_min_dPhi'] = df["mmj2_dPhi"]
    
    # overwrite dimuon mass for regions not in h_peak
    not_h_peak = (df.h_peak ==0)
    df["dimuon_mass"] = ak.where(not_h_peak, 125.0,  df["dimuon_mass"]) # line 2056 of RERECO AN + 7.16
    

    # temporary definition of event bc I don't have it, and we need it for 4-fold method to work
    if "event" not in df.fields:
        df["event"] = np.arange(len(df.dimuon_pt))
    
    score_name = "DNN_score"

    score_total = np.zeros(len(df['dimuon_pt']))
    score_raw = np.zeros(len(df['dimuon_pt'])) # raw sigmoid output
    
    nfolds = 4

    for i in range(nfolds):
        # eval_folds are the list of test dataset chunks that each bdt is trained to evaluate
        eval_folds = [(i + f) % nfolds for f in [3]]
        eval_filter = (df.event % nfolds ) == (np.array(eval_folds) * ak.ones_like(df.event))
        scalers_path = f"{parameters['models_path']}/{model}/scalers_{model}_{i}.npy"
        print(f"scalers_path: {scalers_path}")
        scalers = np.load(scalers_path, allow_pickle=True)
        df_i = df[eval_filter]
        if len(df_i) == 0:
            continue
        print(f"scalers: {scalers.shape}")
        print(f"df_i: {df_i}")
        df_i_feat = df_i[features]
        df_i_feat = ak.concatenate([df_i_feat[field][:, np.newaxis] for field in df_i_feat.fields], axis=1)
        print(f"df_i_feat[:,0]: {df_i_feat[:,0]}")
        print(f'df_i.dimuon_cos_theta_cs: {df_i.dimuon_cos_theta_cs}')
        df_i_feat = ak.to_numpy(ak.Array(df_i_feat))
        df_i = (df_i_feat - scalers[0]) / scalers[1]
        print(f"df_i: {df_i.shape}")
        # df_i = torch.tensor(df_i.values).float()
        df_i = torch.from_numpy(df_i).float()

        dnn_model = DnnVBF(len(features))
        model_path = f"{parameters['models_path']}/{model}/{model}_{i}.pt"
        dnn_model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        dnn_model.eval()

        prediction = dnn_model(df_i).detach().numpy().flatten()
        print(f"prediction: {prediction.shape}")
        # prediction = (prediction - 1/2)*2 # temporary shift the sigmoid result to have range of a tanh
            
        score_total[eval_filter] = np.arctanh(prediction)
        score_raw[eval_filter] = prediction
    df[score_name] = score_total
    df[score_name+"_sigmoid"] = score_raw
    print(f"dnn score_total: {score_total}")
    print(f"dnn score_total max: {np.max(score_total)}")
    print(f"dnn score_total min: {np.min(score_total)}")
    return df


