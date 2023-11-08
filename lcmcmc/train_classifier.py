import logging 
import os
import pandas as pd
import numpy as np
import time

from astropy.table import Table

from lcmcmc.utils import get_data_dir_path
from kndetect.utils import load_pcs
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from joblib import dump


# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)

pcs = load_pcs()

data_dir = get_data_dir_path()


def get_features_df(dataset):

    if dataset not in ['train', 'test']:
        raise ValueError("Dataset must be train or test.")

    data_head_path = f'/sps/lsst/users/bbiswas/data/kilonova_datasets/{dataset}_final_master_HEAD.FITS'
    data_phot_path = f'/sps/lsst/users/bbiswas/data/kilonova_datasets/{dataset}_final_master_PHOT.FITS'

    df_head = Table.read(data_head_path, format='fits').to_pandas()
    df_phot = Table.read(data_phot_path, format='fits').to_pandas()

    features_df = []

    max_file_num = 43
    if dataset=='train':
        max_file_num = 45

    for file_num in range(max_file_num):
        LOG.info(file_num)
        
        trained_features = pd.read_pickle(os.path.join(data_dir, dataset, f"{dataset}_{file_num}_data.pkl"))
        
        for index, row in trained_features.iterrows():

            current_obj_df = {}
            norm_fact = np.array([row['norm_factor']]*2000)

            features = np.concatenate((row['MCMC_samples_kn'], np.expand_dims(np.mean(row['data-likelihood']['data_value'], axis=2), 2)), axis=2).reshape((2000, 7))

            for band_num, band in enumerate(['g', 'r']):
                for coeff_num in range(3):
                    current_obj_df[f"{band}coeff{coeff_num}"]=features[:, band_num*3+coeff_num]

            current_obj_df["likelihood"] = features[:, 6]
            current_obj_df["g_norm"] = norm_fact[:, 0]
            current_obj_df["r_norm"] = norm_fact[:, 1]
            event_type = df_head[row['SNID']==df_head["SNID"]]["SNTYPE"].values[0]
            if event_type in [149, 150]:
                current_obj_df["type"] = 1
            else:
                current_obj_df["type"] = 0 

            current_obj_df['SNID'] = row['SNID']

            current_obj_df=pd.DataFrame(current_obj_df)
            

            features_df.append(current_obj_df)

    features_df = pd.concat(features_df, ignore_index=True)

    return features_df



train_features_df = get_features_df("train")

print(train_features_df.columns)

train_features_df = train_features_df.groupby("type").sample(100000)
test_features_df = get_features_df("test")

features_col_names = [] 

for band_num, band in enumerate(['g', 'r']):
    for coeff_num in range(3):
        features_col_names.append(f"{band}coeff{coeff_num}")
features_col_names.append("likelihood")
features_col_names.append("g_norm")
features_col_names.append("r_norm")

classifiers = ['XGBoost', 'RF']

for classifier in classifiers:

    if classifier == 'RF':
        clf = RandomForestClassifier(n_estimators=30, max_depth=42)
    if classifier == 'SVM':
        clf = SVC(probability=True)
    if classifier=='XGBoost':
        clf = XGBClassifier(n_estimators=30, max_depth=42)

    clf.fit(train_features_df[features_col_names], train_features_df["type"])

    LOG.info(f"Trained {classifier} classifier")

    train_probabilities = clf.predict_proba(train_features_df[features_col_names])
    train_kne_prob = train_probabilities.T[1]

    start = time.time() 
    test_probabilities = clf.predict_proba(test_features_df[features_col_names])
    test_kne_prob = test_probabilities.T[1]
    end = time.time()
    LOG.info(f"Time taken by {classifier}: {end-start}")

    dump(clf, os.path.join(get_data_dir_path(), f'{classifier}.joblib'))
    

   
    train_save_scores ={}
    train_save_scores["SNID"] = train_features_df["SNID"]
    train_save_scores["KNe_prob"] = train_kne_prob
    train_save_scores["SNTYPE"] = train_features_df["type"]

    save_scores=pd.DataFrame(train_save_scores)

    save_scores.to_pickle(os.path.join(get_data_dir_path(), f"train_predicted_scores_{classifier}.pkl"))

    test_save_scores ={}
    test_save_scores["SNID"] = test_features_df["SNID"]
    test_save_scores["KNe_prob"] = test_kne_prob
    test_save_scores["SNTYPE"] = test_features_df["type"]

    test_save_scores=pd.DataFrame(test_save_scores)

    test_save_scores.to_pickle(os.path.join(get_data_dir_path(), f"test_predicted_scores_{classifier}.pkl"))

