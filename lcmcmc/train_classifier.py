import logging 
import os
import pandas as pd
import numpy as np

from astropy.table import Table

from lcmcmc.utils import get_data_dir_path
from kndetect.utils import load_pcs
from sklearn.ensemble import RandomForestClassifier

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

train_features_df = train_features_df.groupby("type").sample(50000)
test_features_df = get_features_df("test")

rf = RandomForestClassifier(n_estimators=30, max_depth=42)

features_col_names = [] 

for band_num, band in enumerate(['g', 'r']):
    for coeff_num in range(3):
        features_col_names.append(f"{band}coeff{coeff_num}")
features_col_names.append("likelihood")
features_col_names.append("g_norm")
features_col_names.append("r_norm")

rf.fit(train_features_df[features_col_names], train_features_df["type"])

probabilities = rf.predict_proba(test_features_df[features_col_names])
kne_prob = probabilities.T[1]

save_scores ={}
save_scores["SNID"] = test_features_df["SNID"]
save_scores["KNe_prob"] = kne_prob
save_scores["SNTYPE"] = test_features_df["type"]

save_scores=pd.DataFrame(save_scores)

save_scores.to_pickle(os.path.join(get_data_dir_path(), "predicted_scores.pkl"))

