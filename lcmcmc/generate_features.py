import logging
import sys 
import os
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from astropy.table import Table

from lcmcmc.preprocessing import add_object_band_index, preprocess_SNANA, extract_subsample
from lcmcmc.utils import get_data_dir_path
from lcmcmc.model import jd_model_pcs

from kndetect.utils import load_pcs

tfd = tfp.distributions

# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)

rng = jax.random.PRNGKey(0)

training_or_test = sys.argv[1]

if training_or_test not in ['train', 'test']:
    raise ValueError("First argument should be either train or test")

data_head_path = f'/sps/lsst/users/bbiswas/data/kilonova_datasets/{training_or_test}_final_master_HEAD.FITS'
data_phot_path = f'/sps/lsst/users/bbiswas/data/kilonova_datasets/{training_or_test}_final_master_PHOT.FITS'

df_head = Table.read(data_head_path, format='fits').to_pandas()
df_phot = Table.read(data_phot_path, format='fits').to_pandas()

pcs = load_pcs()

mu_kn = np.load(os.path.join(get_data_dir_path(), "mu_kn.npy"))
scale_kn = np.load(os.path.join(get_data_dir_path(), "scale_kn.npy"))

# mu_non_kn = np.load(os.path.join(get_data_dir_path(), "mu_non_kn.npy"))
# scale_non_kn = np.load(os.path.join(get_data_dir_path(), "scale_non_kn.npy"))

print(jax.devices())

def return_data_loglike(mcmc_sample, obs, sigma, pinned_jd):
    dists, _ = pinned_jd.distribution.sample_distributions(seed=rng, value=mcmc_sample)
    return {'data_value': dists.obs.log_prob(obs)}

@jax.jit
def run_mcmc_sampling(index, x_range, pcs, mu, scale, observed_sigma, observed_value, rng):
    jd = jd_model_pcs(index, x_range, pcs, mu, scale)
    pinned_jd = jd.experimental_pin(sigma=observed_sigma, obs=observed_value)
    
    # Run the mcmc
    run_mcmc = jax.jit(
        lambda seed: tfp.experimental.mcmc.windowed_adaptive_nuts(
            500, 
            pinned_jd, 
            n_chains=4, 
            seed=seed,
        )
    )

    rng, sample_rng = jax.random.split(rng, 2)
    mcmc_samples, sampler_stats = run_mcmc(sample_rng)

    data_likelihood = jax.vmap(jax.vmap(lambda x: return_data_loglike(x, observed_value, observed_sigma, pinned_jd)))(mcmc_samples)

    return mcmc_samples, sampler_stats, data_likelihood

pd.options.mode.chained_assignment = None 
mcmc_samples_kn_prior=[]
data_likelihood_list = []
snid_list=[]
sampler_stats_list_kn = []

norm_factor_list=[]
max_flux_date_list = []
compare_res = []

@jax.jit
def compute_snr(flux, fluxerr):
    return flux/fluxerr

for object_num, snid in enumerate(df_head["SNID"].values):
# for snid in kn_snids[:3]:  
    LOG.info(f"SNID {snid}")
    
    compare_dict = {}

    current_df = df_phot[df_phot["SNID"] == snid]

    current_df_head = df_head[df_head["SNID"]==snid]
    
    detection_points = current_df[(current_df["PHOTFLAG"] == 4096) | (current_df["PHOTFLAG"] == 999999)]

    max_snr_loc = jnp.argmax(compute_snr(detection_points["FLUXCAL"].values, detection_points["FLUXCALERR"].values))
    max_snr_date = detection_points["MJD"].values[max_snr_loc]
    
    current_df = current_df[(current_df["MJD"]>=(max_snr_date-10)) & (current_df["MJD"]<= (max_snr_date+20))]
    current_df = add_object_band_index(current_df, bands=[b'g', b'r'])

    normed_current_df = preprocess_SNANA(df_head=current_df_head, df_phot=current_df, bands=[b'g', b'r'], norm_band_index=None)

    band_index = normed_current_df["band_index"]

    # x_range = jnp.asarray(normed_current_df["time"])

    # observed_value = jnp.array(np.asarray(normed_current_df["flux"]), dtype=jnp.float32)
    # observed_sigma = jnp.array(np.asarray(normed_current_df["fluxerr"]), dtype=jnp.float32)

    # use KN prior 
     
    mcmc_samples, sampler_stats, data_likelihood = run_mcmc_sampling(
        index=band_index, 
        x_range=normed_current_df["time"], 
        pcs=pcs, 
        mu=mu_kn, 
        scale=scale_kn, 
        observed_sigma=normed_current_df["fluxerr"], 
        observed_value=normed_current_df["flux"],
        rng=rng,
    )

    sampler_stats_list_kn.append(sampler_stats)
    mcmc_samples_kn_prior.append(mcmc_samples[0])
    data_likelihood_list.append(data_likelihood)
    
    snid_list.append(snid)
    norm_factor_list.append(normed_current_df['norm_factor'][0])
    max_flux_date_list.append(normed_current_df['max_time'][0])

    if (object_num+1)%500 == 0: 


        mcmc_samples_df = pd.DataFrame(
            {
                'SNID': snid_list, 
                'MCMC_samples_kn': mcmc_samples_kn_prior, 
                'norm_factor': norm_factor_list , 
                'sampler_stats_kn': sampler_stats_list_kn, 
                'max_time':max_flux_date_list,
                'data-likelihood': data_likelihood_list
            }
        )
        save_path = os.path.join(get_data_dir_path(), training_or_test, f"{training_or_test}_{int(object_num/500)}_data.pkl")
        mcmc_samples_df.to_pickle(save_path)

        mcmc_samples_kn_prior=[]
        data_likelihood_list = []
        snid_list=[]
        sampler_stats_list_kn = []

        norm_factor_list=[]
        max_flux_date_list = []
        compare_res = []

mcmc_samples_df = pd.DataFrame(
    {
        'SNID': snid_list, 
        'MCMC_samples_kn': mcmc_samples_kn_prior, 
        'norm_factor': norm_factor_list , 
        'sampler_stats_kn': sampler_stats_list_kn, 
        'max_time':max_flux_date_list,
        'data-likelihood': data_likelihood_list
    }
)
save_path = os.path.join(get_data_dir_path(), training_or_test, f"{training_or_test}_{int(object_num/500)}_data.pkl")
mcmc_samples_df.to_pickle(save_path)