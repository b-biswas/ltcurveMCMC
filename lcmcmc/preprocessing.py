import numpy as np
import pandas as pd
import jax 
import jax.numpy as jnp

@jax.jit
def normalize_mjd(mjd, max_flux_time):
    return mjd - max_flux_time

@jax.jit
def normalize_flux(flux, max_flux):
    return flux/max_flux


def preprocess_SNANA(df_head, df_phot, bands=['g','r'], norm_band_index=None):
    """Convert from SNANA format to MCMC

    Parameters
    ----------
    df_head: pd.DataFrame
        headerfile
    df_phot: pd.DataFrame
        photometry file

    Returns
    -------
    mcmc_fotmat_df: pd.DataFrame
        datafarme in the format supported for MCMC sampling
    """
    new_object_dfs = []

    current_object_new_df = {}

    if norm_band_index in [0, 1]:
        for object_id in df_head['SNID']:

            object_df = df_phot[df_phot["SNID"] == object_id]
            r_band_data = object_df[object_df['FLT'] == bands[norm_band_index]]
            
            max_flux_loc = np.argmax(r_band_data["FLUXCAL"])
            max_flux = r_band_data['FLUXCAL'].iloc[max_flux_loc]
            max_flux_time = r_band_data['MJD'].iloc[max_flux_loc]
            
            new_time = object_df['MJD'] - max_flux_time
            new_flux = object_df['FLUXCAL'] / max_flux
            new_flux_err = object_df['FLUXCALERR'] / max_flux
                
            current_object_new_df = {}
            current_object_new_df['SNID'] = object_df['SNID']
            current_object_new_df['time'] = new_time
            current_object_new_df['flux'] = new_flux
            current_object_new_df['fluxerr'] = new_flux_err
            current_object_new_df['object_index'] = object_df['object_index']
            current_object_new_df['band_index'] = object_df['band_index']
            
            current_object_new_df['norm_factor'] = max_flux

            current_object_new_df = pd.DataFrame.from_dict(current_object_new_df)
            new_object_dfs.append(current_object_new_df)

    elif norm_band_index is None:

        new_time = []
        new_flux = []
        new_flux_err = []
        band_index = []
        max_flux_values = []
        object_index = []
        current_SNID= []
        max_flux_time_values=[]

        #print(object_df)
        for band in bands:
            band_df = df_phot[df_phot['FLT'] == band]

            if len(band_df)>0:
            
                max_flux_loc = np.argmax(band_df["FLUXCAL"])
                max_flux = band_df['FLUXCAL'].iloc[max_flux_loc]
                max_flux_time = band_df['MJD'].iloc[max_flux_loc]
                
                new_time.extend(normalize_mjd(band_df['MJD'].values, max_flux_time))
                new_flux.extend(normalize_flux(band_df['FLUXCAL'].values, max_flux))
                new_flux_err.extend(normalize_flux(band_df['FLUXCALERR'].values, max_flux))

                band_index.extend(band_df['band_index'].values)

                current_SNID.extend(band_df["SNID"].values)

                max_flux_values.append(max_flux)
                max_flux_time_values.append(max_flux_time)
            else:
                max_flux_values.append(0)
            
        
        current_object_new_df['SNID'] = jnp.array(current_SNID)
        current_object_new_df['time'] = jnp.array(new_time)
        current_object_new_df['flux'] = jnp.array(new_flux)
        current_object_new_df['fluxerr'] = jnp.array(new_flux_err)
        current_object_new_df['band_index'] = jnp.array(band_index)
        
        current_object_new_df['norm_factor'] = [max_flux_values]*len(band_index)
        current_object_new_df['max_time'] = [max_flux_time_values]*len(band_index)

    else:
        raise ValueError("the norm must be either 0, 1 or None")

    return current_object_new_df

def add_object_band_index(df_phot, bands=['g','r']):
    """Add the object and band index to photometric dataframe

    Parameters
    ----------
    df_phot: pd.DataFrame
        photometry file

    Returns
    -------
    df_phot: pd.DataFrame
        updated photometry file
    """

    band_index = (df_phot['FLT'] == bands[1]).values*1

    df_phot["band_index"] = band_index

    return df_phot

def extract_subsample(df_head, df_phot, event_type, num_sample):
    """Extract a subsample of data in df_phot

    Parameters
    ----------
    df_head: pd.DataFrame
        headerfile
    df_phot: pd.DataFrame
        photometry file
    event_type: string
        type of extraction to perform.
        Can be "kn", "non-kn", "random"
    num_sample: int
        number of samples to draw

    Returns
    -------
    df_head_sampled: pd.DataFrame
        header file for the selcted objects
    df_phot_sampled: pd.DataFrame
        photometric data of selected objects
    """

    if event_type not in ["kn", "non-kn", "random"]:
        raise ValueError("Can only be kn, non-kn, random")
    
    if event_type == "kn":
        ids = df_head["SNID"][df_head["type"] == "KN"]
    elif event_type == "nonkn":
        ids = df_head["SNID"][df_head["type"] == "KN"]
    else:
        ids = df_head["SNID"]

    selected_ids = ids.sample(num_sample)

    df_head_sampled = df_head[df_head["SNID"].isin(selected_ids)]
    df_phot_sampled = df_phot[df_phot["SNID"].isin(selected_ids)]

    return df_head_sampled, df_phot_sampled
    
    
