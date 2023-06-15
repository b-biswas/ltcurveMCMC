import numpy as np
import pandas as pd

def preprocess_SNANA(df_head, df_phot, norm_bands="r"):
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

    if norm_bands not in ["r", "individual"]:
        raise ValueError("the norm must be either individual or r")

    if norm_bands == 'r':
        for object_id in df_head['SNID']:

            object_df = df_phot[df_phot["SNID"] == object_id]
            r_band_data = object_df[object_df['FLT'] == 'r']
            
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

    else:
        for object_id in df_head['SNID']:

            object_df = df_phot[df_phot["SNID"] == object_id]
            new_time = []
            new_flux = []
            new_flux_err = []
            band_index = []
            max_flux_values = []
            #print(object_df)
            for band in ['g', 'r']:
                band_df = object_df[object_df['FLT'] == band]

                if len(band_df)>0:
                
                    max_flux_loc = np.argmax(band_df["FLUXCAL"])
                    max_flux = band_df['FLUXCAL'].iloc[max_flux_loc]
                    max_flux_time = band_df['MJD'].iloc[max_flux_loc]
                    
                    new_time.extend(band_df['MJD'] - max_flux_time)
                    new_flux.extend(band_df['FLUXCAL'] / max_flux)
                    new_flux_err.extend(band_df['FLUXCALERR'] / max_flux)

                    band_index.extend(band_df['band_index'].values)

                    max_flux_values.append(max_flux)
                
            current_object_new_df = {}
            current_object_new_df['SNID'] = object_df['SNID']
            current_object_new_df['time'] = np.array(new_time)
            current_object_new_df['flux'] = np.array(new_flux)
            current_object_new_df['fluxerr'] = np.array(new_flux_err)
            current_object_new_df['object_index'] = object_df['object_index']
            current_object_new_df['band_index'] = np.array(band_index)
            
            current_object_new_df['norm_factor'] = [max_flux_values]*len(band_index)

            current_object_new_df = pd.DataFrame.from_dict(current_object_new_df)
            new_object_dfs.append(current_object_new_df)

    mcmc_format_df = pd.concat(new_object_dfs, axis=0, ignore_index=True)

    return mcmc_format_df

def add_object_band_index(df_phot):
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
    object_index = []
    last_object = 0
    count = 0

    for i in range(len(df_phot)):

        if i == 0:
            last_object = df_phot['SNID'].iloc[0]

        if df_phot['SNID'].iloc[i] == last_object:
            object_index.append(count)

        else:
            last_object = df_phot['SNID'].iloc[i]
            count = count+1
            object_index.append(count)
    
    band_index = (df_phot['FLT'] == 'g').values*1

    df_phot["object_index"] = object_index
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
    
    
