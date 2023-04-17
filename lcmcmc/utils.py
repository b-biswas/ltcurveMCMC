import os 

def get_data_dir_path():
    """Fetch path to the data folder of maddeb.

    Returns
    -------
    data_dir: str
        path to data folder

    """
    curdir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(curdir, "data")

    return data_dir