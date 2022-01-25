import os, requests, zipfile
from .data_catalog import DATA_CATALOG

DATASET_MODULE = "emtk/datasets/"

def check_downloaded(dataset_name):
    """Check if the dataset is already in the dataset dictionary
    Parameters
    ----------
    dataset_name : str
        Name of the dataset, path to raw data directory, e.g. '../../dataset_name/'
    Returns
    -------
    bool
        True if dataset is in dataset folder
        False if not
    """
    return os.path.isfile(DATASET_MODULE + dataset_name + '.zip')


def check_unzipped(dataset_name):
    """Check if the dataset is already unzipped in the datasets dictionary
    Parameters
    ----------
    dataset_name : str
        Name of the dataset, path to raw data directory, e.g. '../../dataset_name/'
    Returns
    -------
    bool
        True if dataset is unzipped in dataset folder
        False if not
    """
    return os.path.isdir(DATASET_MODULE + dataset_name)

        
def download(dataset_name):
    """Download any dataset via a link to the data
    Parameters
    ----------
    dataset_name : str
        Name of the dataset, path to raw data directory, e.g. '../../dataset_name/'
    url : str
        link to the data
    
    is_zipped : bool
        True if the url links to a zip file of the data, False if it simply links to the data
    
    citation : str
        link to the paper where the dataset originates from
    """
    url, is_zipped, citation = DATA_CATALOG[dataset_name].values()

    # Check if dataset has already been downloaded
    if not check_downloaded(dataset_name):
        print('Downloading...')
        
        #creates a zip file of the data if unzipped
        if is_zipped == False:

            r = requests.get(url)
            with open(os.path.join(DATASET_MODULE, dataset_name + '.zip'), 'wb') as f:
                f.write(r.content)

    if not check_unzipped(dataset_name):
        print('unzipping...')

        #extract all data
        with zipfile.ZipFile(os.path.join(DATASET_MODULE, dataset_name + '.zip'), 'r') as data_zip:

            data_zip.extractall(os.path.join(DATASET_MODULE, dataset_name))

    print('Please cite this paper: ', citation)

    return DATASET_MODULE + dataset_name
