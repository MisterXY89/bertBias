import pretty_errors

import os
import glob
import json
import random

import logging

import torch

import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForMaskedLM

# Set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set paths
BASE_DIR = f"{os.sep}".join(os.path.dirname(os.path.abspath(__file__)).split(f"{os.sep}")[:-1]) + f"{os.sep}"
DATA_DIR = BASE_DIR + f"data{os.sep}"
LOG_DIR = BASE_DIR + f"logs{os.sep}"
RESULTS_DIR = BASE_DIR + f"results{os.sep}"


# Set logging
logging.basicConfig(format='[%(asctime)s] \t %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# save additional log file
timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
fh = logging.FileHandler(f'{LOG_DIR}/run_{timestamp}.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)


def get_all_models():
    all_models = [
        "bert-base-uncased",
        "bert-large-uncased",
        "bert-base-multilingual-uncased",
        "distilbert-base-uncased"
    ]

    return all_models

def load_json(sent_file):
    ''' Load from json. We expect a certain format later, so do some post processing '''
    logger.info("Loading %s..." % sent_file)
    all_data = json.load(open(sent_file, 'r'))
    data = {}
    for k, v in all_data.items():
        examples = v["examples"]
        data[k] = examples
        v["examples"] = examples
    return all_data  # data


def build_data_name(test_path):
    """
    Build data name from test path

    Args:
        test_path (str): path to test

    Returns:
        data_name (str): name of data
    """
    return test_path.split("/")[-1].split(".")[0]


def load_data_dict():
    """
    Load data dict from all .jsonl files in data/ folder
    """
    return {
        build_data_name(test_path): load_json(test_path) for test_path in glob.glob(f"{DATA_DIR}/*.jsonl")
    }

