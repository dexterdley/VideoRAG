import sys
import io
import os
import json
import argparse
import time
import warnings
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.stats import spearmanr, kendalltau
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu

from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import matplotlib.pyplot as plt

dataframe = pd.read_csv("./vslice_features/minicpm_extraction_results.csv")
print(dataframe.head())

feature = np.load("./vslice_features/minicpm_summe_Air_Force_One.npz", allow_pickle=True)
print(feature["p_yes"], feature["p_no"])