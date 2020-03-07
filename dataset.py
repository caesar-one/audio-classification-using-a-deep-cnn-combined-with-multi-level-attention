import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
from torch import nn
import torch
from glob import glob
from tqdm import tqdm

dataset_path = "UrbanSound8K/audio"
metadata_path = "UrbanSound8K/metadata/UrbanSound8K.csv"

if __name__ == "__main__":
    wav_paths = glob(dataset_path + "/**/*.wav", recursive=True)
    metadata = pd.read_csv(metadata_path)
    name2class = dict(zip(metadata["slice_file_name"],metadata["class"]))
    X, y = [], []
    for w in tqdm(wav_paths):
        song, sr = librosa.load(w)
        X.append(song)
        song_filename = w.split("/")[-1]
        y.append(name2class[song_filename])
