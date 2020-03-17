import numpy as np
import pandas as pd
import librosa.display
from glob import glob
from tqdm import tqdm
import pickle

'''
This module is used to load the dataset.

Usage:

from dataset import load
X_train, X_val, X_test, y_train, y_val, y_test = load()
'''

dataset_path = "UrbanSound8K/audio"
metadata_path = "UrbanSound8K/metadata/UrbanSound8K.csv"
save_filename = "audio_data.pkl"
samples_number = 88200
s_size = 224
slots_num = 4


def normalize(d: np.ndarray, _min: float, _max: float) -> np.ndarray:
    return (d - _min) / (_max - _min)


# TODO Try "scarrozzatura" (parameter use_sliding)
def load(save: bool = True, load_saved: bool = True):

    if save_filename in glob("*.pkl") and load_saved:
        # Load the dataset
        with open(save_filename, "rb") as f:
            (X_train, X_val, X_test, y_train, y_val, y_test) = pickle.load(f)
    else:
        # DATASET CREATION
        # We split the data into 3 sets: train (60%), val (20%), test (20%).

        # Assign folders to the appropriate set
        wav_paths = glob(dataset_path + "/**/*.wav", recursive=True)
        wav_paths_train, wav_paths_val, wav_paths_test = [], [], []
        for path in wav_paths:
            if path.split("/")[-2] in ["fold1", "fold2"]:
                wav_paths_test.append(path)
            elif path.split("/")[-2] in ["fold3", "fold4"]:
                wav_paths_val.append(path)
            else:
                wav_paths_train.append(path)
        # Load the metadata
        metadata = pd.read_csv(metadata_path)
        # Create a mapping from audio clip names to their respective label IDs
        name2class = dict(zip(metadata["slice_file_name"], metadata["classID"]))

        X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []
        for paths, setname in zip([wav_paths_train, wav_paths_val, wav_paths_test], ["train", "val", "test"]):
            for wav_path in tqdm(paths, desc=f"Converting {setname} samples in spectrograms"):
                # Load the audio clip stored at *wav_path* in an audio array
                audio_array, _ = librosa.load(wav_path)
                # Make sure that all audio arrays are of the same length *samples_number*
                # (cut if larger, zero-fill if smaller)
                audio_array = audio_array[:samples_number]
                reshaped_sample = np.zeros((samples_number,))
                reshaped_sample[:audio_array.shape[0]] = audio_array
                # Split the audio array into *slots_num* slots
                slots = np.split(reshaped_sample, slots_num)
                # Compute the spectrogram for each slot
                spectrograms = [librosa.feature.melspectrogram(slot, hop_length=samples_number // (s_size * slots_num),
                                                               n_mels=s_size) for slot in slots]
                # Convert the spectrogram entries into decibels, with respect to ref=1.0.
                # If x is an entry of the spectrogram, this computes the scaling x ~= 10 * log10(x / ref)
                spectrograms = [librosa.power_to_db(s) for s in spectrograms]
                # Add a new dimension at position 0 (this will be the channel dimension)
                spectrograms = [s[np.newaxis, :s_size, :s_size] for s in spectrograms]
                # Append each spectrogram list to their respective set
                audio_filename = wav_path.split("/")[-1]
                if setname == "train":
                    X_train.append(spectrograms)
                    y_train.append(int(name2class[audio_filename]))
                elif setname == "val":
                    X_val.append(spectrograms)
                    y_val.append(int(name2class[audio_filename]))
                else:
                    X_test.append(spectrograms)
                    y_test.append(int(name2class[audio_filename]))

        # Convert spectrogram lists into numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        X_tot = np.concatenate([X_train, X_val, X_test])

        # Normalize the dataset in between [0,1]
        _min = np.min(X_tot)
        _max = np.max(X_tot)
        X_train = normalize(X_train, _min, _max)
        X_val = normalize(X_val, _min, _max)
        X_test = normalize(X_test, _min, _max)

        if save:
            with open(save_filename, "wb") as f:
                # protocol=4 saves correctly variables of size more than 4 GB
                pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f, protocol=4)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load(load_saved=False)
