from typing import Tuple

try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import librosa.display
from glob import glob
import pickle

from model import s_resnet_shape, s_vggish_shape

from model import T as slots_num

from train import batch_size

'''
This module is used to load the dataset.

Usage:

from dataset import load
X_train, X_val, X_test, y_train, y_val, y_test = load()
'''

dataset_path = "UrbanSound8K/audio/"
metadata_path = "UrbanSound8K/metadata/UrbanSound8K.csv"
# We know in advance that all audio clips are sampled at 22050 kHz, so we fixed the number of samples per clip at 88200,
# which correspond to 4 seconds.
sr = 22050
samples_num = 88200


def normalize(d: np.ndarray, _min: float, _max: float) -> np.ndarray:
    return (d - _min) / (_max - _min)


'''
def load(save: bool = True, load_saved: bool = True, path: str = "", save_filename: str = "audio_data.pkl",
         use_sliding: bool = True, debug: bool = False) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if use_sliding and slots_num < 4:
        raise Exception('Number of slots must be >= 4')

    if path + save_filename in glob(path + "*.pkl") and load_saved:
        # Load the dataset
        with open(path + save_filename, "rb") as f:
            (X_train, X_val, X_test, y_train, y_val, y_test) = pickle.load(f)
    else:
        # DATASET CREATION
        # We split the data into 3 sets: train (~60%), val (~20%), test (~20%).

        # Assign folders to the appropriate set
        wav_paths = glob(path + dataset_path + "**/*.wav", recursive=True)
        wav_paths_train, wav_paths_val, wav_paths_test = [], [], []
        for p in wav_paths:
            if p.split("/")[-2] in ["fold1", "fold2"]:
                wav_paths_test.append(p)
            elif p.split("/")[-2] in ["fold3", "fold4"]:
                wav_paths_val.append(p)
            else:
                wav_paths_train.append(p)
        # Load the metadata
        metadata = pd.read_csv(path + metadata_path)
        # Create a mapping from audio clip names to their respective label IDs
        name2class = dict(zip(metadata["slice_file_name"], metadata["classID"]))

        X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []
        for paths, setname in zip([wav_paths_train, wav_paths_val, wav_paths_test], ["train", "val", "test"]):
            for wav_path in tqdm(paths[:batch_size] if debug else paths,
                                 desc=f"Converting {setname} samples in spectrograms"):
                # Load the audio clip stored at *wav_path* in an audio array
                audio_array, _ = librosa.load(wav_path)
                # Make sure that all audio arrays are of the same length *samples_number*
                # (cut if larger, zero-fill if smaller)
                audio_array = audio_array[:samples_num]
                reshaped_array = np.zeros((samples_num,))
                reshaped_array[:audio_array.shape[0]] = audio_array
                # Split the audio array
                if use_sliding:
                    slot_len = sr
                    # If using the "sliding" mode, the split is done into *slots_num* overlapping slots.
                    # Each slot has a fixed length of 1 second.

                    # slots = as_strided(
                    #    reshaped_array,
                    #    (slots_num, slot_len),  # shape of the *slots* array
                    #    ((samples_num - slot_len) // (slots_num - 1) * 8, 8)  # (bytes step, bytes per element)
                    # )#.copy()

                    slots = [reshaped_array[i: i + slot_len] for i in
                             range(0, samples_num, (samples_num - slot_len) // (slots_num - 1))][:slots_num]
                else:
                    # Otherwise, the split is done into *slots_num* contiguous slots.
                    # Each slot has length *samples_num* / *slots_num*
                    slots = np.split(reshaped_array, slots_num)
                    slot_len = len(slots)
                # Compute the spectrogram for each slot
                spectrograms = [librosa.feature.melspectrogram(slot, hop_length=samples_num // slot_len, n_mels=s_size)
                                for slot in slots]
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
            # Save the dataset
            with open(path + save_filename, "wb") as f:
                # protocol=4 saves correctly variables of size more than 4 GB
                pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f, protocol=4)

    return X_train, X_val, X_test, y_train, y_val, y_test
'''


def load(save: bool = True, load_saved: bool = True, path: str = "", save_filename: str = "audio_data.pkl",
         cnn_type: str = "vggish", use_sliding: bool = True, debug: bool = False) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if use_sliding and slots_num < 4:
        raise Exception('Number of slots must be >= 4')

    if path + save_filename in glob(path + "*.pkl") and load_saved:
        # Load the dataset
        with open(path + save_filename, "rb") as f:
            (X_train, X_val, X_test, y_train, y_val, y_test) = pickle.load(f)
    else:
        # DATASET CREATION
        # We split the data into 3 sets: train (~60%), val (~20%), test (~20%).

        # Assign folders to the appropriate set
        wav_paths = glob(path + dataset_path + "**/*.wav", recursive=True)
        wav_paths_train, wav_paths_val, wav_paths_test = [], [], []
        for p in wav_paths:
            if p.split("/")[-2] in ["fold1", "fold2"]:
                wav_paths_test.append(p)
            elif p.split("/")[-2] in ["fold3", "fold4"]:
                wav_paths_val.append(p)
            else:
                wav_paths_train.append(p)
        # Load the metadata
        metadata = pd.read_csv(path + metadata_path)
        # Create a mapping from audio clip names to their respective label IDs
        name2class = dict(zip(metadata["slice_file_name"], metadata["classID"]))

        X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []
        for paths, setname in zip([wav_paths_train, wav_paths_val, wav_paths_test], ["train", "val", "test"]):
            for wav_path in tqdm(paths[:batch_size] if debug else paths,
                                 desc=f"Converting {setname} samples in spectrograms"):
                # Load the audio clip stored at *wav_path* in an audio array
                audio_array, _ = librosa.load(wav_path)
                # Make sure that all audio arrays are of the same length *samples_number*
                # (cut if larger, zero-fill if smaller)
                audio_array = audio_array[:samples_num]
                reshaped_array = np.zeros((samples_num,))
                reshaped_array[:audio_array.shape[0]] = audio_array
                # Compute the spectrogram of the reshaped array.
                # The granularity on frequency (n_mels) axis depends on the chosen model.
                # The granularity on time axis (hop_length) depends on the fact we use sliding mode or not.
                s_shape = s_vggish_shape if cnn_type == "vggish" else s_resnet_shape
                s = librosa.feature.melspectrogram(reshaped_array, n_mels=s_shape[0],
                                                   hop_length=samples_num // (s_shape[1] * 4) if use_sliding
                                                   else samples_num // (s_shape[1] * slots_num))
                # Convert the spectrogram entries into decibels, with respect to ref=1.0.
                # If x is an entry of the spectrogram, this computes the scaling x ~= 10 * log10(x / ref)
                s = librosa.power_to_db(s)
                # Add a new dimension at position 0 (this will be the channel dimension)
                s = s[np.newaxis, :, :]

                # Split the spectrogram
                if use_sliding:
                    # If using the "sliding" mode, the split is done into *slots_num* overlapping slots.
                    # Each slot has a fixed length of 1 second.

                    # slots = as_strided(
                    #    reshaped_array,
                    #    (slots_num, slot_len),  # shape of the *slots* array
                    #    ((samples_num - slot_len) // (slots_num - 1) * 8, 8)  # (bytes step, bytes per element)
                    # )#.copy()
                    slots = [s[:, :, i: i + s_shape[1]] for i in
                             range(0, s.shape[2], (s.shape[2] - s_shape[1]) // (slots_num - 1))][:slots_num]
                else:
                    # Otherwise, the split is done into *slots_num* contiguous slots.
                    # Each slot has length *samples_num* / *slots_num*
                    slots = [s[:, :, i: i + s_shape[1]] for i in
                             range(0, s.shape[2], s_shape[1])][:slots_num]
                # Append each spectrogram list to their respective set
                audio_filename = wav_path.split("/")[-1]
                if setname == "train":
                    X_train.append(slots)
                    y_train.append(int(name2class[audio_filename]))
                elif setname == "val":
                    X_val.append(slots)
                    y_val.append(int(name2class[audio_filename]))
                else:
                    X_test.append(slots)
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
            # Save the dataset
            with open(path + save_filename, "wb") as f:
                # protocol=4 saves correctly variables of size more than 4 GB
                pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f, protocol=4)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':

    for n in [8, 10]:
        slots_num = n
        for u in [True, False]:
            load(save=True, load_saved=False,
                 path='/Volumes/GoogleDrive/Il mio Drive/Audio-classification-using-multiple-attention'
                      '-mechanism/',
                 save_filename='audio_data_vggish_%d_s.pkl' % n if u else 'audio_data_vggish_%d.pkl' % n,
                 cnn_type="vggish",
                 # debug=True,
                 use_sliding=u)
