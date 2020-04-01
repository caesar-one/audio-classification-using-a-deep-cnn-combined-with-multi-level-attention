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
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from glob import glob
import pickle
import h5py

from params import *
from torchvggish.vggish_input import waveform_to_examples


# This function was used in the past to load data in .pkl format. Now deprecated.
# We now use load_hdf5, which load data in .hdf5 format.
def load_pkl(save: bool = True,
             load_saved: bool = True,
             path: str = "",
             filename: str = "audio_data.pkl",
             cnn_type: str = "vggish",
             use_librosa: bool = False,
             overlap: bool = True,
             debug: bool = False,
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if overlap and T < 4:
        raise Exception('Number of slots must be >= 4')
    if not overlap and cnn_type == "vggish" and not use_librosa and T != 4:
        raise Exception('This combination of params needs T = 4')

    # Force to use librosa for resnet
    if cnn_type == "resnet" and not use_librosa:
        print('use_librosa forced to True')
        use_librosa = True

    if path + filename in glob(path + "*.pkl") and load_saved:
        # Load the dataset
        with open(path + filename, "rb") as f:
            (X_train, X_val, X_test, y_train, y_val, y_test) = pickle.load(f)
    else:
        # DATASET CREATION
        # We split the data into 3 sets: train (~60%), val (~20%), test (~20%).

        # Assign folders to the appropriate set
        wav_paths = glob(path + DATASET_PATH + "**/*.wav", recursive=True)
        wav_paths_train, wav_paths_val, wav_paths_test = [], [], []
        for p in wav_paths:
            if p.split("/")[-2] in ["fold1", "fold2"]:
                wav_paths_test.append(p)
            elif p.split("/")[-2] in ["fold3", "fold4"]:
                wav_paths_val.append(p)
            else:
                wav_paths_train.append(p)

        # Load the metadata
        metadata = pd.read_csv(path + METADATA_PATH)
        # Create a mapping from audio clip names to their respective label IDs
        name2class = dict(zip(metadata["slice_file_name"], metadata["classID"]))

        if cnn_type == "vggish":
            sr = SR_VGGISH
            samples_num = SAMPLES_NUM_VGGISH
            s_shape = S_VGGISH_SHAPE
            x_size = s_shape[0]
            y_size = s_shape[1]
        elif cnn_type == "resnet":
            sr = SR_RESNET
            samples_num = SAMPLES_NUM_RESNET
            s_shape = S_RESNET_SHAPE
            x_size = s_shape[1]
            y_size = s_shape[0]
        else:
            raise Exception("CNN type is not valid.")

        X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []
        for paths, setname in zip([wav_paths_train, wav_paths_val, wav_paths_test], ["train", "val", "test"]):
            for wav_path in tqdm(paths[:BATCH_SIZE] if debug else paths,
                                 desc=f"Converting {setname} samples in spectrograms"):
                # Create spectrogram and split into frames
                spec = create_spec(wav_path, cnn_type, sr, samples_num, x_size, y_size, use_librosa, overlap)
                # Split the spectrogram
                frames = split(spec, T, x_size, y_size, overlap)
                # Add a new axis for the channel dimension
                for i in range(len(frames)):
                    frames[i] = frames[i][np.newaxis, :, :]
                # Append each frames list to their respective set
                audio_filename = wav_path.split("/")[-1]

                if setname == "train":
                    X_train.append(frames)
                    y_train.append(int(name2class[audio_filename]))
                elif setname == "val":
                    X_val.append(frames)
                    y_val.append(int(name2class[audio_filename]))
                else:
                    X_test.append(frames)
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

        save_filename = f"audio_data_{cnn_type}{'' if use_librosa else '_native'}_{T}{'_s' if overlap else ''}"
        if save:
            # Save the dataset
            with open(path + save_filename, "wb") as f:
                # protocol=4 saves correctly variables of size more than 4 GB
                pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f, protocol=4)

    return X_train, X_val, X_test, y_train, y_val, y_test


# The main loading function.
def load_hdf5(path: str = "",
              save_filename: str = "audio_data.h5",
              cnn_type: str = "vggish",
              use_librosa: bool = False,
              overlap: bool = True,
              debug: bool = False,
              features: tuple = ("spectrogram",)  # "mfcc", "crp"),
              ) -> str:
    """
    Loads data and store it in the specified path.

    :param path: absolute path of the data
    :param save_filename: the name of the file which will contain data
    :param cnn_type: "resnet" or "vggish", depending on which net we are considering
    :param use_librosa: If True, use librosa to generate the spectrogram, otherwise use torchvggish
    :param overlap: If True, the split has *num_frames* overlapping frame. Each frame has a fixed length of 1 second.
        Otherwise, the split has T contiguous slots.
        Each slot has length *samples_num* / *num_frames*
    :param debug: If True, load just a small part of the data in order to debug
    :param features: the features to consider when creating data (now we have only spectrogram, but we could add also
        features like MFCC or CRP, etc.
    """

    audio_data = h5py.File(path + save_filename, "w-")
    group_name = f"{cnn_type}_{T}{'_s' if overlap else ''}"
    audio_data.create_group(group_name)
    dataset = audio_data[group_name]
    if overlap and T < 4:
        raise Exception('Number of slots must be >= 4')
    if not overlap and cnn_type == "vggish" and not use_librosa and T != 4:
        raise Exception('This combination of params needs T = 4')

    # Force to use librosa for resnet
    if cnn_type == "resnet" and not use_librosa:
        print('use_librosa forced to True')
        use_librosa = True

    # DATASET CREATION
    # We split the data into 3 sets: train (~60%), val (~20%), test (~20%).

    # Assign folders to the appropriate set
    wav_paths = glob(path + DATASET_PATH + "**/*.wav", recursive=True)
    wav_paths_train, wav_paths_val, wav_paths_test = [], [], []
    for p in wav_paths:
        if p.split("/")[-2] in ["fold1", "fold2"]:
            wav_paths_test.append(p)
        elif p.split("/")[-2] in ["fold3", "fold4"]:
            wav_paths_val.append(p)
        else:
            wav_paths_train.append(p)

    # Load the metadata
    metadata = pd.read_csv(path + METADATA_PATH)
    # Create a mapping from audio clip names to their respective label IDs
    name2class = dict(zip(metadata["slice_file_name"], metadata["classID"]))  # TODO check this!!!!!

    if cnn_type == "vggish":
        sr = SR_VGGISH
        samples_num = SAMPLES_NUM_VGGISH
        s_shape = S_VGGISH_SHAPE
        x_size = s_shape[0]
        y_size = s_shape[1]
    elif cnn_type == "resnet":
        sr = SR_RESNET
        samples_num = SAMPLES_NUM_RESNET
        s_shape = S_RESNET_SHAPE
        x_size = s_shape[1]
        y_size = s_shape[0]
    else:
        raise Exception("CNN type is not valid.")

    if debug:
        wav_paths_test, wav_paths_train, wav_paths_val = wav_paths_test[:8], wav_paths_train[:8], wav_paths_val[
                                                                                                  :8]
    X_train = dataset.create_dataset("X_train", shape=(len(wav_paths_train), T, len(features), y_size, x_size))
    y_train = dataset.create_dataset("y_train", shape=(len(wav_paths_train),))
    X_val = dataset.create_dataset("X_val", shape=(len(wav_paths_val), T, len(features), y_size, x_size))
    y_val = dataset.create_dataset("y_val", shape=(len(wav_paths_val),))
    X_test = dataset.create_dataset("X_test", shape=(len(wav_paths_test), T, len(features), y_size, x_size))
    y_test = dataset.create_dataset("y_test", shape=(len(wav_paths_test),))

    _min, _max = np.float('inf'), np.float('-inf')

    for paths, setname in zip([wav_paths_train, wav_paths_val, wav_paths_test], ["train", "val", "test"]):
        counter = 0
        for wav_path in tqdm(paths, desc=f"Converting {setname} samples in spectrograms"):
            try:
                if use_librosa:
                    # Load the audio clip stored at *wav_path* in an audio array
                    audio_array, _ = librosa.load(wav_path, sr=sr)
                    # Make sure that all audio arrays are of the same length *samples_num*
                    # (cut if larger, zero-fill if smaller)
                    audio_array = audio_array[:samples_num]
                    reshaped_array = np.zeros((samples_num,))
                    reshaped_array[:audio_array.shape[0]] = audio_array
                else:
                    audio_array, _ = sf.read(wav_path, dtype='int16')
                    assert audio_array.dtype == np.int16, 'Bad sample type: %r' % audio_array.dtype
                    audio_array = audio_array / 32768.0  # Convert to [-1.0, +1.0]
                # Create spectrogram
                spec = create_spec(audio_array, cnn_type, sr, samples_num, x_size, y_size, use_librosa, overlap)
                # mfcc = create_mfcc(S=spec, sr=sr, y_size=y_size)
            except RuntimeError:
                print(f"{wav_path} was not read (Runtime error)")
                continue
            except OSError:
                print(f"{wav_path} was not read (OS error")
                continue
            # Split the spectrogram
            frames = split(spec, T, x_size, y_size, overlap)
            # Add a new axis for the channel dimension
            frames = frames[:, np.newaxis, :, :]
            # Append each frames list to their respective set
            audio_filename = wav_path.split("/")[-1]
            if setname == "train":
                X_train[counter] = frames
                y_train[counter] = int(name2class[audio_filename])
            elif setname == "val":
                X_val[counter] = frames
                y_val[counter] = int(name2class[audio_filename])
            else:
                X_test[counter] = frames
                y_test[counter] = int(name2class[audio_filename])

            _min, _max = np.min([_min, np.min(frames)]), np.max([_max, np.max(frames)])

            counter += 1

    audio_data.close()

    return path + save_filename


def create_spec(audio_array: np.ndarray,
                cnn_type: str,
                sr: int,
                samples_num: int,
                x_size: int,
                y_size: int,
                use_librosa: bool,
                overlap: bool
                ) -> np.ndarray:
    """
    Create the spectrogram of the audio data contained in *wav_path*, np.ndarray (y_size, *)
        Note: * means that the second dimension is variable. The parameters that affect it are T and overlap
        (on which depends the param hop_length of librosa.feature.melspectrogram)

    :param audio_array: the data of which computing the spectrogram
    :param cnn_type: from load_hdf5
    :param sr: sampling rate
    :param samples_num: number of total samples in the audio array (required for cutting/padding)
    :param x_size: "time pixels" of a single spectrogram frame (see the split function)
    :param y_size: "frequency pixels" of a single spectrogram frame (see the split function)
    :param use_librosa: from load_hdf5
    :param overlap: from load_hdf5
    :return: the spectrogram of the audio data contained in *audio_array*
    """

    if use_librosa:
        # Compute the spectrogram of the reshaped array.
        # The granularity on frequency (n_mels) axis is y_size
        # The granularity on time axis (hop_length) depends on whether we will divide the result in frames or not
        if cnn_type == "vggish":
            spec = librosa.feature.melspectrogram(audio_array, sr=sr, n_mels=64, hop_length=160, center=False,
                                                  htk=True, fmin=125, fmax=7500)
        elif cnn_type == "resnet":
            spec = librosa.feature.melspectrogram(audio_array, sr=sr, n_mels=y_size,
                                                  hop_length=samples_num // (x_size * 4) if overlap
                                                  else samples_num // (x_size * T))
        else:
            raise Exception("CNN type is not valid.")
        # Convert the spectrogram entries into decibels, with respect to ref=1.0.
        # If x is an entry of the spectrogram, this computes the scaling x ~= 10 * log10(x / ref)
        spec = librosa.power_to_db(spec)

    else:
        # slots = wavfile_to_examples(wav_path, return_tensor=False)
        slots = waveform_to_examples(audio_array, sr, return_tensor=False)
        reshaped = np.zeros((4, slots.shape[1], slots.shape[2]))
        reshaped[:slots.shape[0]] = slots
        reshaped = np.swapaxes(reshaped, 1, 2)
        spec = np.concatenate(reshaped[:4], axis=1)

    return spec


def split(spec: np.ndarray, num_frames: int, x_size: int, y_size: int, overlap: bool) -> np.ndarray:
    """
        Split the spectrogram *spec* in frames (by overlapping them or not, depending on the param *overlap*.
        Each frame will represent a feature of a specific audio clip.

        :param spec: the spectogram to split
        :param num_frames: number of frames into which we split the spectrogram
        :param x_size: (see create_spec)
        :param y_size: (see create_spec)
        :param overlap: from load_hdf5
        :return: the spectrogram split in frames, np.ndarray (y_size, x_size)
    """

    if overlap:
        frames = overlapping_split(spec, num_frames, x_size)
    else:
        frames = contiguous_split(spec, num_frames, x_size)

    # Check shape and add a new axis in position 0 (this will be the channel axis)
    for i in range(frames.shape[0]):
        assert frames[i].shape == (y_size, x_size), \
            "{} should be ({},{}); instead is ({},{})".format(i, y_size, x_size, frames[i].shape[0], frames[i].shape[1])

    return frames


def overlapping_split(spec: np.ndarray, num_frames: int, frame_length: int) -> np.ndarray:
    step_length = spec.shape[1] - frame_length

    return np.array(
        [spec[:, i: i + frame_length] for i in range(0, spec.shape[1], step_length // (num_frames - 1))][:num_frames])


def contiguous_split(spec: np.ndarray, num_frames: int, frame_length: int) -> np.ndarray:
    return np.array([spec[::, i: i + frame_length] for i in range(0, spec.shape[1], frame_length)][:num_frames])


def create_mfcc(S: np.ndarray,
                sr: int,
                y_size: int,
                ) -> np.ndarray:
    return librosa.feature.mfcc(S=S, sr=sr, n_mfcc=y_size)


def normalize(d: np.ndarray, _min: float, _max: float) -> np.ndarray:
    return (d - _min) / (_max - _min)


def plot_spec(spec: np.ndarray) -> None:
    librosa.display.specshow(spec, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    pth = load_hdf5(
        save_filename='audio_data.h5',
        path='/Volumes/GoogleDrive/Il mio Drive/Audio-classification-using-multiple-attention-mechanism/',
        cnn_type='resnet',
        overlap=True,
        use_librosa=True)
