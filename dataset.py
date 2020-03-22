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
import librosa.display
from glob import glob
import pickle

from model import T as slots_num
from params import *
from torchvggish.vggish_input import wavfile_to_examples

'''
This module is used to load the dataset.

Usage:

from dataset import load
X_train, X_val, X_test, y_train, y_val, y_test = load()
'''


def normalize(d: np.ndarray, _min: float, _max: float) -> np.ndarray:
    return (d - _min) / (_max - _min)


def load(save: bool = True,
         load_saved: bool = True,
         create_spec: bool = True,
         path: str = "",
         save_filename: str = "audio_data.pkl",
         cnn_type: str = "vggish",
         use_sliding: bool = True,
         debug: bool = False,
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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
        elif cnn_type == "resnet":
            sr = SR_RESNET
            samples_num = SAMPLES_NUM_RESNET
            s_shape = S_RESNET_SHAPE
        else:
            raise Exception("CNN type is not valid.")

        X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []
        for paths, setname in zip([wav_paths_train, wav_paths_val, wav_paths_test], ["train", "val", "test"]):
            for wav_path in tqdm(paths[:BATCH_SIZE] if debug else paths,
                                 desc=f"Converting {setname} samples in spectrograms"):
                if create_spec:
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
                    if cnn_type == "vggish":
                        s = librosa.feature.melspectrogram(reshaped_array, sr=sr, n_mels=64, hop_length=160, window="hann",
                                                           center=False, pad_mode="reflect", htk=True, fmin=125, fmax=7500)
                    elif cnn_type == "resnet":
                        s = librosa.feature.melspectrogram(reshaped_array, sr=sr, n_mels=s_shape[0],
                                                           hop_length=samples_num // (s_shape[1] * 4) if use_sliding
                                                           else samples_num // (s_shape[1] * slots_num))
                    else:
                        raise Exception("CNN type is not valid.")
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
                else:
                    slots = wavfile_to_examples(wav_path, return_tensor=False)
                    # TODO codice pecionata
                    slots = slots[:4]
                    reshaped_slots = np.zeros((4, 1, slots.shape[1], slots.shape[2]))
                    reshaped_slots[:slots.shape[0], 0] = slots
                    #slots = np.swapaxes(reshaped_slots, 2, 3)
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
    import dataset, model
    from train import *

    data_path = '/Volumes/GoogleDrive/Il mio Drive/Audio-classification-using-multiple-attention-mechanism/'
    postfix = 'vggish_nospec.pkl'

    model.T = 10
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.load(
        load_saved=False,
        save=True,
        path=data_path,
        save_filename='audio_data_' + postfix,
        create_spec=False
    )

    '''
    FROM_SCRATCH = True

    batch_size = 64
    num_epochs = 20
    feature_extract = True
    lr = 0.001

    dataloaders_dict = {
        "train": DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True),
        "val": DataLoader(list(zip(X_val, y_val)), batch_size=batch_size, shuffle=False),
        "test": DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False)
    }

    input_conf = "repeat"

    model_conf = [2, 2]

    cnn_conf = {
        "num_classes": 128,
        "use_pretrained": True,
        "just_bottleneck": True,
        "cnn_trainable": False,
        "first_cnn_layer_trainable": False,
        "in_channels": 3} if not FROM_SCRATCH else {

        "num_classes": 128,
        "use_pretrained": False,
        "just_bottleneck": True,
        "cnn_trainable": True,
        "first_cnn_layer_trainable": False,
        "in_channels": 3
    }

    model_ft = model.Ensemble(input_conf, cnn_conf, model_conf, device, cnn_type="vggish")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update if not FROM_SCRATCH else model_ft.parameters(), lr=lr)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    ###############################################################

    # Train and evaluate
    model_ft, hist, test_acc = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                           save_model_path=data_path + 'model_' + postfix + '.pkl', resume=False)
                                           '''
