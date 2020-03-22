


### DATASET PARAMS

dataset_path = "UrbanSound8K/audio/"
metadata_path = "UrbanSound8K/metadata/UrbanSound8K.csv"
# We know in advance that all audio clips are sampled at 22050 kHz, so we fixed the number of samples per clip at 88200,
# which correspond to 4 seconds.
sr = 16_000
samples_num = sr * 4



### MODEL PARAMS

s_resnet_shape = (224, 224)
s_vggish_shape = (96, 64)

T = 10  # number of bottleneck features
M = 2048  # size of a bottleneck feature
H = 600  # size of hidden layers
DR = 0.4  # dropout rate
K = 10  # number of classes



### SETTINGS

# Batch size for training
batch_size = 64

# Number of epochs to train for
num_epochs = 10

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

# Learning rate
lr = 0.001