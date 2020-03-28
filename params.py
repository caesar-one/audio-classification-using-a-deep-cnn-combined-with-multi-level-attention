##################
# DATASET PARAMS #
##################

DATASET_PATH = "UrbanSound8K/audio/"
METADATA_PATH = "UrbanSound8K/metadata/UrbanSound8K.csv"

# We know in advance that all audio clips are sampled at 22050 kHz, so we fixed the number of samples per clip at 88200,
# which correspond to 4 seconds.
MAX_SECONDS = 4
SR_VGGISH = 16_000
SAMPLES_NUM_VGGISH = SR_VGGISH * MAX_SECONDS
SR_RESNET = 22_050
SAMPLES_NUM_RESNET = SR_RESNET * MAX_SECONDS

#[x[1] for x in sorted(list(set([tuple(x) for x in pd.read_csv(METADATA_PATH)[["classID","class"]].to_numpy().tolist()])))]
TARGET_NAMES = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
                'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

##################
#  MODEL PARAMS  #
##################

S_RESNET_SHAPE = (224, 224)
S_VGGISH_SHAPE = (96, 64)

T = 10  # number of bottleneck features
M_RESNET = 2048  # size of a bottleneck feature (resnet)
M_VGGISH = 128 # size of a bottleneck feature (vggish)
M_VGGISH_JB = 512 * 6 * 4
H = 600  # size of hidden layers
DR = 0.4  # dropout rate
K = 10  # number of classes


##################
#    SETTINGS    #
##################

# Batch size for training
BATCH_SIZE = 64

# Number of epochs to train for
NUM_EPOCHS = 25

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
FEATURE_EXTRACT = True

# Learning rate
LR = 0.001
