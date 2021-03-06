
# POS task constants
POS_PATH = 'data/POS_data/POS_data'
POS_PATH_TRAIN = POS_PATH + '/VLSP2013_POS_train.txt'
POS_PATH_TEST = POS_PATH + '/VLSP2013_POS_test.txt'
POS_PATH_DEV = POS_PATH + '/VLSP2013_POS_dev.txt'
POS_TARGET = [
    'N',
    'Np',
    'CH',
    'M',
    'R',
    'A',
    'P',
    'V',
    'Nc',
    'E',
    'L',
    'C',
    'Ny',
    'T',
    'Nb',
    'Y',
    'Nu',
    'Cc',
    'Vb',
    'I',
    'X',
    'Z',
    'B',
    'Eb',
    'Vy',
    'Cb',
    'Mb',
    'Pb',
    'Ab',
    'Ni',
    'Xy',
    'NY',
    ]

# NER task constants
NER_PATH = 'data/NER_data'
NER_PATH_TRAIN = NER_PATH + '/train.txt'
NER_PATH_TEST = NER_PATH + '/test.txt'
NER_PATH_DEV = NER_PATH + '/dev.txt'
NER_TARGET = [
    "O",       # Outside of a named entity
    "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
    "I-MISC",  # Miscellaneous entity
    "B-PER",   # Beginning of a person's name right after another person's name
    "I-PER",   # Person's name
    "B-ORG",   # Beginning of an organisation right after another organisation
    "I-ORG",   # Organisation
    "B-LOC",   # Beginning of a location right after another location
    "I-LOC"    # Location
]

# Checkpoint
RESULT_PATH = '/content/drive/MyDrive/Colab Notebooks/Research/checkpoint'
# CHECKPOINT_PATH = RESULT_PATH + '/POS_Pretrained_phoBERT.pt'
# CHECKPOINT_PATH = RESULT_PATH + '/quantized_POS_Pretrained_phoBERT.pt'
# CHECKPOINT_PATH = RESULT_PATH + '/NER_Pretrained_phoBERT.pt'
CHECKPOINT_PATH = RESULT_PATH + '/quantized_NER_Pretrained_phoBERT.pt'

# MEASURE_MODEL_PATH = RESULT_PATH + '/POS_Pretrained_phoBERT.pt'
# MEASURE_MODEL_PATH = RESULT_PATH + '/quantized_POS_Pretrained_phoBERT.pt'
MEASURE_MODEL_PATH = RESULT_PATH + '/NER_Pretrained_phoBERT.pt'
# MEASURE_MODEL_PATH = RESULT_PATH + '/quantized_NER_Pretrained_phoBERT.pt'

# Device
CUDA_VISIBLE_DEVICES = '0'

# Training procedure
T_TOTAL = None
GLOBAL_STEP = 1
WARMUP_STEPS = 400
FINAL_THRESHOLD = 0.15
INITIAL_THRESHOLD = 1.0
FINAL_WARMUP = 2.0
INITIAL_WARMUP = 1.0
FINAL_LAMBDA = 1.0

MASKED_SCORE_LEARNING_RATE = 1e-2
LEARNING_RATE = 3e-5

REGULARIZATION = None