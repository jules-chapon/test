"""Constants"""

from pathlib import Path

###############################################################
#                                                             #
#                             PATHS                           #
#                                                             #
###############################################################

ROOT_DIR = Path(__file__).parent

OUTPUT_FOLDER = "output"

REMOTE_TRAINING_FOLDER = "remote_training"

###############################################################
#                                                             #
#                        REMOTE CONFIG                        #
#                                                             #
###############################################################

NOTEBOOK_ID = "test-notebook"  # This will be the name which appears on Kaggle.

GIT_USER = "jules-chapon"  # Your git user name

GIT_REPO = "test"  # Your current git repo

KAGGLE_DATASET_LIST = []  # Keep free unless you need to access kaggle datasets.
# You'll need to modify the remote_training_template.ipynb.


###############################################################
#                                                             #
#                          DATASETS                           #
#                                                             #
###############################################################

### HUGGING FACE

HF_DATASET_FOLDER = "jules-chapon/train-qrt-2024"

HF_DATASET_FILES = {"train": "train.csv", "test": "test.csv"}

### TRAINING RATIO

TRAIN_RATIO = 0.8


###############################################################
#                                                             #
#                        FIXED VALUES                         #
#                                                             #
###############################################################

RANDOM_SEED = 42

###############################################################
#                                                             #
#                      FEATURE SELECTION                      #
#                                                             #
###############################################################

### BORUTA

NB_RF_CLASSIFIERS_BORUTA = 100

NB_BORUTA_ESTIMATORS = "auto"

### RANDOM COLUMNS

NB_RANDOM_COLUMNS = 5

NB_RF_CLASSIFIERS_RANDOM_COLUMNS = 100

LOW_VALUE_FEATURE = 0

HIGH_VALUE_FEATURE = 1

### RANDOM COLUMNS

CORR_TYPE = "spearman"

THRESHOLD_CORR_LABEL = 0.05

THRESHOLD_CORR_FEATURE = 0.05
