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
# Keep free unless you need to access kaggle datasets. You'll need to modify the remote_training_template.ipynb.
KAGGLE_DATASET_LIST = []

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
