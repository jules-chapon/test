"""Functions for remote training"""

import logging

from src.configs import ml_config

def remote_training() -> None:
    for exp, config in ml_config.EXPERIMENTS_CONFIGS:
        logging.info(f"Training experiment {exp}")
        model = 

