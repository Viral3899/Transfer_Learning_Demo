import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,ReLU,LeakyReLU,Flatten

STAGE = "PREPARE_BASE_MODEL" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    X_valid, X_train = X_train_full[: 5000] / 255., X_train_full[5000: ]/255.
    y_valid, y_train = y_train_full[: 5000], y_train_full[5000:]
    X_test = X_test/255.
    
    seed = 2023
    
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    #define layers
    
    LAYERS = [
        Flatten(input_shape= (28,28)),
        Dense(300,kernel_initializer='he_normal'),
        LeakyReLU(),
        Dense(150,kernel_initializer='he_normal'),
        LeakyReLU(),
        Dense(10,activation='softmax')
    ]
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e