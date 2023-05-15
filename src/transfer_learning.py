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
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.metrics import Accuracy 





STAGE = "PREPARE_BASE_MODEL" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def update_odd_even_labels(labels):
    for idx,label in enumerate(labels):
        labels[idx] = np.where(label%2==0,1,0)
    return labels
    

def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    
    
    seed = 2023
    
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # params = read_yaml(params_path)
    base_model_path =  os.path.join('artifacts','models','base_model.h5')
    base_model = tf.keras.models.load_model(base_model_path)
    
    base_model.summary()
    
    # freeze the weights
    
    for layer in base_model.layers[:-1]:
        print(f'before freezing weights {layer.name}:{layer.trainable}')
        layer.trainable =False
        print(f'after freezing weights {layer.name}:{layer.trainable}')
        
    # modifing last layer for our Problem statement
    
    base_layers = base_model.layers[:-1]
    
    new_model = Sequential(layers=base_layers)
    new_model.add(
        Dense(2,activation='softmax',name = 'output_layer')
    )
    new_model.summary()
    
    
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    X_valid, X_train = X_train_full[: 5000] / 255., X_train_full[5000: ]/255.
    y_valid, y_train = y_train_full[: 5000], y_train_full[5000:]
    X_test = X_test/255.
    
    y_train_bin,y_valid_bin,y_test_bin = update_odd_even_labels([y_train,y_valid,y_test])
    
    new_model.compile(loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=SGD(learning_rate=1e-3)
                  )
    history = new_model.fit(X_train,y_train_bin,epochs=20,validation_data=(X_valid,y_valid_bin),verbose=2)
    
    new_model.evaluate(X_test,y_test_bin)
    
    
    # global model_path
    new_model_path = os.path.join('artifacts','models','new_model.h5')
    
    new_model.save(new_model_path)
    logging.info(f"Base Model is saved at {new_model_path}")
    
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e