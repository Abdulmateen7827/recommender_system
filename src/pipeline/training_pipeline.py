from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
import tensorflow as tf
import os
import sys
import pickle
from src.utils import save_object
import mlflow

logging.info("Trainer")
@dataclass
class ModelTrainerConfig:
    model_pkl_path: str=os.path.join('./artifacts','model.pkl')
    model_h5_path: str=os.path.join('./artifacts','model.h5')
    model_matrix: str=os.path.join('./artifacts','model_m.pkl')


class CreateNN(tf.keras.models.Sequential):
    logging.info("Creating NN")
    def __init__(self,num_outputs):
        super().__init__()
        self.add(tf.keras.layers.Dense(256,activation='relu'))
        self.add(tf.keras.layers.Dense(128,activation='relu'))
        self.add(tf.keras.layers.Dense(num_outputs))
class UserNN(CreateNN):
    def __init__(self, num_outputs):
        super().__init__(num_outputs)

class ItemNN(CreateNN):
    def __init__(self, num_outputs):
        super().__init__(num_outputs)


class ModelTrainer:
    logging.info("Model traner config")
    def __init__(self):
        self.model_config = ModelTrainerConfig()
    
    def initialize_training(self,num_user_features,num_item_features,learning_rate):
        input_user = tf.keras.layers.Input(shape=(num_user_features))
        vu = UserNN(num_outputs=32)(input_user)
        vu = tf.linalg.l2_normalize(vu, axis=1)

        input_item = tf.keras.layers.Input(shape=(num_item_features))
        vi = ItemNN(num_outputs=32)(input_item)
        vi = tf.linalg.l2_normalize(vi, axis=1)

        output = tf.keras.layers.Dot(axes=1)([vu,vi])
        model = tf.keras.Model([input_user,input_item],output)

        tf.random.set_seed(1)
        cost_fn = tf.keras.losses.MeanSquaredError()

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt,loss=cost_fn)

        return model

    def train(self,epochs,batch_size,user_train,item_train,y_train,model):
        try:
            logging.info("Training>>>>>......")
            tf.random.set_seed(1)
            model.fit([user_train[:, 1:], item_train[:, 1:]], y_train, epochs=epochs,batch_size=batch_size)
            

            logging.info('Saving trained model')
            save_object(self.model_config.model_pkl_path,model)
        except Exception as e:
            raise CustomException(e,sys)
        
    def dist_matrix(self,num_item_ft):
        try:
            logging.info('Matrix of distance between movies')

            input_item_m = tf.keras.layers.Input(shape=(num_item_ft))
            vm_m = ItemNN(num_outputs=32)(input_item_m) # use the trained item_NN
            vm_m = tf.linalg.l2_normalize(vm_m, axis=1)  # incorporate normalization as was done in the original model
            model_m = tf.keras.Model(input_item_m, vm_m)  

            save_object(self.model_config.model_matrix,model_m)
            logging.info("Saved dist_matrix")
        except Exception as e:
            raise CustomException(e,sys)

    




    
