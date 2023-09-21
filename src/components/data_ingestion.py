from src.logger import logging
import tensorflow as tf
import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from dataclasses import dataclass
from numpy import genfromtxt,savetxt
from src.utils import load_data
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.pipeline.training_pipeline import ModelTrainer
import mlflow




mlflow.set_experiment("Recommender system")
mlflow.tensorflow.autolog()
@dataclass
class DataIngestionConfig:
    logging.info('Initiating data config')
    movie_train: str=os.path.join('./artifacts','content_item_train.csv')
    movie_test: str=os.path.join('artifacts','content_item_test.csv')
    ratings_train:  str=os.path.join('artifacts','content_user_train.csv')
    ratings_test: str=os.path.join('artifacts','content_user_test.csv')
    y_train: str=os.path.join('artifacts','y_train.csv')
    y_test: str=os.path.join('artifacts','y_test.csv')

#Item content = movie genre represented in a one-hot vector
# # user content = matrix factorization of per genre rating by user
class DataIngestion:
    logging.info("Initiating data ingestion...")
    def __init__(self):
        self.Ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('Loading data')
            item_train,user_train,y_train = load_data()

            logging.info("Splitting data into train and test")
            item_train,item_test = train_test_split(item_train,train_size=0.8,shuffle=True,random_state=42)
            user_train,user_test = train_test_split(user_train,train_size=0.8,shuffle=True,random_state=42)
            y_train,y_test = train_test_split(y_train,shuffle=True,train_size=0.8,random_state=42)

            savetxt(self.Ingestion_config.movie_train,item_train,delimiter=',')
            savetxt(self.Ingestion_config.movie_test,item_test,delimiter=',')
            savetxt(self.Ingestion_config.ratings_train,user_train,delimiter=',')
            savetxt(self.Ingestion_config.ratings_test,user_test,delimiter=',')
            savetxt(self.Ingestion_config.y_train,y_train,delimiter=',')
            savetxt(self.Ingestion_config.y_test,y_test,delimiter=',')

            logging.info("Ingestion completed.")

            return(
                self.Ingestion_config.movie_train,
                self.Ingestion_config.movie_test,
                self.Ingestion_config.ratings_train,
                self.Ingestion_config.ratings_test,
                self.Ingestion_config.y_train,
                self.Ingestion_config.y_test
            )
            # item_train.to_csv(self.Ingestion_config.movie_train,index=False,header=True)


        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj = DataIngestion()
    item_train,item_test,user_train,user_test,y_train,y_test = obj.initiate_data_ingestion()

    transform = DataTransformation()
    Itrain,Itest,Utrain,Utest,y_train,y_test = transform.intiate_transform(item_train,item_test,user_train,user_test,y_train,y_test)
    logging.info("Transformation completed")
    train = ModelTrainer()
    logging.info("Begin train")
    model = train.initialize_training(num_user_features=Utrain.shape[1]-1,num_item_features=Itrain.shape[1]-1,learning_rate=0.01)
<<<<<<< HEAD
    train.train(batch_size=128,user_train=Utrain,item_train=Itrain,model=model,y_train=y_train,epochs=3)
=======
    train.train(batch_size=len(Utrain),user_train=Utrain,item_train=Itrain,model=model,y_train=y_train,epochs=30)
>>>>>>> parent of d8d7152 (Web service)
    train.dist_matrix(num_item_ft=Itrain.shape[1]-1)



