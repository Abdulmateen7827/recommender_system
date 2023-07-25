import tensorflow as tf
import numpy as np
import os
from src.logger import logging
from src.exception import CustomException
import sys
import pandas as pd
from dataclasses import dataclass

logging.info("Head of ingestion script")

@dataclass
class DataIngestionConfig:
    movie_data_path: str=os.path.join('artifacts','content_item_train.csv')
    ratings_data_path: str=os.path.join('artifacts','content_user_train.csv')
    # y_

#Item content = movie genre represented in a one-hot vector
# user content = matrix factorization of per genre rating by user
class DataIngestion:

    def __init__(self) -> str:
        self.Ingestion_config = DataIngestionConfig()

    logging.info("Data Ingestion starting ...")
    try:
        logging.info('Reading movies and ratings from local repo')
        df_movies = pd.read_csv('artifacts/movies.csv')
        df_ratings = pd.read_csv('artifacts/ratings.csv')



    except Exception as e:
        raise CustomException(e,sys)

    
    

if __name__=="__main__":
    DataIngestion()