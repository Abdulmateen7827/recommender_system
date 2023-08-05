from numpy import genfromtxt
import pickle
import os
from src.exception import CustomException
import sys


def load_data():
    item_train = genfromtxt(
        './notebooks/data/ml-latest-small/content_item_train.csv', delimiter=',')
    user_train = genfromtxt(
        'notebooks/data/ml-latest-small/content_user_train.csv', delimiter=',')
    y_train = genfromtxt(
        'notebooks/data/ml-latest-small/content_y_train.csv', delimiter=',')
    # item_vecs = genfromtxt('./data/content_item_vecs.csv', delimiter=',')
    
    return item_train,user_train,y_train

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

    
