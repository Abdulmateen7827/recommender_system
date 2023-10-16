from numpy import genfromtxt
import pickle
import os
from src.exception import CustomException
import sys
import numpy as np

def load_data():
    item_train = genfromtxt(
        'notebooks/data/content_item_train.csv', delimiter=',')
    user_train = genfromtxt(
        'notebooks/data/content_user_train.csv', delimiter=',')
    y_train = genfromtxt(
        'notebooks/data/content_y_train.csv', delimiter=',')

    return item_train,user_train,y_train,
    
def load_items():
    item_vecs = genfromtxt(
        'notebooks/data/content_item_vecs.csv', delimiter=',')
    return item_vecs
def load_item_train():
    item= genfromtxt(
        './notebooks/data/ml-latest-small/content_item_train.csv', delimiter=',')
    return item
def load_user_train():
    user = genfromtxt(
        'notebooks/data/ml-latest-small/content_user_train.csv', delimiter=',')
    return user
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def sq_dist(a,b):
    """
    Returns the squared distance between two vectors
    Args:
      a (ndarray (n,)): vector with n features
      b (ndarray (n,)): vector with n features
    Returns:
      d (float) : distance
    """  
    d = np.sum(np.square(a-b))
    return d

def gen_user_vecs(user_vec, num_items):
    """ given a user vector return:
        user predict maxtrix to match the size of item_vecs """
    user_vecs = np.tile(user_vec, (num_items, 1))
    return(user_vecs)
