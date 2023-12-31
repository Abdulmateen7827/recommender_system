from numpy import genfromtxt
import pickle
import os
from src.exception import CustomException
import sys
import numpy as np
import tabulate
from collections import defaultdict
import csv

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
        'notebooks/data/content_item_train.csv', delimiter=',')
    return item
def load_user_train():
    user = genfromtxt(
        'notebooks/data/content_user_train.csv', delimiter=',')
    return user
def movie_dict():
    movie_dict = defaultdict(dict)
    count = 0
#    with open('./data/movies.csv', newline='') as csvfile:
    with open('notebooks/data/content_movie_list.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in reader:
            if count == 0: 
                count +=1  #skip header
                #print(line) 
            else:
                count +=1
                movie_id = int(line[0])  
                movie_dict[movie_id]["title"] = line[1]  
                movie_dict[movie_id]["genres"] =line[2]  
    return movie_dict
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

def print_pred_movies(y_p, user, item, movie_dict, maxcount=10):
    """ print results of prediction of a new user. inputs are expected to be in
        sorted order, unscaled. """
    count = 0
    movies_listed = defaultdict(int)
    disp = [["y_p", "movie id", "rating ave", "title", "genres"]]

    for i in range(0, y_p.shape[0]):
        if count == maxcount:
            break
        count += 1
        movie_id = item[i, 0].astype(int)
        if movie_id in movies_listed:
            continue
        movies_listed[movie_id] = 1
        disp.append([y_p[i, 0], item[i, 0].astype(int), item[i, 2].astype(float),
                    movie_dict[movie_id]['title'], movie_dict[movie_id]['genres']])

    table = tabulate.tabulate(disp, tablefmt='html',headers="firstrow")
    return(table)
# def get_pred_movies_dict(y_p, user, item, movie_dict, maxcount=10):
#     """ Get a dictionary with predicted movie ratings and names for a new user. 
#         Inputs are expected to be in sorted order, unscaled."""
#     count = 0
#     movies_listed = set()
#     pred_movies_dict = {}

#     for i in range(y_p.shape[0]):
#         if count == maxcount:
#             break
#         count += 1
#         movie_id = item[i, 0].astype(int)
#         if movie_id in movies_listed:
#             continue
#         movies_listed.add(movie_id)
#         pred_movies_dict[y_p[i, 0]] = {
#             'movie_id': movie_id,
#             'rating_ave': item[i, 2].astype(float),
#             'title': movie_dict[movie_id]['title'],
#             'genres': movie_dict[movie_id]['genres']
#         }

#     return pred_movies_dict


# def get_pred_movies_dict(y_p, item, movie_dict, maxcount=10):
#     """ Get a dictionary with predicted movie ratings and names.
#         Inputs are expected to be in sorted order, unscaled."""
#     count = 0
#     movies_listed = set()
#     pred_movies_dict = {}

#     for i in range(y_p.shape[0]):
#         if count == maxcount:
#             break
#         count += 1
#         movie_id = item[i, 0].astype(int)
#         if movie_id in movies_listed:
#             continue
#         movies_listed.add(movie_id)
#         pred_movies_dict[y_p[i, 0]] = movie_dict[movie_id]['title']

#     return pred_movies_dict


def get_pred_movies_dict(y_p, item, movie_dict, maxcount=10):
    """ Get a dictionary with predicted movie names as keys and genres as values.
        Inputs are expected to be in sorted order, unscaled."""
    count = 0
    movies_listed = set()
    pred_movies_dict = {}

    for i in range(y_p.shape[0]):
        if count == maxcount:
            break
        count += 1
        movie_id = item[i, 0].astype(int)
        if movie_id in movies_listed:
            continue
        movies_listed.add(movie_id)
        pred_movies_dict[movie_dict[movie_id]['title']] = movie_dict[movie_id]['genres']

    return pred_movies_dict

#