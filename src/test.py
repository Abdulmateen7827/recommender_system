import requests
import numpy as np
from src.pipeline.prediction_pipeline import PredictPipeline
from src.utils import load_data, load_items, movie_dict, print_pred_movies, get_pred_movies_dict
from src.components.data_transformation import DataTransformation
new_user_id = 5000
new_rating_ave = 5.0
new_action = 5.0
new_adventure = 0.0
new_animation = 0.0
new_childrens = 0.0
new_comedy = 0.0
new_crime = 0.0
new_documentary = 0.0
new_drama = 0.0
new_fantasy = 0.0
new_horror = 0.0
new_mystery = 0.0
new_romance = 0.0
new_scifi = 5.0
new_thriller = 0.0
new_rating_count = 2

user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                      new_action, new_adventure, new_animation, new_childrens,
                      new_comedy, new_crime, new_documentary,
                      new_drama, new_fantasy, new_horror, new_mystery,
                      new_romance, new_scifi, new_thriller]])


def prediction(user_vec):
    _,_,y_train = load_data()
    item_vecs = load_items()
    movie_dic = movie_dict()
    transform = DataTransformation()
    scaled_y_train = transform.scalerTarget.fit_transform(y_train.reshape(-1, 1))

    predict = PredictPipeline()
    pred = predict.predict(user_vec)
    y_pu = transform.scalerTarget.inverse_transform(pred)
    sorted_index = np.argsort(-y_pu, axis=0).reshape(-1).tolist()  #negate to get largest rating first
    sorted_ypu   = y_pu[sorted_index]
    sorted_items = item_vecs[sorted_index] 
    mov =  get_pred_movies_dict(y_p=sorted_ypu,item=sorted_items,movie_dict=movie_dic,maxcount=10)
    return(mov)
print(prediction(user_vec))