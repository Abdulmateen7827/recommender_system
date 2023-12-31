from src.logger import logging
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.utils import load_items, load_item_train, load_user_train, load_data
from src.utils import load_object, gen_user_vecs, movie_dict, get_pred_movies_dict
from src.components.data_transformation import DataTransformation
import numpy as np

class PredictPipeline:
    logging.info("Start prediction")
    def __init__(self):
        self.transform = DataTransformation()
        self.model = load_object('artifacts/model.pkl')

        self.item_train = load_item_train()
        self.user_train = load_user_train()
    
        self.transform.scalerItem.fit_transform(self.item_train)

        self.item_vecs = load_items()
        self.user_train = load_user_train()
        self.transform.scalerUser.fit_transform(self.user_train)

        _,_,y_train = load_data()
        self.transform.scalerTarget.fit_transform(y_train.reshape(-1, 1))

    def get_predict(self, user_vec):
        movie_dic = movie_dict()
        item_vecs = load_items()
        # generate and replicate the user vector to match the number movies in the data set.
        user_vecs = gen_user_vecs(user_vec, len(self.item_vecs))

        # scale our user and item vectors
        suser_vecs = self.transform.scalerUser.transform(user_vecs)
        sitem_vecs = self.transform.scalerItem.transform(self.item_vecs)
        
        y_p = self.model.predict([suser_vecs[:, 1:], sitem_vecs[:, 1:]])
        self.transform.scalerTarget.inverse_transform(y_p)
        sorted_index = np.argsort(-y_p, axis=0).reshape(-1).tolist()  #negate to get largest rating first
        sorted_ypu = y_p[sorted_index]
        sorted_items = item_vecs[sorted_index] 
        mov =  get_pred_movies_dict(y_p=sorted_ypu, item=sorted_items, movie_dict=movie_dic,maxcount=10)
        return mov
    logging.info("End prediction")


