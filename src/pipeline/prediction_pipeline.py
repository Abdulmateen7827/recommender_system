from src.logger import logging
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.utils import load_items,load_item_train, gen_user_vecs,load_user_train,load_y
from src.utils import load_object
from src.components.data_transformation import DataTransformation
from numpy import genfromtxt
from src.exception import CustomException
import sys
import numpy as np

class PredictPipeline:
    def __init__(self) -> None:
        pass
    def predict(user_vec):
        try:
            logging.info("Predicting....")
            transform = DataTransformation()
            model = load_object('artifacts/model.pkl')

            item_train = load_item_train()
            scalerItem = transform.scalerItem.fit(item_train)

            item_vecs = load_items()
            user_train = load_user_train()
            scalerUser = transform.scalerUser.fit(user_train)
            y_train = load_y()
            scalerTarget =transform.scalerTarget.fit(y_train.reshape(-1,1))
           
            logging.info('Transforming completed')
            user_vecs = gen_user_vecs(user_vec,len(item_vecs))
            logging.info("Generated user vecs")
            # scale our user and item vectors
            suser_vecs = scalerUser.transform(user_vecs)
            sitem_vecs = scalerItem.transform(item_vecs)
            logging.info('Predicting....')
            # make a prediction
        
            y_p = model.predict([suser_vecs[:, 1:], sitem_vecs[:, 1:]])

            # unscale y prediction 
            y_pu = scalerTarget.inverse_transform(y_p)

            # sort the results, highest prediction first
            sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
            sorted_ypu   = y_pu[sorted_index]
            sorted_items = item_vecs[sorted_index]
            return sorted_items

        except Exception as e:
            raise CustomException(e,sys)

class UserPreferences:
    def __init__(self,
                 user_id,
                 action,adventure,animation,children,comedy,crime,documentry,drama,fantasy,horror,mystery,romance,scifi,thriller,film_noir,
                 musical,war,western) -> float:
        self.user_id = user_id
        self.action = action
        self.adventure = adventure
        self.animation = animation
        self.children = children
        self.comedy = comedy
        self.crime = crime
        self.documentry = documentry
        self.drama = drama
        self.fantasy = fantasy
        self.horror = horror
        self.mystery = mystery  
        self.romance =romance
        self.scifi =scifi
        self.thriller = thriller
        self.film_noir = film_noir
        self.musical = musical
        self.war = war
        self.western =western

    def get_data_as_np_array(self):
        try:
            user_vec = np.array([[self.user_id,
                                  self.action,self.adventure,self.animation,self.children,self.comedy,self.crime,
                                  self.documentry,self.drama,self.fantasy,self.horror,self.mystery,self.romance,
                                  self.scifi,self.thriller,self.film_noir,self.musical,self.war,self.western]])
            
            return user_vec
        except Exception as e:
            raise CustomException(e,sys)

            