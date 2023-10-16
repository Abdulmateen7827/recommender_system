from src.logger import logging
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.utils import load_items,load_item_train,load_user_train
from src.utils import load_object,gen_user_vecs
from src.components.data_transformation import DataTransformation
from numpy import genfromtxt


class PredictPipeline:
    def __init__(self):
        self.transform = DataTransformation()
        self.model = load_object('artifacts/model.pkl')

        self.item_train = load_item_train()
        self.user_train = load_user_train()
    
        self.transform.scalerItem.fit_transform(self.item_train)

        self.item_vecs = load_items()
        self.user_train = load_user_train()
        self.transform.scalerUser.fit_transform(self.user_train)

    def predict(self,user_vec):
                # generate and replicate the user vector to match the number movies in the data set.
        user_vecs = gen_user_vecs(user_vec,len(self.item_vecs))

        # scale our user and item vectors
        suser_vecs = self.transform.scalerUser.transform(user_vecs)
        sitem_vecs = self.transform.scalerItem.transform(self.item_vecs)

        
        y_p = self.model.predict([suser_vecs[:, 1:], sitem_vecs[:, 1:]])
        return y_p



