from src.logger import logging
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.utils import load_items,load_item_train
from src.utils import load_object
from src.components.data_transformation import DataTransformation
from numpy import genfromtxt


class PredictPipeline:
    pass
logging.info("Testing....")
transform = DataTransformation()
model = load_object('artifacts/model_m.pkl')

item_train = load_item_train()
scalerItem = transform.scalerItem.fit(item_train)

item_vecs = load_items()
scaled_iv = scalerItem.transform(item_vecs)
vms = model.predict(scaled_iv[:,1:])


