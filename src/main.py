import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import uvicorn
from src.pipeline.prediction_pipeline import PredictPipeline

app = FastAPI(title='Recommender system')

@app.get("/")
def read_root():
    return {"message": "Welcome to the Recommender system"}


class Rec(BaseModel):
        user_id : int
        action : float
        adventure : float
        animation : float
        childrens : float
        comedy : float
        crime :float
        documentary : float
        drama : float
        fantasy : float
        horror : float
        mystery : float
        romance : float
        scifi : float
        thriller : float 
        film_Noir : float 
        musical : float 
        war : float
        western : float


# @app.on_event('startup')
# def load_pkl():
#     global model
#     with open('artifacts/model.pkl', 'rb') as file:
#         model = pickle.load(file)
global predict
predict = PredictPipeline()

@app.post("/predict")
def predict(rec: Rec):
    data_point = np.array(
          [
               [
                    rec.user_id,
                    rec.action,
                    rec.adventure,
                    rec.animation,
                    rec.childrens,
                    rec.comedy,
                    rec.crime,
                    rec.documentary,
                    rec.drama,
                    rec.fantasy,
                    rec.horror,
                    rec.fantasy,
                    rec.mystery,
                    rec.romance,
                    rec.scifi,
                    rec.thriller,
                    rec.film_Noir,
                    rec.musical,
                    rec.war,
                    rec.western
               ]
          ]
     )
    pred = predict.predict(data_point)
    # print(pred)
    return pred



