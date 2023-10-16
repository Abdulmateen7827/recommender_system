import requests
import numpy as np
from src.pipeline.prediction_pipeline import PredictPipeline

new_user_id = 50000
new_ave_rating = 5.0
new_year = 2002
new_action = 0.0
new_adventure = 5.0
new_animation = 5.0
new_childrens = 5.0
new_comedy = 0.0
new_crime = 0.0
new_documentary = 0.0
new_drama = 0.0
new_fantasy = 0.0
new_horror = 0.0
new_mystery = 0.0
new_romance = 0.0
new_scifi = 0.0
new_thriller = 0.0
new_Film_Noir = 0.0
new_Musical = 0.0
new_War = 0.0
new_Western = 0.0

user_vec = np.array([[new_user_id,
                      new_action, new_adventure, new_animation, new_childrens,
                      new_comedy, new_crime, new_documentary,
                      new_drama, new_fantasy, new_horror, new_mystery,
                      new_romance, new_scifi, new_thriller,
                      new_Film_Noir,new_Musical, new_War,new_Western]])

def prediction(user_vec):

    predict = PredictPipeline()
    pred = predict.predict(user_vec)
    print(pred)
    return {'Prediction': pred}

print(prediction(user_vec))