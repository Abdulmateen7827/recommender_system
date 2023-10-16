from flask import Flask, render_template, request
import numpy as np
import pickle
# from src.pipeline.prediction_pipeline import PredictPipeline
from src.exception import CustomException
import sys


application = Flask(__name__)
app = application


with open('artifacts/model.pkl','rb') as m:
    model = pickle.load(m)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        user_id = int(request.form['user_id'])
        action = float(request.form['action'])
        adventure = float(request.form['adventure'])
        animation = float(request.form['animation'])
        childrens = float(request.form['childrens'])
        comedy = float(request.form['comedy'])
        crime = float(request.form['crime'])
        documentary = float(request.form['documentary'])
        drama = float(request.form['drama'])
        fantasy = float(request.form['fantasy'])
        horror = float(request.form['horror'])
        mystery = float(request.form['mystery'])
        romance = float(request.form['romance'])
        scifi = float(request.form['scifi'])
        thriller = float(request.form['thriller'])
        film_Noir = float(request.form['film_Noir'])
        musical = float(request.form['musical'])
        war = float(request.form['war'])
        western = float(request.form['western'])
        
        data = [user_id,action,adventure,animation,childrens,comedy,crime,documentary,drama,fantasy,horror,mystery,romance,scifi,thriller,film_Noir,musical,war,western]
        data = np.array(data)

        # model = PredictPipeline()
        return(model.predict(user_id = data))

    except Exception as e:
        raise CustomException(e,sys)
        
if __name__=="__main__":
    app.run(host="0.0.0.0", port = 9696)