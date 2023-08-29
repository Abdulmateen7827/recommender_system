import requests
import numpy as np


data = (user_id=2000,action=5.0,adventure=0.0,animation=0.0,children=0.0,comedy=0.0,
                       crime=0.0,documentry=0.0,drama=0.0,
                       scifi=5.0,thriller=0.0,film_noir=0.0,musical=0.0,war=0.0,western=0.0,horror=0.0,fantasy=0.0,romance=0.0,mystery=0.0)

data = np.array([data])

url = 'http://localhost:9696/predict'
response = requests.post(url=url,data=data)
print(response)