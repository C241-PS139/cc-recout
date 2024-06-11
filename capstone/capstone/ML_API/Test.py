import json
import requests

data ={"gender_product": "Men",
       "city": "Jakarta"}

url="http://localhost:5000/recommend"

response = requests.post(url, json=data)

print(response.json())