from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity



app = Flask(__name__)
CITY = [ 'Sulawesi Selatan','Nusa Tenggara Barat','Jawa Barat','Maluku Utara','Kalimantan Tengah','Jakarta','Yogyakarta','Jawa Tengah','Lampung','Jawa Timur','Kalimantan Barat','Kalimantan Timur','Bali','Maluku','Gorontalo','Sulawesi Barat','Papua Barat','Kepulauan Riau','Kalimantan Selatan','Nusa Tenggara Timur','Sulawesi Utara','Sumatera Utara','Sulawesi Tengah','Sumatera Barat','Aceh','Riau','Bengkulu','Sulawesi Tenggara','Papua','Sumatera Selatan','Jambi','Bangka Belitung','Banten']
GENDER = ["Men" , "Women"]
# Load OneHotEncoder and dataset
onehot_encoder = joblib.load('onehot_encoder.pkl')
ds = pd.read_csv('dataset.csv')

# Convert temperature to categories
def convert_temperature(temperature):
    if temperature < 20:
        return "Cold"
    elif 20 <= temperature < 30:
        return "Normal"
    else:
        return "Hot"

# Function to convert Kelvin to Celsius
def kelvin_to_celcius(kelvin):
    return kelvin - 273.15

# Recommendation function
def recommend_outfits(gender, city):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    API_KEY = '5b48d68ce066834a01e3b108a9e5891a'

    url = base_url + "appid=" + API_KEY + "&q=" + city
    response = requests.get(url).json()

    temp_kelvin = response['main']['temp']
    temp_celcius = kelvin_to_celcius(temp_kelvin)
    feels_like_kelvin = response['main']['feels_like']
    feels_like_celcius = kelvin_to_celcius(feels_like_kelvin)
    description = response['weather'][0]['description']

    temperature = convert_temperature(temp_celcius)

    features_onehot = onehot_encoder.transform(ds[['gender_product', 'temperature', 'city']])
    query_vector = onehot_encoder.transform(np.array([gender, temperature, city]).reshape(1, -1))
    similarities = cosine_similarity(query_vector, features_onehot)
    top_n = 5
    sorted_indices = similarities.argsort()[0][-top_n:]
   
    recommended_outfits = ds.iloc[sorted_indices]

    return recommended_outfits.to_dict('records')

@app.route('/recommend', methods=['POST'])
def get_recommendation():
    data = request.json
    gender_product = data.get('gender_product')
    capitalize_gender = gender_product.capitalize()
    city = data.get('city')
    if city in CITY and capitalize_gender in GENDER :
        recommendations = recommend_outfits(capitalize_gender, city)
        res =  jsonify(recommendations)
    else :
        res = jsonify({"status" : "Bad Request" , "messages" : "City or Gender is wrong"}) , 400
    
    return res

@app.route('/hello', methods=['GET'])
def hello_world():
    return "Hello, World!"

@app.route('/city', methods=['GET'])
def get_city():
    return jsonify({"city" : CITY})

@app.route('/gender', methods=['GET'])
def get_gender():
    return jsonify({"gender" : GENDER})

@app.route('/', methods=['GET'])
def main():
    return jsonify({"status" : "success"}) 

if __name__ == '__main__':
    app.run(debug=True)
else :
    gunicorn_app = app
