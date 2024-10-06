import os
import pandas as pd
import flask
import google.generativeai as genai
from dotenv import load_dotenv
from flask_cors import CORS
import json
from neuralforecast import NeuralForecast
import typing_extensions as typing
from utils import generate_prompt
from utils import get_rainfall_data
from utils import get_weather_data
from utils import get_year_data
import utils

CORS_ALLOW_ORIGINS = ["https://firebase.com", "https://agricultural-platform.web.app", "https://localhost:5173"]

app = flask.Flask(__name__)
CORS(app, resources={r"/*": {"origins": CORS_ALLOW_ORIGINS}})


load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash-latest")


@app.route("/")
def hello_world():
    return "<h1>Hello, World!</h1>"


@app.get("/test")
@app.get("/test/<name>")
def test(name=None):
    args = flask.request.args
    resp_json = {"args": args, "name": name}
    return resp_json, 200



@app.post("/predict")
def predict():
    rainfall_nf = NeuralForecast.load(path='./model/rain_model')
    temperature_nf = NeuralForecast.load(path='./model/temp_model')
    moist_nf = NeuralForecast.load(path='./model/moist_model')
    wind_speed_nf = NeuralForecast.load(path='./model/wind_speed_model')

    data, error = utils.get_input()
    if not data:
        code, message = error or (400, "Bad request")
        error_json = {"error": {"code": code, "message": message}}
        return error_json, 400
    
    lat, long, crop, language = data

    try:
        rainfall_df = get_rainfall_data(0, long, lat)
        temperature_df = get_weather_data("T2M",long, lat)
        moist_df = get_weather_data("RH2M",long, lat)
        wind_speed_df = get_weather_data("WS2M",long, lat)
    except:
        code, message = error or (400, "internal error")
        error_json = {"error": {"code": code, "message": message}}
        return error_json, 400


    # Check if data is valid 
    if rainfall_df.empty or temperature_df.empty or moist_df.empty or wind_speed_df.empty:
        print("invalid data")
        return 0
    
    # Make a forecast in month format
    
    rainfall_forecast = rainfall_nf.predict(rainfall_df)
    temperature_forecast = temperature_nf.predict(temperature_df)
    moist_forecast = moist_nf.predict(moist_df)
    wind_speed_forecast = wind_speed_nf.predict(wind_speed_df)

    # Format the forecast into a query

    rainfall_data = get_year_data(rainfall_forecast, rainfall_df)
    rainfall_data = rainfall_data.rename(columns={'y': 'Rainfall'})
    temperature_data = get_year_data(temperature_forecast, temperature_df)
    temperature_data = temperature_data.rename(columns={'y': 'Temperature'})
    moist_data = get_year_data(moist_forecast, moist_df)
    moist_data = moist_data.rename(columns={'y': 'Moist'})
    wind_speed_data = get_year_data(wind_speed_forecast, wind_speed_df)
    wind_speed_data = wind_speed_data.rename(columns={'y': 'Wind Speed'})

    # Start with the initial dataframe
    all_data = rainfall_data.copy()

    # List of dataframes to merge
    dataframes_to_merge = [temperature_data, moist_data, wind_speed_data]

    for df in dataframes_to_merge:
        # Identify columns that are not in all_data
        columns_to_use = df.columns.difference(all_data.columns).tolist() + ['ds']
        # Merge using only non-duplicate columns and 'ds' as the key
        all_data = pd.merge(all_data, df[columns_to_use], on='ds')

    # Rename 'ds' column to 'Date'
    all_data = all_data.rename(columns={'ds': 'Date'})

    # Ensure 'Rainfall' values are non-negative
    all_data.loc[all_data['Rainfall'] < 0, 'Rainfall'] = 0.0


    prompt = generate_prompt(all_data, crop, language)
    print(prompt)
    class Report(typing.TypedDict):
        action_you_should_take_immediately: str
        water_resoure_management: str
        irrigation_management: str
        temperature_management: str
        pest_and_disease_management: str
        soil_and_nutrient_management: str
        crop_selection_and_rotation: str
        harvest_and_post_harvest_management: str
    raw_response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(response_mime_type="application/json", response_schema=Report),
    )
    response = json.loads(raw_response.text)
    return response, 200


@app.get("/genai/test")
def genai_test():
    query = flask.request.args["query"]
    if not query:
        return {"error": "query is required"}, 400
    response = model.generate_content(query)
    return response.text, 200


if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")
