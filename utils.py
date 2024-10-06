import flask
import pandas as pd
import requests
import climateserv
from datetime import date
from dateutil.relativedelta import relativedelta

def get_input():
    if flask.request.is_json:
        data = flask.request.get_json()
        if not data:
            return None, (400, "Wrong Format")
    else:
        return None, (400, "No json")

    lat = data["lat"]
    long = data["long"]
    crop = data["crop"]
    language = data["language"]

    # Load data based on latitude and longitude
    resolution = 0.25
    lat = round(lat / resolution) * resolution
    long = round(long / resolution) * resolution
    data = lat, long, crop, language

    return data, None

def query_firestore_by_value(db, collection_name, field_name, value):
    
    # Perform the query using the 'where' clause
    docs = db.collection(collection_name).where(field_name, '==', value).stream()

    # Convert Firestore documents to a list of dictionaries
    data = []
    for doc in docs:
        doc_dict = doc.to_dict()  # Convert document to dictionary
        data.append(doc_dict)     # Append dictionary to the list

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)

    return df

def get_year_data(forecast, past):
    forecast = forecast.rename(columns = {"AutoNLinear": "y"})
    base_date = forecast["ds"].min()
    past_date = base_date - relativedelta(months=6)
    past_six_month = past[past["ds"] >= past_date]
    df = pd.concat([past_six_month, forecast], ignore_index=True)
    return df

def generate_prompt(df, crop, language):
    today = date.today()
    prompt = f"""You are an expert in agriculture.
Today's date is {today}.
The crop type is {crop}. 
Write a easy to understand document in {language} aiming to suggesting risk mitigation strategies and implement sustainable farming practices based on the following 18 months of data:
Give concrete advices, tell us what to do rather than explanation.(example: action_you_should_take_immediately: rush in the harvest)
Write in columnar format. Be concise about the strategies. Your decision is very important and may affect people's lives.\n"""
    for index, row in df.iterrows():
        prompt += (
            f"Date: {row['Date']} | Averge_Temperature: {row['Temperature']}°C | "
            f"Rainfall_of_the_month: {row['Rainfall']*30} mm | "
            f"Relative humidity: {row['Moist']} % | "
            f"Wind Speed: {row['Wind Speed']} m/s \n"
        )
    return prompt

def get_rainfall_data(data_type,long, lat):
    GeometryCoords = [[long-.01,lat+.01],[long+.01, lat+.01],
                    [long+.01, lat-.01],[long-.01,lat-.01],[long-.01,lat+.01]]
    
    resolution = 0.1
    lat = round (lat / resolution) * resolution
    long = round (long/ resolution) * resolution
    DatasetType = data_type
    OperationType = 'Average'
    EarliestDate = '08/31/2023'
    LatestDate = '08/31/2024'
    SeasonalEnsemble = '' # Leave empty when using the new integer dataset IDs
    SeasonalVariable = '' # Leave empty when using the new integer dataset IDs
    Outfile = 'memory_object'

    raw_data = climateserv.api.request_data(DatasetType, OperationType, 
                EarliestDate, LatestDate,GeometryCoords, 
                SeasonalEnsemble, SeasonalVariable,Outfile)
    avg =  [long['value']['avg'] for long in raw_data['data']]
    date = [long['date'] for long in raw_data['data']]
    unique_id = [0] * len(avg)
    df = pd.DataFrame({"unique_id": unique_id, "y": avg, "ds": date})
    df["ds"] = pd.to_datetime(df["ds"])

    invalid_index = df[df["y"] == -9999].index
    df.drop(invalid_index, inplace = True)
    df = df.resample('M', on='ds').mean().reset_index()
    df['ds'] = df['ds'].apply(lambda x: x.replace(day=1))
    return df   

def get_weather_data(parameters, long, lat):
    start_date = '20230101'
    end_date = '20240831'
    parameters = parameters
    url = (
        'https://power.larc.nasa.gov/api/temporal/daily/point'
        f'?start={start_date}'
        f'&end={end_date}'
        f'&latitude={lat}'
        f'&longitude={long}'
        f'&parameters={parameters}'
        '&community=AG'
        '&format=JSON'
    )
    # Make the API request
    response = requests.get(url)
    data = response.json()

    # Extract temperature data
    temperature_data = data['properties']['parameter'][parameters]
    dates = list(temperature_data.keys())
    temperatures = list(temperature_data.values())
    unique_id = [0] * len(dates)
    df = pd.DataFrame({"unique_id": unique_id, "y": temperatures, "ds": dates})
    invalid_index = df[df["y"] == -999].index
    df.drop(invalid_index, inplace = True)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.resample('M', on='ds').mean().reset_index()
    df['ds'] = df['ds'].apply(lambda x: x.replace(day=1))
    return df