import os
import random
import json

import cloudpickle
import pandas as pd
import numpy as np
import requests

TRAIN_DATA = "static/data/raw_data.csv"
INPUT_DATA = "static/data/inputs.csv"
GEN_SIZE = 50
FEATURES = ["sepal-length", "sepal-width", "petal-length", "petal-width"]
LABEL = "species"


def _writeCSVToTextFile(examples):
    data_location = INPUT_DATA
    with open(data_location, "w") as writer:
        for example in examples:
            writer.write(example + "\n")
    return data_location


def load_logreg():
    with open("logreg.pkl", "rb") as logreg_io:
        model = cloudpickle.load(logreg_io)
    with open("scaler.pkl", "rb") as scaler_io:
        scaler = cloudpickle.load(scaler_io)
    return scaler, model


def gen_data_drift(drift_mean=0.5, drift_std=0.2):
    train_data = pd.read_csv(TRAIN_DATA)
    inputs = train_data[FEATURES]
    index = random.sample(range(inputs.shape[0]), GEN_SIZE)
    inputs = inputs.iloc[index]
    noises = np.random.normal(loc=drift_mean, scale=drift_std, size=inputs.shape)
    print(f"Mean value:\n{inputs.mean()}")
    print(f"STD value:\n{inputs.std()}")
    inputs = inputs + noises
    print("After adding noises:")
    print(f"Mean value:\n{inputs.mean()}")
    print(f"STD value:\n{inputs.std()}")
    return inputs.to_json(orient="split")


if __name__ == "__main__":
    input_data = gen_data_drift()
    response = requests.post(
        "http://172.28.23.175:3000/inference",
        json=input_data,
        params={"prediction_key": "classes"},
    )
    response_json = json.loads(response.text)
    anomaly = response_json["Anomaly"]
    prediction = response_json["classes"]
    print(prediction)
