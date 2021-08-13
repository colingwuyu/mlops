import os
from os.path import join as pjoin
import pathlib

from flask import Flask, request, abort, send_file, render_template, Response
import numpy as np
import pandas as pd
from requests.models import HTTPError

from mlops.config import Config as conf
from mlops.utils.sysutils import path_splitall
from mlops.utils.collectionutils import convert_json_type
from mlops.model_analysis import FILE_EVAL_RESULT, view_report
from examples.iris.serving.helper import load_model

conf.load("config_{}.yaml".format(os.environ.get("IRIS_ENV", "local")))
data_conf = conf.settings.data_path

DATA_STORE = pjoin(data_conf.data_store)
LABEL_DATA = pjoin(DATA_STORE, data_conf.label_data, "data.csv")
INPUT_DATA = pjoin(DATA_STORE, data_conf.upload_data, "data.csv")

model = load_model(stage="Production")

app = Flask(__name__)


@app.route("/inference", methods=["POST"])
def inference():
    global model
    if model:
        data_inputs = pd.read_json(request.json, orient="split")
        prediction_key = request.args.get("prediction_key")
        # input_file = pjoin(DATA_INBOUND, "inputs.csv")
        prediction = model.predict(data_inputs, prediction_key)
        if not os.path.isfile(INPUT_DATA):
            data_inputs.to_csv(INPUT_DATA, index=False)
        else:
            data_inputs.to_csv(INPUT_DATA, mode="a", index=False, header=False)
        convert_json_type(prediction)
        return prediction
    else:
        return "Model is not available in registry"


@app.route("/upload_label_data", methods=["POST"])
def upload_label_data():
    label_data = pd.read_json(request.json, orient="split")
    if not os.path.isfile(LABEL_DATA):
        label_data.to_csv(LABEL_DATA, index=False)
    else:
        label_data.to_csv(LABEL_DATA, mode="a", index=False, header=False)
    return "Label data uploaded."


@app.route("/data")
def get_data():
    with open("static/data/label_data/data.csv") as fp:
        csv = fp.read()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=data.csv"},
    )


@app.route("/original_data")
def get_original_data():
    with open("static/data/raw_data.csv") as fp:
        csv = fp.read()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=original_data.csv"},
    )


@app.route("/performance_reports", defaults={"req_path": ""})
@app.route("/performance_reports/<path:req_path>")
def perf_report_listing(req_path):
    BASE_DIR = "templates/performance_reports"

    # Joining the base and the requested path
    abs_path = os.path.join(BASE_DIR, req_path)

    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    for file_name in os.listdir(abs_path):
        if file_name == "report.html":
            return render_template(
                os.path.join(*path_splitall(os.path.join(abs_path, file_name))[1:])
            )

    # Show directory contents
    files = []
    creation_times = []
    for subdir in os.walk(abs_path):
        if "report.html" in subdir[2]:
            creation_times.append(pathlib.Path(subdir[0]).stat().st_ctime)
            files.append(os.path.relpath(subdir[0], BASE_DIR))
    files = [x for _, x in sorted(zip(creation_times, files))]
    return render_template("files.html", files=files, header="Performance Reports")


@app.route("/data_reports", defaults={"req_path": ""})
@app.route("/data_reports/<path:req_path>")
def data_report_listing(req_path):
    BASE_DIR = "templates/data_reports"

    # Joining the base and the requested path
    abs_path = os.path.join(BASE_DIR, req_path)

    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    for file_name in os.listdir(abs_path):
        if file_name == "report.html":
            return render_template(
                os.path.join(*path_splitall(os.path.join(abs_path, file_name))[1:])
            )

    # Show directory contents
    files = []
    creation_times = []
    for subdir in os.walk(abs_path):
        if "report.html" in subdir[2]:
            creation_times.append(pathlib.Path(subdir[0]).stat().st_ctime)
            files.append(os.path.relpath(subdir[0], BASE_DIR))
    files = [x for _, x in sorted(zip(creation_times, files))]
    return render_template("files.html", files=files, header="Data Reports")


@app.route("/h")
def h():
    return "h"


@app.route("/reload_model")
def reload_model():
    global model
    model = load_model(stage="Production")
    if model:
        return f"Model {model.model_version} Reloaded"
    else:
        return "Model is not available in registry"


@app.route("/model_ver")
def model_ver():
    if model:
        return model.model_version
    else:
        return "Model is not loaded yet."


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
