import os
import shutil

import mlflow.tracking

curdir = os.path.dirname(os.path.abspath(__file__))

try:
    client = mlflow.tracking.MlflowClient(
        "http://localhost:5000", "http://localhost:5000"
    )
    client.delete_registered_model("IRIS")
except:
    ...

label_data = f"{curdir}/static/data/label_data/data.csv"
eval_counter = f"{curdir}/static/data/label_data/perf_eval_counter"
upload_data = f"{curdir}/static/data/upload_data/data.csv"
upload_counter = f"{curdir}s/tatic/data/upload_data/data_val_counter"
raw_data = f"{curdir}/static/data/raw_data.csv"
if os.path.exists(label_data):
    os.remove(label_data)
    shutil.copyfile(raw_data, label_data)
if os.path.exists(upload_data):
    os.remove(upload_data)
with open(upload_data, "w") as f:
    f.write("sepal-length,sepal-width,petal-length,petal-width\n")
if os.path.exists(eval_counter):
    os.remove(eval_counter)
if os.path.exists(upload_counter):
    os.remove(upload_counter)

eval_result_folder = f"{curdir}/static/performance_reports"
perf_report_folder = f"{curdir}/templates/performance_reports"
data_report_folder = f"{curdir}/templates/data_reports"
airflow_comm_folder = f"{curdir}/static/data/airflow_comm"
shutil.rmtree(eval_result_folder)
shutil.rmtree(perf_report_folder)
shutil.rmtree(data_report_folder)
shutil.rmtree(airflow_comm_folder)

os.makedirs(eval_result_folder)
os.makedirs(perf_report_folder)
os.makedirs(data_report_folder)
os.makedirs(airflow_comm_folder)

mlflow_db = "/home/colin/mlflow.db"
mlflow_art = "/home/colin/mlflow_artifacts"
try:
    os.remove(mlflow_db)
    shutil.rmtree(mlflow_art)
except:
    ...
