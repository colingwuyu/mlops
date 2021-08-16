import os
import logging
import tempfile
import shutil

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

from mlops.components import pipeline_init_component


logger = logging.getLogger("IRIS")


OPS_NAME = "data_gen"
OPS_DES = """
# Data Generation 
This is a data generation operation
## Task type
- any
## Upstream dependencies
None
## Parameters
- url:                  data url
- x_headers (list):     x header
- y_headers (str):     y header
- test_size:            percentage for test set [0-1.0] (default 0.20)
## Metrics
None
## Artifacts
1. train_X.csv:        train dataset file of X
2. test_X.csv:         test dataset file of X
3. train_y.csv:        train dataset file of y
4. test_y.csv:        test dataeset file of y
## Helper functions
- `load_train_X(run_id: str)`
- `load_train_y(run_id: str)`
- `load_test_X(run_id: str)`
- `load_test_y(run_id: str)`
- `get_label_header(run_id: str)`
- `get_feature_header(run_id: str)`
"""
PARAM_X_HEADERS = "x_headers"
PARAM_Y_HEADER = "y_header"
PARAM_TEST_SIZE = "test_size"
PARAM_URL = "url"

ARTIFACT_TRAIN_X = "train_X.csv"
ARTIFACT_TRAIN_Y = "train_y.csv"
ARTIFACT_TEST_X = "test_X.csv"
ARTIFACT_TEST_Y = "test_y.csv"


@pipeline_init_component(name=OPS_NAME, note=OPS_DES)
def run_func(**kwargs):
    # Load parameters
    url = kwargs[PARAM_URL]
    test_size = kwargs.get(PARAM_TEST_SIZE, 0.20)
    x_headers = kwargs[PARAM_X_HEADERS]
    y_header = kwargs[PARAM_Y_HEADER]

    artifact_dir = tempfile.mkdtemp()

    headers = x_headers + [y_header]
    first_row = pd.read_csv(url, header=None, nrows=1)
    if first_row.values[0][0] == x_headers[0]:
        # has header already
        dataset = pd.read_csv(url)
    else:
        dataset = pd.read_csv(url, names=headers)

    X = dataset[x_headers]
    y = dataset[y_header]

    train_X, val_X, train_y, val_y = train_test_split(
        X, y, test_size=test_size, random_state=1
    )

    train_X.to_csv(os.path.join(artifact_dir, ARTIFACT_TRAIN_X), index=False)
    val_X.to_csv(os.path.join(artifact_dir, ARTIFACT_TEST_X), index=False)
    train_y.to_csv(os.path.join(artifact_dir, ARTIFACT_TRAIN_Y), index=False)
    val_y.to_csv(os.path.join(artifact_dir, ARTIFACT_TEST_Y), index=False)
    mlflow.log_artifacts(artifact_dir)
    shutil.rmtree(artifact_dir)
