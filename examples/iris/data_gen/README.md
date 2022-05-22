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
