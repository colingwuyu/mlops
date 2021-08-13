import mlflow

from examples.iris.data_extraction import helper as de_helper
from mlops.utils import convert_json_type
from mlops.components import base_component

_OPS_NAME = "random_dataset_split"

_OPS_DES = """
# Random Dataset Split
This is a dataset random split operation
## Task type
- supervised-learning
## Upstream dependencies
Data extraction
## Parameters
- x_headers:        column names for features in raw data
- y_header:         column name for label in raw data
- test_size:        percentage for test set [0-1.0] (default 0.20)
## Metrics
None
## Artifacts
1. train_test_split.json:
    * 'X_train_ind'     :   x train set index
    * 'X_val_ind'       :   x validation set index
    * 'Y_train_ind'     :   y train set index
    * 'Y_val_ind'       :   y validation set index
## Helper functions
- `split_data(run_id: str, raw_data)`
```python
    from data_extraction import helper as de_helper
    from dataset_split import helper as ds_helper
    raw_data = de_helper.load_raw_data(upstream_ids['data_extraction'])
    X_train, X_test, Y_train, Y_test = ds_helper.split_data(upstream_ids['dataset_split'], raw_data)
```
- `get_label_header(run_id: str)`
```python
    from dataset_split import helper as ds_helper
    label_header = ds_helper.get_label_header(upstream_ids['dataset_split'])
```
"""


@base_component(name=_OPS_NAME, note=_OPS_DES)
def run_func(
    upstream_ids: dict,
    x_headers: list,
    y_header: list,
    test_size: float = 0.20,
    **kwargs
):
    """split data into train and test by using sklearn's train_test_split

    Args:
        upstream_ids (dict): upstream dependent operations with the corresponding mlflow run id
        x_headers (list): x header
        y_headers (list): y heade
        test_size (float, optional): test portion on the total sample. Defaults to 0.20.
    """
    dataset = de_helper.load_raw_data(upstream_ids["data_extraction"])
    X = dataset[x_headers]
    y = dataset[y_header]
    from sklearn.model_selection import train_test_split

    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, y, test_size=test_size, random_state=1
    )
    train_test_split_val = {
        "X_train_ind": list(X_train.index.values),
        "X_val_ind": list(X_validation.index.values),
        "Y_train_ind": list(Y_train.index.values),
        "Y_val_ind": list(Y_validation.index.values),
    }
    convert_json_type(train_test_split_val)
    mlflow.log_dict(train_test_split_val, "train_test_split.json")
