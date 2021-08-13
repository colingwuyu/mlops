import os
import logging
import tempfile
import shutil
from typing import List

import pandas as pd
import mlflow

from mlops.components import pipeline_init_component


logger = logging.getLogger("IRIS")


_OPS_NAME = "url_data_extraction"

_OPS_DES = """
# URL Data Extraction
This is a data extraction operation by URL downloading
## Task type
- any
## Upstream dependencies
None
## Parameters
- url:                  data url
- headers:              column names (default is to take the first row as column names)
## Metrics
None
## Artifacts
1. raw_data.csv:        raw data file
2. data_info.json:      data_shape and headers info
## Helper functions
- `load_raw_data(run_id: str)`
```pythn
    from data_extraction import helper as de_helper
    raw_data = de_helper.load_raw_data(run_id=upstream_ids['data_extraction'])
```
"""


@pipeline_init_component(name=_OPS_NAME, note=_OPS_DES)
def run_func(url: str, headers: List[str] = None, **kwargs):
    # Load dataset
    artifact_dir = tempfile.mkdtemp()
    dataset = pd.read_csv(url, names=headers)
    dataset.to_csv(os.path.join(artifact_dir, "raw_data.csv"), index=False)
    mlflow.log_dict(
        {"data_shape": dataset.shape, "headers": list(dataset.columns.values)},
        "data_info.json",
    )
    mlflow.log_artifacts(artifact_dir)
    shutil.rmtree(artifact_dir)
