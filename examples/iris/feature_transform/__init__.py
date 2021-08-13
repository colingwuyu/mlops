import os
import tempfile
import shutil

import mlflow
import cloudpickle

from mlops.components import base_component
from examples.iris.data_gen import helper as dg_helper
from examples.iris.data_gen import OPS_NAME as DATA_GEN_OPS_NAME


OPS_NAME = "feature_transform"

OPS_DES = """
# Feature Transformation
This is a feature transformation operation.
This operation adopts several common feature scaling transformation approaches, such as:
- min-max
- standardization
## Task type
- any
## Upstream dependencies
- Data extraction
- Dataset split
## Paramter
- transform_approach: transformation approache (minmax/std)
## Metrics
None
## Artifacts
1. scaler.pkl
## Helper functions
- `load_scaler(run_id: str)`
"""

PARAM_TRANSFORM_APPROACH = "transform_approach"
PARAM_VAL_MIN_MAX = "min-max"
PARAM_VAL_STD = "standardization"

ARTIFACT_SCALER = "scaler.pkl"


@base_component(name=OPS_NAME, note=OPS_DES)
def run_func(upstream_ids: dict, **kwargs):
    import sklearn.preprocessing as processor

    # Load parameters
    transform_approach = kwargs[PARAM_TRANSFORM_APPROACH]

    data_gen_id = upstream_ids[DATA_GEN_OPS_NAME]
    X_train = dg_helper.load_train_X(data_gen_id)
    artifact_dir = tempfile.mkdtemp()

    if transform_approach == PARAM_VAL_MIN_MAX:
        scaler = processor.MinMaxScaler()
    elif transform_approach == PARAM_VAL_STD:
        scaler = processor.StandardScaler()
    else:
        raise NotImplementedError(
            "transform_approach %s is not implemented." % transform_approach
        )

    scaler.fit(X_train)
    cloudpickle.dump(scaler, open(os.path.join(artifact_dir, ARTIFACT_SCALER), "wb"))
    mlflow.log_artifacts(artifact_dir)
    shutil.rmtree(artifact_dir)
