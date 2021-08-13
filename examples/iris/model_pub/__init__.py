import shutil
import tempfile
import os

import mlflow

from mlops.components import base_component, ARG_MLFLOW_RUN_ID
from mlops.utils.mlflowutils import MlflowUtils
import mlops.serving.model as mlops_model
import mlops.model_analysis as mlops_ma
from examples.iris.data_gen import OPS_NAME as DATA_GEN_OPS_NAME
from examples.iris.data_gen import helper as dg_helper
from examples.iris.model_eval import OPS_NAME as MODEL_EVAL_OPS_NAME
from examples.iris.model_eval import helper as me_helper

OPS_NAME = "model_pub"
OPS_DES = """
# Model Publisher
This is to publish a model into registry for serving
## Task type
- any
## Upstream dependencies
- Data Extraction
- Dataset Split
- Feature Transformation
- Data Validation
- Feature Transformation
- Model Evaluation
"""

ARTIFACT_MODEL = "mlops_model"
PARAM_MODEL_REGISTRY_NAME = "model_registry_name"
TAG_MODEL_PUB = "model_pub"
TAG_VAL_MODEL_PUB_SUCCESS = "success"
TAG_VAL_MODEL_PUB_FAIL = "fail"


@base_component(name=OPS_NAME, note=OPS_DES)
def run_func(upstream_ids: dict, **kwargs):
    # load parameters
    model_registry_name = kwargs[PARAM_MODEL_REGISTRY_NAME]
    # load upstream run ids
    model_eval_run_id = upstream_ids[MODEL_EVAL_OPS_NAME]

    loaded_model: mlops_model.MlopsLoadModel = me_helper.load_model(model_eval_run_id)

    # evaluate production model
    cur_run_id = kwargs[ARG_MLFLOW_RUN_ID].info.run_id
    if not MlflowUtils.mlflow_client.search_registered_models(
        f"name='{model_registry_name}'"
    ):
        # create new model
        MlflowUtils.mlflow_client.create_registered_model(model_registry_name)

    artifact_dir = tempfile.mkdtemp()

    if MlflowUtils.mlflow_client.get_latest_versions(
        model_registry_name, ["Production"]
    ):
        # existing production version in serving
        prod_model = mlops_model.load_model(
            model_uri=f"models:/{model_registry_name}/Production"
        )
        prod_ver = int(
            MlflowUtils.get_latest_versions(model_registry_name, ["Production"])[
                0
            ].version
        )
        data_gen_run_id = upstream_ids[DATA_GEN_OPS_NAME]
        eval_X = dg_helper.load_test_X(data_gen_run_id)
        eval_y = dg_helper.load_test_y(data_gen_run_id)
        eval_data = eval_X.join(eval_y)

        retrain_eval_result_dir = os.path.join(
            artifact_dir, "retrain_model_eval_report"
        )
        mlops_ma.run_model_analysis(
            model=loaded_model,
            data=eval_data,
            output_path=retrain_eval_result_dir,
            save_report=True,
        )
        prod_eval_result_dir = os.path.join(artifact_dir, "prod_model_eval_report")
        mlops_ma.run_model_analysis(
            model=prod_model,
            data=eval_X.join(eval_y),
            eval_config=loaded_model.eval_config,
            output_path=prod_eval_result_dir,
            save_report=True,
        )
        model_result = mlops_ma.select_model(
            eval_results=[retrain_eval_result_dir, prod_eval_result_dir],
            eval_config=loaded_model.eval_config,
        )
        mlflow.log_artifacts(artifact_dir)
        shutil.rmtree(artifact_dir)

        if not model_result.model_spec.name:
            # retrained model beat the production version
            loaded_model.eval_config.model_spec.model_ver = f"Version {prod_ver + 1}"
            loaded_model.eval_config.model_spec.name = model_registry_name
            conda_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "../conda.yaml"
            )
            mlops_model.log_model(
                ARTIFACT_MODEL,
                loaded_model,
                conda_env=conda_file,
            )
            model_ver = MlflowUtils.mlflow_client.create_model_version(
                name=model_registry_name,
                source=f"runs:/{cur_run_id}/{ARTIFACT_MODEL}",
                run_id=cur_run_id,
            )
            assert int(model_ver.version) == prod_ver + 1

            MlflowUtils.mlflow_client.transition_model_version_stage(
                name=model_registry_name,
                version=int(model_ver.version),
                stage="Production",
            )
            mlflow.set_tag(TAG_MODEL_PUB, TAG_VAL_MODEL_PUB_SUCCESS)
        else:
            # retrained model cannot beat the production version
            mlflow.set_tag(TAG_MODEL_PUB, TAG_VAL_MODEL_PUB_FAIL)
    else:
        # not exist production version in serving
        # register the model and push into production
        loaded_model.eval_config.model_spec.model_ver = f"Version {1}"
        loaded_model.eval_config.model_spec.name = model_registry_name
        conda_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../conda.yaml"
        )
        mlops_model.log_model(
            ARTIFACT_MODEL,
            loaded_model,
            conda_env=conda_file,
        )
        model_ver = MlflowUtils.mlflow_client.create_model_version(
            name=model_registry_name,
            source=f"runs:/{cur_run_id}/{ARTIFACT_MODEL}",
            run_id=cur_run_id,
        )
        assert int(model_ver.version) == 1
        MlflowUtils.mlflow_client.transition_model_version_stage(
            name=model_registry_name, version=int(model_ver.version), stage="Production"
        )
        mlflow.set_tag(TAG_MODEL_PUB, TAG_VAL_MODEL_PUB_SUCCESS)
