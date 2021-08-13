import logging
import pickle
import uuid
import os
import importlib
import yaml
from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Union, List, Dict, Text
from os import mkdir
from copy import deepcopy
import posixpath
import shutil

import cloudpickle
import numpy as np
import pandas
import mlflow.pyfunc
from mlflow.pyfunc.model import get_default_conda_env
from mlflow.models import Model as Mlflow_Model, ModelSignature, ModelInputExample
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import PYTHON_VERSION, get_major_minor_py_version
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.file_utils import TempDir, _copy_file_or_tree
from mlflow.exceptions import MlflowException
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils.display_util import get_anomalies_dataframe

import mlops.serving
from mlops.types import RegPath, ModelPath, ModelComponents, ModelInput, ModelOutput
import mlops.model_analysis as mlops_ma

FLAVOR_NAME = "mlops_model"
CODE = "code"
DATA = "data"
ENV = "env"
PY_VERSION = "python_version"

CONFIG_KEY_MLOPS_MODEL = "mlops_model"
CONFIG_KEY_CLOUDPICKLE_VERSION = "cloudpickle_version"
CONFIG_KEY_MODEL_COMP = "model_components"
CONFIG_KEY_ARTIFACTS = "artifacts"
CONFIG_KEY_RELATIVE_PATH = "path"

MODEL_COMP_SCALER = "scaler"
MODEL_COMP_STATS = "trainset_stats"
MODEL_COMP_SCHEMA = "schema"
MODEL_COMP_PERF_EVAL_CONFIG = "eval_config"
MODEL_COMP_MODEL = "model"

PREDICTION_KEY_ANOMALY = "Anomaly"

_logger = logging.getLogger(__name__)


def add_to_model(model, code=None, data=None, env=None, **kwargs):
    """
    Add a ``mlops_model`` spec to the model configuration.

    Defines ``mlops_model`` configuration schema. Caller can use this to create a valid ``mlops_model`` model
    flavor out of an existing directory structure.

    NOTE:

        All paths are relative to the exported model root directory.

    :param model: Existing MLflow Model.
    :param loader_module: The module to be used to load the model.
    :param env: Conda environment.
    :param kwargs: Additional key-value pairs to include in the ``mlops_model`` flavor specification.
                   Values must be YAML-serializable.
    :return: Updated model configuration.
    """
    parms = deepcopy(kwargs)
    parms[PY_VERSION] = PYTHON_VERSION
    if code:
        parms[CODE] = code
    if data:
        parms[DATA] = data
    if env:
        parms[ENV] = env
    return model.add_flavor(FLAVOR_NAME, **parms)


class MlopsLoadModel(object):
    """
    MLOps model

    Wrapper around model implementation and model metadata. This class is constructed and returned from
    `load_model() <mlops.serving.model.load_model>

    ``model_impl`` can be any Python object that implements the interface including methods:
        - predict
        - data_validate
        - evaluate
    and is returned by invoking the model's ``loader_module``.

    ``model_meta`` contains model metadata loaded from the MLmodel file.
    """

    def __init__(self, model_meta: Mlflow_Model, model_impl: Any):
        if not hasattr(model_impl, "predict"):
            raise MlflowException(
                "Model implementation is missing required predict method."
            )
        if not hasattr(model_impl, "evaluate"):
            raise MlflowException(
                "Model implementation is missing required evaluate method."
            )
        if not model_meta:
            raise MlflowException("Model is missing metadata.")
        self._model_meta = model_meta
        self._model_impl = model_impl

    def predict(
        self, data: ModelInput, predict_keys: Optional[List[Text]] = None
    ) -> ModelOutput:
        """
        Generate model predictions.

        If the model contains signature, enforce the input schema first before calling the model
        implementation with the sanitized input. If the pyfunc model does not include model schema,
        the input is passed to the model implementation as is. See `Model Signature Enforcement
        <https://www.mlflow.org/docs/latest/models.html#signature-enforcement>`_ for more details."

        :param data: Model input as one of pandas.DataFrame, numpy.ndarray, or
                     Dict[str, numpy.ndarray]
        :return: Model predictions as one of pandas.DataFrame, pandas.Series, numpy.ndarray or list.
        """
        return self._model_impl.predict(data=data, predict_keys=predict_keys)

    @property
    def metadata(self):
        """Model metadata."""
        if self._model_meta is None:
            raise MlflowException("Model is missing metadata.")
        return self._model_meta

    @property
    def eval_config(self):
        return self._model_impl.eval_config

    @property
    def data_schema(self):
        return self._model_impl.data_schema

    @property
    def trainset_stats(self):
        return self._model_impl.trainset_stats

    @property
    def model_version(self):
        return self._model_impl.model_version

    def __repr__(self):
        # model_meta is the flavor of MLmodel
        info = {}
        if self._model_meta is not None:
            if (
                hasattr(self._model_meta, "run_id")
                and self._model_meta.run_id is not None
            ):
                info["run_id"] = self._model_meta.run_id
            if (
                hasattr(self._model_meta, "artifact_path")
                and self._model_meta.artifact_path is not None
            ):
                info["artifact_path"] = self._model_meta.artifact_path
            info["flavor"] = self._model_meta.flavors[FLAVOR_NAME]["loader_module"]
        return yaml.safe_dump({"mlops.loaded_model": info}, default_flow_style=False)

    def evaluate(self, labeled_data: ModelInputExample):
        return self._model_impl.evaluate(labeled_data)


class MlopsModelContext(object):
    def __init__(self, artifacts):
        self._artifacts = artifacts

    @property
    def artifacts(self):
        return self._artifacts


class MlopsModel(object):
    """
    Represents a generic Python model that evaluates inputs and produces API-compatible outputs.
    By subclassing :class:`~PythonModel`, users can create customized MLflow models with the
    "python_function" ("pyfunc") flavor, leveraging custom inference logic and artifact
    dependencies.
    """

    def __init__(self, model_comps: ModelComponents) -> None:
        self._scaler = model_comps[MODEL_COMP_SCALER]
        self._model = model_comps[MODEL_COMP_MODEL]
        self._trainset_stats = model_comps[MODEL_COMP_STATS]
        self._data_schema = model_comps[MODEL_COMP_SCHEMA]
        self._eval_config = model_comps.get(MODEL_COMP_PERF_EVAL_CONFIG)

    def load_context(self, context: MlopsModelContext):
        """[summary]

        Args:
            context ([type]): [description]
        """
        ...

    @property
    def eval_config(self):
        return self._eval_config

    @property
    def data_schema(self):
        return self._data_schema

    @property
    def trainset_stats(self):
        return self._trainset_stats

    @property
    def model_version(self):
        return self.eval_config.model_spec.model_ver

    def predict(
        self, data: ModelInput, predict_keys: Optional[List[Text]] = None
    ) -> ModelOutput:
        """
        Evaluates a pyfunc-compatible input and produces a pyfunc-compatible output.
        For more information about the pyfunc input/output API, see the :ref:`pyfunc-inference-api`.

        :param model_input: A pyfunc-compatible input for the model to evaluate.
        """
        # validate data stats
        serving_stats = tfdv.generate_statistics_from_dataframe(data)
        serving_stats_anomalies = tfdv.validate_statistics(
            serving_stats, self._data_schema, environment="SERVING"
        )
        anomalies_df = get_anomalies_dataframe(serving_stats_anomalies)
        scaled_inputs = self._scaler.transform(data)
        predictions = self._model.predict(scaled_inputs, predict_keys)
        predictions[PREDICTION_KEY_ANOMALY] = anomalies_df.to_dict(orient="split")
        return predictions

    def evaluate(self, labeled_data: ModelInputExample):
        ...


def load_model(model_uri: ModelPath, suppress_warnings: bool = True) -> MlopsLoadModel:
    """
    Load a Mlops model.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.
    :param suppress_warnings: If ``True``, non-fatal warning messages associated with the model
                              loading process will be suppressed. If ``False``, these warning
                              messages will be emitted.
    """
    local_path = _download_artifact_from_uri(artifact_uri=model_uri)
    model_meta = Mlflow_Model.load(os.path.join(local_path, MLMODEL_FILE_NAME))

    conf = model_meta.flavors.get(FLAVOR_NAME)
    if conf is None:
        raise MlflowException(
            'Model does not have the "{flavor_name}" flavor'.format(
                flavor_name=FLAVOR_NAME
            ),
            RESOURCE_DOES_NOT_EXIST,
        )
    model_py_version = conf.get(PY_VERSION)
    if not suppress_warnings:
        # check python version
        _warn_potentially_incompatible_py_version_if_necessary(
            model_py_version=model_py_version
        )
    if CODE in conf and conf[CODE]:
        # add dependent code into path
        code_path = os.path.join(local_path, conf[CODE])
        mlflow.pyfunc.utils._add_code_to_system_path(code_path=code_path)
    model_impl = _load_mlops_model(local_path)
    return MlopsLoadModel(model_meta=model_meta, model_impl=model_impl)


def _load_mlops_model(model_path: ModelPath):
    mlops_model_config = _get_flavor_configuration(
        model_path=model_path, flavor_name=FLAVOR_NAME
    )

    python_model_cloudpickle_version = mlops_model_config.get(
        CONFIG_KEY_CLOUDPICKLE_VERSION, None
    )
    if python_model_cloudpickle_version is None:
        _logger.warning(
            "The version of CloudPickle used to save the model could not be found in the MLmodel"
            " configuration"
        )
    elif python_model_cloudpickle_version != cloudpickle.__version__:
        # CloudPickle does not have a well-defined cross-version compatibility policy. Micro version
        # releases have been known to cause incompatibilities. Therefore, we match on the full
        # library version
        _logger.warning(
            "The version of CloudPickle that was used to save the model, `CloudPickle %s`, differs"
            " from the version of CloudPickle that is currently running, `CloudPickle %s`, and may"
            " be incompatible",
            python_model_cloudpickle_version,
            cloudpickle.__version__,
        )

    # create model
    model_comps = {}
    for saved_model_comp_name, saved_model_comp_info in mlops_model_config.get(
        CONFIG_KEY_MODEL_COMP, {}
    ).items():
        if saved_model_comp_info[CONFIG_KEY_RELATIVE_PATH]:
            model_comps[saved_model_comp_name] = _load_model_comp(
                saved_model_comp_name,
                os.path.join(
                    model_path, saved_model_comp_info[CONFIG_KEY_RELATIVE_PATH]
                ),
            )
        else:
            model_comps[saved_model_comp_name] = None

    mlops_model = MlopsModel(model_comps=model_comps)

    # TODO: can add data into context
    artifacts = {}
    for saved_artifact_name, saved_artifact_info in mlops_model_config.get(
        CONFIG_KEY_ARTIFACTS, {}
    ).items():
        artifacts[saved_artifact_name] = os.path.join(
            model_path, saved_artifact_info[CONFIG_KEY_RELATIVE_PATH]
        )

    context = MlopsModelContext(artifacts=artifacts)
    mlops_model.load_context(context=context)
    return mlops_model


def _load_model_comp(model_comp_name, model_comp_path):
    if model_comp_name in [MODEL_COMP_MODEL, MODEL_COMP_SCALER]:
        with open(model_comp_path, "rb") as f:
            return cloudpickle.load(f)
    if model_comp_name == MODEL_COMP_SCHEMA:
        return tfdv.load_schema_text(model_comp_path)
    if model_comp_name == MODEL_COMP_STATS:
        return tfdv.load_statistics(model_comp_path)
    if model_comp_name == MODEL_COMP_PERF_EVAL_CONFIG:
        if model_comp_path:
            return mlops_ma.load_eval_config_text(model_comp_path)
        else:
            return
    raise NotImplementedError()


def _warn_potentially_incompatible_py_version_if_necessary(model_py_version=None):
    """
    Compares the version of Python that was used to save a given model with the version
    of Python that is currently running. If a major or minor version difference is detected,
    logs an appropriate warning.
    """
    if model_py_version is None:
        _logger.warning(
            "The specified model does not have a specified Python version. It may be"
            " incompatible with the version of Python that is currently running: Python %s",
            PYTHON_VERSION,
        )
    elif get_major_minor_py_version(model_py_version) != get_major_minor_py_version(
        PYTHON_VERSION
    ):
        _logger.warning(
            "The version of Python that the model was saved in, `Python %s`, differs"
            " from the version of Python that is currently running, `Python %s`,"
            " and may be incompatible",
            model_py_version,
            PYTHON_VERSION,
        )


def save_model(
    path: ModelPath,
    model: Union[ModelComponents, MlopsLoadModel],
    eval_config: Optional[mlops_ma.EvalConfig] = None,
    code_paths: Optional[RegPath] = None,
    artifacts: Optional[Dict[Text, RegPath]] = None,
    conda_env: Optional[Union[Text, Dict]] = None,
    mlflow_model: Optional[Mlflow_Model] = None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
):
    """
    Save model to a path on the local file system

    Args:
        path (ModelPath): save local path for the model
        model (ModelComponents): components to be saved for the model
        conda_env (Union[Text, Dict], optional): Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model. The
                      following is an *example* dictionary representation of a Conda environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'pytorch=0.4.1',
                                'torchvision=0.2.1'
                            ]
                        }
        mlflow_model ([type], optional): [description]. Defaults to None.
        signature (ModelSignature, optional): [description]. Defaults to None.
        input_example (ModelInputExample, optional): [description]. Defaults to None.

    Raises:
        MlflowException: [description]
    """
    # prepare model save path
    if os.path.exists(path):
        raise MlflowException(
            message="Path '{}' already exists".format(path),
            error_code=RESOURCE_ALREADY_EXISTS,
        )
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Mlflow_Model()

    # prepare signature and input example
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    # prepare conda.yaml
    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if isinstance(model, MlopsLoadModel):
        model_impl: MlopsModel = model._model_impl
        model = {
            MODEL_COMP_SCALER: model_impl._scaler,
            MODEL_COMP_MODEL: model_impl._model,
            MODEL_COMP_STATS: model_impl._trainset_stats,
            MODEL_COMP_SCHEMA: model_impl._data_schema,
            MODEL_COMP_PERF_EVAL_CONFIG: model_impl._eval_config,
        }

    if MODEL_COMP_PERF_EVAL_CONFIG not in model:
        model[MODEL_COMP_PERF_EVAL_CONFIG] = None

    if eval_config:
        model[MODEL_COMP_PERF_EVAL_CONFIG] = eval_config

    custom_model_config_kwargs = {
        CONFIG_KEY_CLOUDPICKLE_VERSION: cloudpickle.__version__,
    }

    saved_model_comp_config = {}
    with TempDir() as tmp_model_comp_dir:
        saved_model_comp_dir_subpath = "model_comp"
        for model_comp_name, model_comp in model.items():
            model_comp_path = _save_model_comp(
                model_comp, model_comp_name, output_path=tmp_model_comp_dir
            )
            if model_comp_path:
                saved_artifact_subpath = posixpath.join(
                    saved_model_comp_dir_subpath,
                    os.path.relpath(
                        path=model_comp_path, start=tmp_model_comp_dir.path()
                    ),
                )
            else:
                saved_artifact_subpath = ""
            saved_model_comp_config[model_comp_name] = {
                CONFIG_KEY_RELATIVE_PATH: saved_artifact_subpath,
            }

        shutil.move(
            tmp_model_comp_dir.path(),
            os.path.join(path, saved_model_comp_dir_subpath),
        )
    custom_model_config_kwargs[CONFIG_KEY_MODEL_COMP] = saved_model_comp_config

    saved_code_subpath = None
    if code_paths is not None:
        saved_code_subpath = "code"
        for code_path in code_paths:
            _copy_file_or_tree(src=code_path, dst=path, dst_dir=saved_code_subpath)

    if artifacts:
        saved_artifacts_config = {}
        with TempDir() as tmp_artifacts_dir:
            saved_artifacts_dir_subpath = "artifacts"
            for artifact_name, artifact_uri in artifacts.items():
                tmp_artifact_path = _download_artifact_from_uri(
                    artifact_uri=artifact_uri, output_path=tmp_artifacts_dir.path()
                )
                saved_artifact_subpath = posixpath.join(
                    saved_artifacts_dir_subpath,
                    os.path.relpath(
                        path=tmp_artifact_path, start=tmp_artifacts_dir.path()
                    ),
                )
                saved_artifacts_config[artifact_name] = {
                    CONFIG_KEY_RELATIVE_PATH: saved_artifact_subpath,
                }

            shutil.move(
                tmp_artifacts_dir.path(),
                os.path.join(path, saved_artifacts_dir_subpath),
            )
        custom_model_config_kwargs[CONFIG_KEY_ARTIFACTS] = saved_artifacts_config

    add_to_model(
        model=mlflow_model,
        code=saved_code_subpath,
        env=conda_env_subpath,
        **custom_model_config_kwargs
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def _save_model_comp(model_comp, model_comp_name, output_path: TempDir):
    """
    :param sk_model: The scikit-learn model to serialize.
    :param output_path: The file path to which to write the serialized model.
    """
    if model_comp_name in [MODEL_COMP_MODEL, MODEL_COMP_SCALER]:
        saved_path = os.path.join(output_path.path(), model_comp_name + ".pkl")
        with open(saved_path, "wb") as out:
            pickle.dump(model_comp, out)
    elif model_comp_name in [
        MODEL_COMP_SCHEMA,
        MODEL_COMP_STATS,
        MODEL_COMP_PERF_EVAL_CONFIG,
    ]:
        saved_path = os.path.join(output_path.path(), model_comp_name + ".txt")
        if model_comp_name == MODEL_COMP_SCHEMA:
            tfdv.write_schema_text(model_comp, saved_path)
        elif model_comp_name == MODEL_COMP_STATS:
            tfdv.write_stats_text(model_comp, saved_path)
        elif (model_comp_name == MODEL_COMP_PERF_EVAL_CONFIG) and model_comp:
            mlops_ma.write_eval_config_text(model_comp, saved_path)
        elif not model_comp:
            saved_path = ""
    else:
        raise NotImplementedError()
    return saved_path


def log_model(
    artifact_path: Text,
    model: Union[ModelComponents, MlopsLoadModel],
    eval_config: Optional[mlops_ma.EvalConfig] = None,
    code_paths: Optional[RegPath] = None,
    artifacts: Optional[Dict[Text, RegPath]] = None,
    conda_env: Optional[Union[Text, Dict]] = None,
    mlflow_model: Optional[Mlflow_Model] = None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
):
    """
    Log model to Mlflow tracking server

    Args:
        artifact_path (Text): the artifact path name for saving the model in mlflow artifact store
        model (ModelComponents): model components to save
        conda_env (Optional[Union[Text, Dict]], optional): Either a dictionary representation of a Conda environment or
                                                           the path to a Conda environment yaml file. This describes the
                                                           environment this model should be run in.  Defaults to None.
        mlflow_model (Optional[Mlflow_Model], optional): [description]. Defaults to None.
        signature (Optional[ModelSignature], optional): [description]. Defaults to None.
        input_example (Optional[ModelInputExample], optional): [description]. Defaults to None.
        await_registration_for ([type], optional): [description]. Defaults to DEFAULT_AWAIT_MAX_SLEEP_SECONDS.

    Returns:
        [type]: [description]
    """
    return Mlflow_Model.log(
        artifact_path=artifact_path,
        flavor=mlops.serving.model,
        model=model,
        eval_config=eval_config,
        code_paths=code_paths,
        artifacts=artifacts,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
    )
