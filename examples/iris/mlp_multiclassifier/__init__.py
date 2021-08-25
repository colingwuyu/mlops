import tempfile
import os

import torch
from torch.utils.tensorboard import SummaryWriter
import mlflow

from mlops.components import base_component
from mlops.serving import model as mlops_model
from mlops.serving.model import (
    MODEL_COMP_SCALER,
    MODEL_COMP_SCHEMA,
    MODEL_COMP_STATS,
    MODEL_COMP_MODEL,
)
from mlops.serving.estimator.torchestimator import TorchEstimator
from examples.iris.data_gen import helper as dg_helper, OPS_NAME as DATA_GEN_OPS_NAME
from examples.iris.data_validation import (
    helper as dv_helper,
    OPS_NAME as DATA_VAL_OPS_NAME,
)
from examples.iris.feature_transform import (
    helper as ft_helper,
    OPS_NAME as FEATURE_TRANSFORM_OPS_NAME,
)
from examples.iris.mlp_multiclassifier.model import MlpClassifierDataSet, MlpClassifier

OPS_NAME = "mlp_multiclass"
OPS_DES = """
# MLP Multi-Classification
"""

ARTIFACT_MODEL = "mlops_model"
ARTIFACT_TB = "tensorboard_log"


@base_component(name=OPS_NAME, note=OPS_DES)
def run_func(upstream_ids: dict, **kwargs):
    model_comps = {}
    num_hidden_layers = kwargs.get("num_hidden_layers", 1)
    num_nerons = kwargs.get("num_neurons", 4)
    lr = kwargs.get("learning_rate", 1e-3)
    n_epochs = kwargs.get("epochs", 5)
    batch_size = kwargs.get("batch_size", 120)
    weight_decay = kwargs.get("weight_decay", 0)
    n_jobs_dataloader = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tb_tmpdir = tempfile.mkdtemp()
    tb_logdir = os.path.join(tb_tmpdir, ARTIFACT_TB)
    os.mkdir(tb_logdir)
    tb = SummaryWriter(
        tb_logdir,
        comment=f"mlp_multiclassifier_HIDDEN_{num_hidden_layers}_NERONS_{num_nerons}_LR_{lr}_EPOCHS_{n_epochs}_BATCH_{batch_size}_WEIGHTDECAY_{weight_decay}",
    )

    data_gen_id = upstream_ids[DATA_GEN_OPS_NAME]
    X_train = dg_helper.load_train_X(data_gen_id)
    y_train = dg_helper.load_train_y(data_gen_id)
    X_test = dg_helper.load_test_X(data_gen_id)
    y_test = dg_helper.load_test_y(data_gen_id)

    data_val_id = upstream_ids[DATA_VAL_OPS_NAME]
    model_comps[MODEL_COMP_SCHEMA] = dv_helper.get_schema(data_val_id)
    model_comps[MODEL_COMP_STATS] = dv_helper.get_trainset_stat(data_val_id)

    feature_scaler_id = upstream_ids[FEATURE_TRANSFORM_OPS_NAME]
    feature_scaler = ft_helper.load_scaler(feature_scaler_id)
    model_comps[MODEL_COMP_SCALER] = feature_scaler

    X_train = feature_scaler.transform(X_train)
    X_test = feature_scaler.transform(X_test)

    train_dataset = MlpClassifierDataSet(
        model_comps[MODEL_COMP_SCHEMA], X_train, y_train
    )
    test_dataset = MlpClassifierDataSet(model_comps[MODEL_COMP_SCHEMA], X_test, y_test)
    num_features = X_train.shape[1]
    num_classes = len(train_dataset.classes_)
    model = MlpClassifier(num_features, num_hidden_layers, num_nerons, num_classes)
    model.train(
        train_dataset,
        test_dataset,
        lr,
        n_epochs,
        batch_size,
        weight_decay,
        n_jobs_dataloader,
        device,
        tb,
    )
    model_estimator = TorchEstimator()
    model_estimator.set_model(model)
    model_comps[MODEL_COMP_MODEL] = model_estimator

    conda_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../conda.yaml"
    )
    mlops_model.log_model(ARTIFACT_MODEL, model_comps, conda_env=conda_file)
    mlflow.log_artifact(tb_logdir)
    tb.close()
