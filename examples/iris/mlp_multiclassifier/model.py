from typing import Any, Tuple, Dict
import copy

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2
from sklearn.metrics import classification_report
import mlflow

from mlops.types import ModelInput, ModelOutput
from mlops.estimator.torchestimator import TorchModel


class MlpClassifierNet(nn.Module):
    def __init__(self, num_features, num_hidden_layers, num_nerons, num_classes):
        super(MlpClassifierNet, self).__init__()
        self.num_hidden_layers = num_hidden_layers

        self.fc_layer_0 = nn.Linear(num_features, num_nerons)
        self.relu_0 = nn.ReLU()
        for i in range(1, num_hidden_layers):
            setattr(self, f"fc_layer_{i}", nn.Linear(num_nerons, num_nerons))
            setattr(self, f"relu_{i}", nn.ReLU())
        self.fc_layer_output = nn.Linear(num_nerons, num_classes)

    def forward(self, X):
        X = self.relu_0(self.fc_layer_0(X))
        for i in range(1, self.num_hidden_layers):
            X = getattr(self, f"fc_layer_{i}")(X)
            X = getattr(self, f"relu_{i}")(X)
        return self.fc_layer_output(X)


class MlpClassifierDataSet(Dataset):
    def __init__(
        self,
        data_schema: schema_pb2.Schema,
        features: pd.DataFrame,
        label: pd.DataFrame,
    ) -> None:
        super().__init__()
        label_name = label.columns.values[0]
        self.classes_ = list(tfdv.get_domain(data_schema, label_name).value)
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes_)}
        self.idx_to_classes = {y: x for x, y in self.class_to_idx.items()}
        self.targets = torch.from_numpy(
            label[label_name].map(self.class_to_idx, na_action="ignore").values
        )
        self.features = torch.from_numpy(features.astype(float))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        feature, target = self.features[index].float(), self.targets[index]
        return feature, target

    def __len__(self):
        return self.targets.shape[0]


class MlpClassifierTrainer:
    def __init__(
        self,
        lr: float = 1e-3,
        n_epochs: int = 150,
        batch_size: int = 120,
        weight_decay: float = 0,
        n_jobs_dataloader: int = 0,
        device: str = "cuda",
    ) -> None:
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.n_jobs_dataloader = n_jobs_dataloader
        self.device = device

    def train(
        self,
        train_dataset: MlpClassifierDataSet,
        net: MlpClassifierNet,
        test_dataset: MlpClassifierDataSet,
        tb: SummaryWriter,
    ):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_jobs_dataloader,
            shuffle=True,
        )
        optimizer = optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()
        net = net.to(self.device)
        criterion = criterion.to(self.device)

        net.train()
        log_graph = False
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            n_batches = 0
            label_prediction = []

            for features, labels in train_loader:
                if not log_graph:
                    tb.add_graph(net, features)
                    log_graph = True
                features, labels = features.to(self.device), labels.to(self.device)
                scores = net(features)
                _, predictions = torch.max(scores, dim=1)
                label_prediction += list(
                    zip(
                        labels.cpu().data.numpy().tolist(),
                        predictions.cpu().data.numpy().tolist(),
                    )
                )

                loss = criterion(scores, labels)
                epoch_loss += loss.item()
                n_batches += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            labels, predictions = zip(*label_prediction)

            # log epoch
            train_metrics = self.metrics_log(
                epoch=epoch,
                tb=tb,
                labels=labels,
                predictions=predictions,
                other_scalars={"loss": epoch_loss / n_batches},
                dataset_type="Train",
            )

            for i in range(net.num_hidden_layers):
                layer_name = f"fc_layer_{i}"
                tb.add_histogram(
                    f"{layer_name}.bias", getattr(net, layer_name).bias, epoch
                )
                tb.add_histogram(
                    f"{layer_name}.weight",
                    getattr(net, layer_name).weight,
                    epoch,
                )
            tb.add_histogram("fc_layer_output.bias", net.fc_layer_output.bias, epoch)
            tb.add_histogram(
                "fc_layer_output.weight", net.fc_layer_output.weight, epoch
            )

            temp_net = copy.deepcopy(net)
            test_metrics = self.test(epoch, test_dataset, temp_net, tb)
            print(
                f"| Epoch: {epoch + 1:03}/{self.n_epochs:03} "
                f"| Train Loss: {train_metrics['Train loss']:.6f} "
                f"| Train Accuracy: {train_metrics['Train accuracy']:.6f} "
                f"| Val Loss: {test_metrics['Test loss']:.6f} ",
                f"| Val Accuracy: {test_metrics['Test accuracy']:.6f} |",
                end="\r",
            )

    def test(
        self,
        epoch: int,
        test_dataset: MlpClassifierDataSet,
        net: MlpClassifierNet,
        tb: SummaryWriter = None,
    ):
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, num_workers=self.n_jobs_dataloader
        )
        criterion = torch.nn.CrossEntropyLoss()
        net = net.to(self.device)
        criterion = criterion.to(self.device)

        epoch_loss = 0
        n_batches = 0
        label_prediction = []
        net.eval()

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                scores = net(features)
                _, predictions = torch.max(scores, dim=1)
                label_prediction += list(
                    zip(
                        labels.cpu().data.numpy().tolist(),
                        predictions.cpu().data.numpy().tolist(),
                    )
                )

                loss = criterion(scores, labels)
                epoch_loss += loss.item()
                n_batches += 1

        labels, predictions = zip(*label_prediction)
        if tb is not None:
            # log epoch
            metrics = self.metrics_log(
                epoch=epoch,
                tb=tb,
                labels=labels,
                predictions=predictions,
                other_scalars={"loss": epoch_loss / n_batches},
                dataset_type="Test",
            )
        return metrics

    def metrics_log(
        self,
        epoch: int,
        tb: SummaryWriter,
        labels,
        predictions,
        other_scalars: Dict,
        dataset_type,
    ):
        labels = np.array(labels)
        predictions = np.array(predictions)
        test_report = classification_report(labels, predictions, output_dict=True)

        # log epoch
        metrics = {}
        for scalar_name, scalar_value in other_scalars.items():
            metrics[f"{dataset_type} {scalar_name}"] = scalar_value
        metrics[f"{dataset_type} accuracy"] = test_report["accuracy"]
        metrics[f"{dataset_type} precision -macro avg-"] = test_report["macro avg"][
            "precision"
        ]
        metrics[f"{dataset_type} recall -macro avg-"] = test_report["macro avg"][
            "recall"
        ]
        metrics[f"{dataset_type} f1-score -macro avg-"] = test_report["macro avg"][
            "f1-score"
        ]
        metrics[f"{dataset_type} precision -weighted avg-"] = test_report[
            "weighted avg"
        ]["precision"]
        metrics[f"{dataset_type} recall -weighted avg-"] = test_report["weighted avg"][
            "recall"
        ]
        metrics[f"{dataset_type} f1-score -weighted avg-"] = test_report[
            "weighted avg"
        ]["f1-score"]
        for name, metric in metrics.items():
            tb.add_scalar(name, metric, epoch)
            mlflow.log_metric(name, metric)
        return metrics


class MlpClassifier(TorchModel):
    def __init__(
        self, num_features, num_hidden_layers, num_neurons, num_classes
    ) -> None:
        super().__init__(
            estimator_type="classifier", model_cls_name=self.__class__.__name__
        )
        self.num_features = num_features
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons = num_neurons
        self.num_classes = num_classes
        self.net = MlpClassifierNet(
            num_features, num_hidden_layers, num_neurons, num_classes
        )
        self.__classes = None
        self.class_to_idx = None
        self.idx_to_classes = None

    @property
    def classes_(self):
        return self.__classes

    @classes_.setter
    def classes_(self, classes_):
        self.__classes = classes_
        self.class_to_idx = {_class: i for i, _class in enumerate(self.__classes)}
        self.idx_to_classes = {y: x for x, y in self.class_to_idx.items()}

    def train(
        self,
        train_dataset: MlpClassifierDataSet,
        test_dataset: MlpClassifierDataSet,
        lr: float = 1e-3,
        n_epochs: int = 150,
        batch_size: int = 120,
        weight_decay: float = 0,
        n_jobs_dataloader: int = 0,
        device: str = "cuda",
        tb: SummaryWriter = None,
    ):
        self.classes_ = train_dataset.classes_
        self.class_to_idx = train_dataset.class_to_idx
        self.idx_to_classes = train_dataset.idx_to_classes
        trainer = MlpClassifierTrainer(
            lr, n_epochs, batch_size, weight_decay, n_jobs_dataloader, device
        )
        trainer.train(train_dataset, self.net, test_dataset, tb)

    def predict(self, X: ModelInput) -> ModelOutput:
        inputs = torch.from_numpy(X.astype(float)).float()
        scores = self.net(inputs)
        _, pred_class_idx = torch.max(scores, dim=1)
        pred_class_idx = pred_class_idx.detach().numpy()
        pred_class = np.empty(shape=pred_class_idx.shape, dtype=np.object)
        for k, v in self.idx_to_classes.items():
            pred_class[pred_class_idx == k] = v
        return pred_class

    def predict_proba(self, X: ModelInput) -> ModelOutput:
        inputs = torch.from_numpy(X.astype(float))
        scores = self.net(inputs.float())
        return F.softmax(scores, dim=-1).detach().numpy()

    def state_dict(self):
        torch_model_state_dict = super().state_dict()
        torch_model_state_dict.update(
            {
                "num_features": self.num_features,
                "num_hidden_layers": self.num_hidden_layers,
                "num_neurons": self.num_neurons,
                "num_classes": self.num_classes,
            }
        )
        return torch_model_state_dict
