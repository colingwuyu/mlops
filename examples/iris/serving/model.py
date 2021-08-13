from os import mkdir
import shutil
import uuid

import mlflow.pyfunc
import tensorflow
import tensorflow_data_validation as tfdv
import cloudpickle
import pandas as pd


class PyModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self._trainset_stats = tfdv.load_stats_text(context.artifacts["trainset_stats"])
        self._data_schema = tfdv.load_schema_text(context.artifacts["data_schema"])
        with open(context.artifacts["scaler"], "rb") as scaler_io:
            self._feature_scaler = cloudpickle.load(scaler_io)
        with open(context.artifacts["model"], "rb") as model_io:
            self._model = cloudpickle.load(model_io)

    def predict(self, context, model_input: pd.DataFrame):
        scaled_inputs = self._feature_scaler.transform(model_input)
        y = self._model.predict_proba(scaled_inputs)
        return y

    def predict_prob(self, context, model_input: pd.DataFrame):
        scaled_inputs = self._feature_scaler.transform(model_input)
        y_prob = self._model.predict_proba(scaled_inputs)
        return y_prob


class ServingModel(mlflow.pyfunc.PyFuncModel):
    def __init__(self, model_name: str, stage: str) -> None:
        pyfunc_model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{stage}"
        )
        super().__init__(
            model_meta=pyfunc_model.metadata, model_impl=pyfunc_model._model_impl
        )

    def validate_input(self, model_input: pd.DataFrame):
        model_input_path = "data/{}/".format(uuid.uuid4())
        mkdir(model_input_path)
        model_input.to_csv(model_input_path + "model_input.csv")
        # TODO generate tfdv anomaly report
        serving_stats = tfdv.generate_statistics_from_dataframe(model_input)
        options = tfdv.StatsOptions(schema=self._data_schema)
        serving_per_example_anomalies = tfdv.validate_examples_in_csv(
            data_location=model_input_path, stats_options=options
        )
        serving_stats_anomalies = tfdv.validate_statistics()
        serving_skew_anomalies = tfdv.validate_statistics()
        shutil.rmtree(model_input_path)

    def performace_measure(
        self,
    ):
        ...

    def predict_prob(self, data):
        return self._model_impl.predict_prob(data)
