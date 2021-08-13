import mlflow

from mlops.components.base_component import base_component


def model_training_component(
    name: str = None, note: str = None, framework: str = None, log_models: bool = False
):
    def inner_init_component(run_func):
        def inner_mlflow_wrapper(*args, **kwargs):
            base_component_ops = base_component(name, note)(run_func)
            if not framework:
                mlflow.autolog(log_model_signatures=False, log_models=log_models)
            else:
                assert framework in [
                    "sklearn",
                    "tensorflow",
                    "gluon",
                    "xhboost",
                    "lightgbm",
                    "statsmodels",
                    "spark",
                    "fastai",
                    "pytorch",
                ], (
                    "ERROR: autolog is not supported for framework %s" % framework
                )
                import importlib

                mlflow_module = importlib.import_module("mlflow." + framework)
                mlflow_module.autolog(log_model_signatures=False, log_models=log_models)
            return base_component_ops(*args, **kwargs)

        return inner_mlflow_wrapper

    return inner_init_component
