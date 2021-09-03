

if __name__ == "__main__":

    from mlops.serving import model as mlops_serving_model
    import mlops.serving.estimator.consts as consts

    # Load model as a PyFuncModel.
    logged_model = "/home/colin/mlflow_artifacts/1/e5d5ba20b47d4c3aa8eb45cc05ae13aa/artifacts/mlops_model"

    loaded_model = mlops_serving_model.load_model(logged_model)
    import pandas as pd

    data = pd.read_csv("examples/iris/serving/data/inputs.csv")
    import pprint

    pprint.pprint(loaded_model.predict(
        data, [consts.PREDICTION_KEY_PROBABILITIES]))
