from examples.iris.data_gen import run_func
from mlops.orchetrators import datatype


def main():
    return run_func(
        x_headers=["sepal-length", "sepal-width", "petal-length", "petal-width"],
        y_header="species",
        test_size=0.2,
        url="http://iris_serving:3000/data",
        mlflow_info=datatype.MLFlowInfo(
            name="IRIS Training Pipeline",
            mlflow_tracking_uri="http://mlflow_server:5000",
            mlflow_registry_uri="http://mlflow_server:5000",
            mlflow_exp_id="Default",
        ),
    )


if __name__ == "__main__":
    pipeline_mlflow_runid, component_mlflow_runid = main()
    print(
        {
            "pipeline_mlflow_runid": pipeline_mlflow_runid,
            "component_mlflow_runid": component_mlflow_runid,
        }
    )
