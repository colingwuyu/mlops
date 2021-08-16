import sys

from examples.iris.linear_discriminant_analysis import run_func
from mlops.orchetrators import datatype


def main():
    argk = sys.argv[1].split("=")[0]
    assert argk == "mlflow_ids"
    argv = sys.argv[1].split("=")[1]
    upstream_ids = {}
    for mlflow_id in argv.split(";"):
        run_name = mlflow_id.split(":")[0]
        run_id = mlflow_id.split(":")[1]
        if run_name == "pipeline":
            pipeline_run_id = run_id
        else:
            upstream_ids.update({run_name: run_id})
    assert "data_gen" in upstream_ids
    assert "data_validation" in upstream_ids
    assert "feature_transform" in upstream_ids
    return run_func(
        upstream_ids=upstream_ids,
        mlflow_info=datatype.MLFlowInfo(
            name="IRIS Training Pipeline",
            mlflow_tracking_uri="http://mlflow_server:5000",
            mlflow_registry_uri="http://mlflow_server:5000",
            mlflow_exp_id="Default",
            mlflow_run_id=pipeline_run_id,
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
