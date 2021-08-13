from google.protobuf import text_format
import mlops.model_analysis as mlops_ma

eval_config = text_format.Parse(
    """
model_spec {
    name: "IRIS"
    model_ver: "v1"
}
metric_spec {
    metrics { metric_name: "f1" }
    metrics { metric_name: "recall" }
    metrics { metric_name: "precision" }
    metrics { metric_name: "confusion_matrix" }
    score: "f1"
}
""",
    mlops_ma.EvalConfig(),
)

print(eval_config)

eval_result = mlops_ma.EvalResult(model_spec=eval_config.model_spec)

print(eval_result)
