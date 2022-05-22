# Model Evaluation

This is to evaluate trained models to select the best one for deployment

## Task type

- any

## Upstreaam dependencies

- Model training

## Parameter

compare_metric: performance metrics for model score

## Metrics

- test_accuracy_score

## Artifacts

1. Model

## Helper functions

- `load_model(run_id: str)`
- `display_performance_eval_report(run_id: str, model_name: str)`
- `get_model_metric_name(run_id: str)`
- `get_selected_model_name(run_id: str)`
- `get_selected_model_metric_value(run_id: str)`
