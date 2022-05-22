# Feature Transformation

This is a feature transformation operation.
This operation adopts several common feature scaling transformation approaches, such as:

- min-max
- standardization

## Task type

- any

## Upstream dependencies

- Data extraction
- Dataset split

## Paramter

- transform_approach: transformation approache (minmax/std)

## Metrics

None

## Artifacts

1. scaler.pkl

## Helper functions

- `load_scaler(run_id: str)`
