# Dataset Validation


this is a data validation operation

## Task type

- supervised-learning
- classification

## Upstream dependencies

- Data extraction
- Dataset split

## Parameters

None

## Metrics

None

## Artifacts

1. schema.pbtxt
2. trainset_stat.pbtxt
3. scatter_plot.png

## Helper functions

- `get_schema(run_id: str)`
- `display_schema(run_id: str)`
- `get_trainset_stat(run_id: str)`
- `display_trainset_stat(run_id: str)`
- `display_scatter_plot(run_id: str)`