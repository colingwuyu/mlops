from mlops.utils.mlflowutils import MlflowUtils


def _assert_ops_type(run_id: str):
    assert MlflowUtils.get_run_name(run_id=run_id) == "random_dataset_split"


def get_label_header(run_id: str):
    _assert_ops_type(run_id)
    return eval(MlflowUtils.get_parameter(run_id, "y_header"))[0]


def get_feature_header(run_id: str):
    _assert_ops_type(run_id)
    return eval(MlflowUtils.get_parameter(run_id, "x_headers"))


def split_data(run_id: str, raw_data):
    X_headers = get_feature_header(run_id)
    Y_header = get_label_header(run_id)
    train_test_split = MlflowUtils.load_dict(run_id, "train_test_split.json")
    X_train = raw_data.loc[train_test_split["X_train_ind"], X_headers]
    X_test = raw_data.loc[train_test_split["X_val_ind"], X_headers]
    Y_train = raw_data.loc[train_test_split["Y_train_ind"], Y_header]
    Y_test = raw_data.loc[train_test_split["Y_val_ind"], Y_header]
    return X_train, X_test, Y_train, Y_test
