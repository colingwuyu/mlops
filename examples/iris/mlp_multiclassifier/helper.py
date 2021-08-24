import os

from tensorboard import program, notebook

from mlops.utils.mlflowutils import MlflowUtils
from mlops.utils.sysutils import parse_rul
from examples.iris.mlp_multiclassifier import ARTIFACT_TB, OPS_NAME


TB_LOCAL_PATH = "./tb_logs"


def _assert_ops_type(run_id: str):
    assert MlflowUtils.get_run_name(run_id=run_id) == OPS_NAME


def display_tb(run_id):
    download_tb(run_id)
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", TB_LOCAL_PATH, "--bind_all"])
    url = tb.launch()
    url_info = parse_rul(url)
    notebook.display(url_info["port"], height=2000)
    return TB_LOCAL_PATH


def download_tb(run_id):
    _assert_ops_type(run_id)
    tb_save_path = os.path.join(TB_LOCAL_PATH, f"run_{run_id}")
    if not os.path.exists(tb_save_path):
        os.makedirs(tb_save_path)
    MlflowUtils.download_artifact(run_id, ARTIFACT_TB, output_path=tb_save_path)
    return tb_save_path
