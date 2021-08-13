import requests
import io
from pathlib import Path
import os
import tempfile
import subprocess

from requests.api import head


class Utils:
    @staticmethod
    def download_zip(request_url, headers, download_path):
        import zipfile

        resp = requests.request("GET", request_url, headers=headers, verify=False)
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        z.extractall(download_path)

    @staticmethod
    def download_checksum(checksum_sha1_url, headers):
        resp = requests.request("GET", checksum_sha1_url, headers=headers, verify=False)
        return resp.text

    @staticmethod
    def load_yaml_file(file_path):
        import yaml

        yaml_file = io.StringIO()
        try:
            file_path = str(file_path)
            if file_path[:4] == "hdfs":
                yaml_file = Utils.open_hdfs_file(file_path)
            else:
                yaml_file = io.open(file_path)

            dict_obj = yaml.load(yaml_file, Loader=yaml.FullLoader)
            return dict_obj

        except FileNotFoundError:
            print("Utils.load_yaml_file(): File not found -", file_path)
            raise FileNotFoundError
        except Exception as e:
            print(e)
            raise
        finally:
            yaml_file.close()

    @staticmethod
    def is_file(file_name):
        try:
            if file_name[:4] == "hdfs":
                return Utils.is_hdfs_file(file_name)
            else:
                return Path(file_name).is_file()

        except Exception as e:
            print("Utils.is_file(): ERROR: ", e)
            raise e

    @staticmethod
    def is_hdfs_file(file_name):
        try:
            hdfs_cmd = ["hdfs", "dfs", "-test", "-f", file_name]
            resp = subprocess.run(
                hdfs_cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if not resp.returncode:
                return True
            else:
                return False

        except Exception as e:
            print("Utils.is_hdfs_file(): ERROR: ", e)
            raise e

    @staticmethod
    def open_hdfs_file(hdfs_file_path):
        try:
            temp_dir = tempfile.TemporaryDirectory()

            if not Utils.is_hdfs_file(hdfs_file_path):
                raise FileNotFoundError

            config_file_name = hdfs_file_path.split("/")[-1]
            hdfs_cmd = [
                "hdfs",
                "dfs",
                "-get",
                hdfs_file_path,
                os.path.join(temp_dir.name, config_file_name),
            ]

            resp = subprocess.run(
                hdfs_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if not resp.returncode:
                file_io = io.open(
                    Path(temp_dir.name) / config_file_name, "rb", buffering=0
                )
                return file_io

        except ChildProcessError as e:
            print("Utils.open_hdfs_file(): ", e.stderr)
        except Exception as e:
            print("Utils.open_hdfs_file(): ", e)
            raise
        finally:
            temp_dir.cleanup()
