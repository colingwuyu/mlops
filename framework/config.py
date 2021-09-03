import os
from collections import namedtuple
import collections
from pathlib import Path
from framework.utils import Utils


def convert_to_namedtuple(dictionary: dict):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_to_namedtuple(value)
    return namedtuple("configuration", dictionary.keys())(**dictionary)


def update_dictionary(base, updated, restricted={}):
    for k, v in updated.items():
        if isinstance(v, collections.abc.Mapping):
            base[k] = update_dictionary(
                base.get(k, {}), v, restricted.get(k, {}))
        else:
            if k not in restricted:
                base[k] = v
            else:
                print(
                    "CONFIG WARN: Tried to update resticted " "configuration:",
                    k,
                    "->",
                    v,
                )
    return base


def get_config_path(environment):
    mlops_config_dir = os.environ.get(
        "MLOPS_CONFIG_DIR",
        os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "config_files"),
    )
    if not mlops_config_dir:
        print(
            "MLOps Config dir and Version not found.",
            "Set the env variable: MLOPS_CONFIG_DIR and MLOPS_CONFIG_VER",
        )
        raise FileNotFoundError

    config_file_path = os.path.join(mlops_config_dir, environment + ".yaml")
    return config_file_path


class Config:
    settings: str = None

    @classmethod
    def load(cls, config_from=None, override={}):
        """
        :param config_from : run environment - dev/qa/prod.
            The yaml file of the corresponding run-env file will be loaded.
            or configuration file to override or extend base env configs.
        :param override: dict object with override or extended configuration
        :return: configuration: namedtuple
        """
        try:
            if isinstance(config_from, dict):
                override = config_from
                config_from = None

            if not config_from:
                config_from = os.environ.get("APP_ENV", "local").lower()

            if not Path(config_from).is_file() and config_from[:4] != "hdfs":
                config_from = get_config_path(config_from)
                base_config_file_path = None
            else:
                base_config_file_path = get_config_path(
                    os.environ.get("APP_ENV", "local").lower()
                )
            if Utils.is_file(config_from):
                print("Loading configuration from: ", config_from)
                file_settings = cls._load_yaml_file(config_from)

                restricted_config = cls._load_restricted_config()

                file_settings = update_dictionary(
                    file_settings, override, restricted_config
                )
                if base_config_file_path is not None and Utils.is_file(
                    base_config_file_path
                ):
                    print("Loading base configuration from: ",
                          base_config_file_path)
                    cls.settings = cls._load_yaml_file(base_config_file_path)
                    cls.settings = update_dictionary(
                        cls.settings, file_settings, restricted_config
                    )
                else:
                    cls.settings = file_settings

            else:
                print("File Not Found: ", config_from)
                raise FileNotFoundError(config_from)

            cls.settings = convert_to_namedtuple(cls.settings)
            return cls.settings

        except Exception as e:
            print(e)
            raise

    @classmethod
    def _load_yaml_file(cls, file_name):
        return Utils.load_yaml_file(file_name)

    @classmethod
    def _load_restricted_config(cls):
        try:
            restricted_file_name = (
                os.environ.get("APP_ENV", "dev").lower() + "_restricted"
            )
            restricted_file_path = get_config_path(restricted_file_name)

            print("Loading config restriction from: ", restricted_file_path)
            return cls._load_yaml_file(restricted_file_path)

        except FileNotFoundError as e:
            return {}
        except Exception as e:
            print(e)
            raise
