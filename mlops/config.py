import os
from pathlib import Path

from framework.utils import Utils
from framework.config import Config as FConfig
from mlops.utils.collectionutils import convert_to_namedtuple, update_dictionary

dirpath = os.path.dirname(os.path.realpath(__file__))


def get_config_path(environment):
    """retrieve config file path
    default path is in package/config folder

    Args:
        environment (str): local/dev/qa/prod

    Raises:
        FileNotFoundError: configuration file not found

    Returns:
        str: config file path
    """
    mlops_config_dir = os.environ.get(
        "MLOPS_CONFIG_DIR",
        os.path.join(dirpath, "config_files"),
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
    """configuration in Singleton

    Raises:
        FileNotFoundError: config file does not exist
    """

    settings: str = None

    @classmethod
    def load(cls, config_from=None, override={}):
        """load configuration file from 'config_from'

        Args:
            config_from (str, optional): run envioronment - dev/qa/prod. Defaults to None.
                The yaml file of the corresponding run-env file will be loaded.
                or configuration file to override or extend base env configs.
            override (dict, optional): dict object with override or extended configuration. Defaults to {}.

        Raises:
            FileNotFoundError: base config file does not exist

        Returns:
            namedtuple: configuration
        """
        try:
            # If argument is none, load default environment config
            if not config_from:
                if (cls.settings is not None) and (len(override) == 0):
                    return cls.settings
                config_from = os.environ.get("APP_ENV", "local").lower()

            # If config is not a path, ex 'dev', get the full path
            if not Path(config_from).is_file() and config_from[:4] != "hdfs":
                config_from = get_config_path(config_from)
                base_config_file_path = None
            else:
                # get the base config file path
                base_config_file_path = get_config_path(
                    os.environ.get("APP_ENV", "local").lower()
                )

            base_config_file_path = None
            if Utils.is_file(config_from):
                print("Loading configuration from: ", config_from)
                file_settings = cls._load_yaml_file(config_from)

                # If override requested on base config, load base configuration
                if base_config_file_path is not None and Utils.is_file(
                    base_config_file_path
                ):
                    print("Loading base configuration from: ",
                          base_config_file_path)
                    cls.settings = cls._load_yaml_file(base_config_file_path)
                    # Override the base configuration with the given configuration file.
                    cls.settings = update_dictionary(
                        cls.settings, file_settings)

                # The file being loaded is not override config, all set.
                else:
                    cls.settings = file_settings
                cls.settings = update_dictionary(cls.settings, override)
            else:
                print("File Not Found: ", config_from)
                raise FileNotFoundError(config_from)

            # upload app config to framework config
            FConfig.load(cls.settings)

            # make the config immutable
            cls.settings = convert_to_namedtuple(cls.settings)
            return cls.settings

        except Exception as e:
            print(e)
            raise

    @classmethod
    def _load_yaml_file(cls, file_name):
        return Utils.load_yaml_file(file_name)
