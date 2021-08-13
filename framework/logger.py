import os
import logging
from framework.config import Config as conf

logger = logging.getLogger("MLOps")
logger.setLevel(logging.INFO)


def init_logger(args=None):
    handler = logging.FileHandler(
        os.path.join(
            conf.settings.log.log_dir, args.run_date + "_" + args.UUID + "_mlops.log"
        )
    )

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s " + "%(module)s %(funcName)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.setHandler(handler)
