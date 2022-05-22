from framework.logger import logger
from framework.config import Config as conf
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, SQLContext
import findspark

findspark.init()


class MLSpark:
    spark = None

    @classmethod
    def getSparkSession(cls):
        try:
            if not cls.spark:
                cls.spark = MLSpark._init_spark()
            return cls.spark
        except Exception as e:
            print("MLSpark.getSparkSession ERROR: ", e)
            raise

    @classmethod
    def getSparkContext(cls):
        return cls.getSparkSession().sparkContext

    @classmethod
    def getSQLContext(cls):
        return SQLContext(cls.getSparkContext())

    @staticmethod
    def _init_spark():
        """
        Initialize the spark environment from the config settings.
        :return: spark session
        """
        try:
            logger.info("Spark Initializing...")
            spark_config = SparkConf().setAppName(conf.settings.spark.app_name)

            spark = (
                SparkSession.builder.appName(conf.settings.spark.app_name)
                .config(conf=spark_config)
                .enableHiveSupport()
                .getOrCreate()
            )

            return spark
        except Exception as e:
            logger.error(e)
            raise
