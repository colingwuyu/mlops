from setuptools import setup, find_packages

PACKAGE_NAME = "mlops"
AUTHOR = "Zeyu Wang"
EMAIL = "colingwuyu@gmail.com"

SUB_MODULES = PACKAGE_NAME + ".*"

setup(
    name=PACKAGE_NAME,
    version="0.0.1",
    description="MLOPS Package",
    author=AUTHOR,
    author_email=EMAIL,
    packages=find_packages(include=(PACKAGE_NAME, SUB_MODULES)),
)
