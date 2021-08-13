"""Types."""
from typing import Text, Dict, Any, Union, List
import numpy as np
import pandas

RegPath = str
ModelPath = str
ArtifactPath = str
ModelComponent = Text
ModelComponents = Dict[ModelComponent, Any]

ModelInput = Union[pandas.DataFrame, np.ndarray, List[Any], Dict[str, Any]]
ModelOutput = Union[pandas.DataFrame, pandas.Series, np.ndarray, list]
