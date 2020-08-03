from enum import Enum


class ModelType(Enum):
    """
    An enum for describing the type of model to
    generate
    """
    CUSTOM = 1
    XCEPTION = 2
    INCEPTION_V3 = 3
