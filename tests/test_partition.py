import random
from src.model import ShufflingSamplePartition
from pprint import pprint

data = [
    {
        "sepal_length" : i + 0.1,
        "sepal_width" : i + 0.2,
        "petal_length" : i + 0.3,
        "petal_width" : i + 0.4,
        "species" : f"sample {i}",
    }
    for i in range(10)
]

random.seed(42)
ssp = ShufflingSamplePartition(data)
pprint(ssp.testing)