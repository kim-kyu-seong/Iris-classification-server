from __future__ import annotations
from dataclasses import dataclass
import abc
import random
import collections
import collections.abc
import csv
import datetime
import enum
from math import isclose
from pathlib import Path
from typing import (
    cast,
    Any,
    Optional,
    Iterator,
    Iterable,
    Counter,
    Protocol,
    TypedDict,
    List,
    Tuple,
    overload
)

import weakref


@dataclass(frozen=True)
class Sample:
    """Abstract superclass for all samples."""
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class Purpose(enum.IntEnum):
    Classification = 0
    Testing = 1
    Training = 2

@dataclass(frozen=True)
class KnownSample(Sample):
    """Represents a sample of testing or training data, the species is set once
    The purpose determines if it can or cannot be classified.
    """
    species: str


class UnknownSample(Sample):
    """A sample provided by a User, to be classified."""

    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> None:
        super().__init__(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )
        self._classification: Optional[str] = None

    @property
    def classification(self) -> Optional[str]:
        return self._classification

    @classification.setter
    def classification(self, value: str) -> None:
        self._classification = value

    def __repr__(self) -> str:
        base_attributes = self.attr_dict
        base_attributes["classification"] = f"{self.classification!r}"
        attrs = ", ".join(f"{k}={v}" for k, v in base_attributes.items())
        return f"{self.__class__.__name__}({attrs})"


@dataclass(frozen=True)
class TrainingKnownSample:
    sample: KnownSample

@dataclass
class TestingKnownSample:
    sample: KnownSample
    classification: Optional[str] = None


class Distance:
    """A distance computation"""

    def distance(self, s1: Sample, s2: Sample) -> float:
        raise NotImplementedError


class Chebyshev(Distance):
    """
    Computes the Chebyshev distance between two samples.

    ::

        >>> from math import isclose
        >>> from model import KnownSample, Purpose, UnknownSample, Chebyshev

        >>> s1 = KnownSample(
        ...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa",
        ...     purpose=Purpose.Training)
        >>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})

        >>> algorithm = Chebyshev()
        >>> isclose(3.3, algorithm.distance(s1, u))
        True

    """

    def distance(self, s1: Sample, s2: Sample) -> float:
        return max(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        )


class Minkowski(Distance):
    """An abstraction to provide a way to implement Manhattan and Euclidean."""

    m: int

    def distance(self, s1: Sample, s2: Sample) -> float:
        return (
            sum(
                [
                    abs(s1.sepal_length - s2.sepal_length) ** self.m,
                    abs(s1.sepal_width - s2.sepal_width) ** self.m,
                    abs(s1.petal_length - s2.petal_length) ** self.m,
                    abs(s1.petal_width - s2.petal_width) ** self.m,
                ]
            )
            ** (1 / self.m)
        )


class Euclidean(Minkowski):
    m = 2


class Manhattan(Minkowski):
    m = 1


class Sorensen(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        ) / sum(
            [
                s1.sepal_length + s2.sepal_length,
                s1.sepal_width + s2.sepal_width,
                s1.petal_length + s2.petal_length,
                s1.petal_width + s2.petal_width,
            ]
        )


class Reduce_Function(Protocol):
    """Define a callable object with specific parameters."""

    def __call__(self, values: list[float]) -> float:
        pass


class Minkowski_2(Distance):
    """A generic way to implement Manhattan, Euclidean, and Chebyshev.

    ::

        >>> from math import isclose
        >>> from model import KnownSample, Purpose, UnknownSample, Minkowski_2


        >>> class CD(Minkowski_2):
        ...     m = 1
        ...     reduction = max

        >>> s1 = KnownSample(
        ...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa",
        ...     purpose=Purpose.Training)
        >>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})

        >>> algorithm = CD()
        >>> isclose(3.3, algorithm.distance(s1, u))
        True

    """

    m: int
    reduction: Reduce_Function

    def distance(self, s1: Sample, s2: Sample) -> float:
        # Required to prevent Python from passing `self` as the first argument.
        summarize = self.reduction
        return (
            summarize(
                [
                    abs(s1.sepal_length - s2.sepal_length) ** self.m,
                    abs(s1.sepal_width - s2.sepal_width) ** self.m,
                    abs(s1.petal_length - s2.petal_length) ** self.m,
                    abs(s1.petal_width - s2.petal_width) ** self.m,
                ]
            )
            ** (1 / self.m)
        )

@dataclass
class Hyperparameter:
    """A hyperparameter value and the overall quality of the classification."""
    k: int
    algorithm: Distance
    data: weakref.ReferenceType["TrainingData"]

    def test(self) -> None:
        """Run the entire test suite."""
        training_data: Optional["TrainingData"] = self.data()
        if not training_data:
            raise RuntimeError("Broken Weak Reference")
        pass_count, fail_count = 0, 0
        for sample in training_data.testing:
            sample.classification = self.classify(sample)
            if sample.matches():
                pass_count += 1
            else:
                fail_count += 1
        self.quality = pass_count / (pass_count + fail_count)

    def classify(self, sample: Sample) -> str:
        """The k-NN algorithm"""
        training_data = self.data()
        if not training_data:
            raise RuntimeError("No TrainingData object")
        distances: list[tuple[float, KnownSample]] = sorted(
            (self.algorithm.distance(sample, known), known)
            for known in training_data.training
        )
        k_nearest = (known.species for d, known in distances[: self.k])
        frequency: Counter[str] = collections.Counter(k_nearest)
        best_fit, *others = frequency.most_common()
        species, votes = best_fit
        return species


@dataclass
class TrainingData:
    """A set of training data and testing data with methods to load and test the samples."""

    testing: List[TestingKnownSample]
    training: List[TrainingKnownSample]
    tuning: List[Hyperparameter]


class BadSampleRow(ValueError):
    pass


class SampleReader:
    """See iris.names for attribute ordering in bezdekIris.data file"""

    target_class = Sample
    header = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

    def __init__(self, source: Path) -> None:
        self.source = source

    def sample_iter(self) -> Iterator[Sample]:
        target_class = self.target_class
        with self.source.open() as source_file:
            reader = csv.DictReader(source_file, self.header)
            for row in reader:
                try:
                    sample = target_class(
                        sepal_length=float(row["sepal_length"]),
                        sepal_width=float(row["sepal_width"]),
                        petal_length=float(row["petal_length"]),
                        petal_width=float(row["petal_width"]),
                    )
                except ValueError as ex:
                    raise BadSampleRow(f"Invalid {row!r}") from ex
                yield sample


class SampleDict(TypedDict):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str

# 데이터 분할을 위한 클래스
class Samplepartition(List[SampleDict], abc.ABC):
    @overload
    def __init__(self, *, training_subset: float = 0.80) -> None: # 80퍼를 사용하겠다
        ...

    @overload
    def __init__(
        self,
        iterable: Optional[Iterable[SampleDict]] = None,
        *,
        training_subset: float = 0.80
    ) -> None:
        ...

    def __init__(
        self,
        iterable: Optional[Iterable[SampleDict]] = None,
        *,
        training_subset: float = 0.80,
    ) -> None:
        self.training_subset = training_subset
        if iterable:
            super().__init__(iterable)
        else:
            super().__init__()

    @abc.abstractproperty
    @property
    def training(self) -> list[TrainingKnownSample]:
        ...

    @abc.abstractproperty
    @property
    def testing(self) -> list[TestingKnownSample]:
        ...

class ShufflingSamplePartition(Samplepartition):
    def __init__(
            self, 
            iterable: Optional[Iterable[SampleDict]] = None,
            *,
            training_subset: float = 0.80
        )->None:
            super().__init__(iterable, training_subset)
            self.split: Optional[int] - None



    @property
    def training(self) -> list[TrainingKnownSample]:
        self.shuffle()
        return [TrainingKnownSample(**sd) for sd in self[:self.split]]
    
    @property
    def testing(self) -> list[TestingKnownSample]: # 태스팅 샘플은 셔플 하지 않음
        return [TrainingKnownSample(**sd) for sd in self[self.split:]]  # **sd는 딕셔너리 언팩을 위해서.
    
    def shuffle(self) -> None:
        if not self.split:
            random.shuffle(self)
            self.split = int(len(self) * self.training_subset) # 실수값인데 정수로 받기로 했으니까. 인트는 바닥함수이므로 소수점을 버림


# 데이터를 추가하는 추상클래스
class DealingPartition(abc.ABC):
    def __init__(
        self, 
        items: Optional[Iterable[SampleDict]],
        *,
        training_subset: Tuple[int, int] = (8, 10)
    ) -> None:
        ...

    @abc.abstractproperty
    @property
    def training(self) -> List[TrainingKnownSample]:
        ...

    @abc.abstractproperty
    @property
    def testing(self) -> List[TestingKnownSample]:
        ...

    @abc.abstractmethod # 이터러블 여러개를 다 넣어줌 
    def extend(self, items: Iterable[SampleDict]) -> None:
        ...

    @abc.abstractmethod # 아이템 하나만 추가
    def append(self, item: SampleDict) -> None:
        ...

class CountingDealingPartition(DealingPartition):
    def __init__(
        self,
        items: Optional[Iterable[SampleDict]],
        *, 
        training_subset: Tuple[int, int] = (8, 10)
    ) -> None:
        self.training_subset = training_subset
        self.counter = 0
        self._training: list[TrainingKnownSample] = []
        self._testing: list[TestingKnownSample] = []
        if items:
            self.extend(items)

    def extend(self, items: Iterable[SampleDict]) -> None:
        for item in items:
            self.append(item)
    
    def append(self, item: SampleDict) -> None:
        n, d = self.training_subset
        if self.counter % d < n:
            self._training.append(TrainingKnownSample(**item))
        else:
            self._testing.append(TestingKnownSample(**item))
        self.counter += 1
    
    @property
    def training(self) -> List[TrainingKnownSample]:
        return self._training
    
    @property
    def testing(self) -> List[TestingKnownSample]:
        return self._testing



# Special case, we don't *often* test abstract superclasses.
# In this example, however, we can create instances of the abstract class.
test_Sample = """
>>> x = Sample(1, 2, 3, 4)
>>> x
Sample(sepal_length=1, sepal_width=2, petal_length=3, petal_width=4)
"""

test_Training_KnownSample = """
>>> s1 = KnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa", purpose=Purpose.Training.value)
>>> s1
KnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, purpose=2, species='Iris-setosa')
>>> s1.classification
Traceback (most recent call last):
...
AttributeError: Training samples have no classification
>>> s1.classification("rejected")
Traceback (most recent call last):
...
AttributeError: Training samples have no classification
"""

test_Testing_KnownSample = """
>>> s2 = KnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa", purpose=Purpose.Testing.value)
>>> s2
KnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, purpose=1, species='Iris-setosa')
>>> s2.classification
>>> s2.classification = "wrong"
>>> s2
KnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, purpose=1, species='Iris-setosa', classification='wrong')
"""

test_UnknownSample = """
>>> u = UnknownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, )
>>> u
UnknownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, classification=None)
>>> u.classification = "something"
>>> u
UnknownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, classification='something')
"""

test_Chebyshev = """
>>> s1 = KnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa", purpose=Purpose.Training.value)
>>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})

>>> algorithm = Chebyshev()
>>> isclose(3.3, algorithm.distance(s1, u))
True
"""

test_Euclidean = """
>>> s1 = KnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa", purpose=Purpose.Training.value)
>>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})

>>> algorithm = Euclidean()
>>> isclose(4.50111097, algorithm.distance(s1, u))
True
"""

test_Manhattan = """
>>> s1 = KnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa", purpose=Purpose.Training.value)
>>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})

>>> algorithm = Manhattan()
>>> isclose(7.6, algorithm.distance(s1, u))
True
"""

test_Sorensen = """
>>> s1 = KnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa", purpose=Purpose.Training.value)
>>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})

>>> algorithm = Sorensen()
>>> isclose(0.2773722627, algorithm.distance(s1, u))
True
"""

test_Hyperparameter = """
>>> td = TrainingData('test')
>>> s2 = KnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa", purpose=Purpose.Testing.value)
>>> td.testing = [s2]
>>> t1 = KnownSample(**{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "species": "Iris-setosa", "purpose": 1})
>>> t2 = KnownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4, "species": "Iris-versicolor", "purpose": 1})
>>> td.training = [t1, t2]
>>> h = Hyperparameter(k=3, algorithm=Chebyshev(), training=td)
>>> u = UnknownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)
>>> h.classify(u)
'Iris-setosa'
>>> h.test()
>>> print(f"data={td.name!r}, k={h.k}, quality={h.quality}")
data='test', k=3, quality=1.0
"""

test_TrainingData = """
>>> td = TrainingData('test')
>>> raw_data = [
... {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "species": "Iris-setosa"},
... {"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4, "species": "Iris-versicolor"},
... ]
>>> td.load(raw_data)
>>> h = Hyperparameter(k=3, algorithm=Chebyshev(), training=td)
>>> len(td.training)
1
>>> len(td.testing)
1
>>> td.test(h)
>>> print(f"data={td.name!r}, k={h.k}, quality={h.quality}")
data='test', k=3, quality=0.0
"""

__test__ = {name: case for name, case in globals().items() if name.startswith("test_")}
