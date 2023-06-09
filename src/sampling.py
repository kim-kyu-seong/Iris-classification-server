from __future__ import annotations
from dataclasses import dataclass
import enum
from typing import (
    Optional,
    TypedDict,
)


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

    
class SampleDict(TypedDict):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str