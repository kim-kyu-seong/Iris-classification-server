from model import SampleDict, TrainingKnownSample, TestingKnownSample
import abc
import random
import collections
import collections.abc
from typing import (
    Optional,
    Iterable,
    List,
    overload
)

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
