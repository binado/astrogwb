from typing import Protocol

from bilby.core.prior import PriorDict


class IntrinsicPriorDictGenerator(Protocol):
    def __call__(self, parameters: dict[str, float]) -> PriorDict: ...
