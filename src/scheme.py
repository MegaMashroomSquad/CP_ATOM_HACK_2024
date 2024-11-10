from dataclasses import dataclass
from typing import Optional


@dataclass
class Fact:
    text: str  # текст утверждения
    doc_name: str  # название дока
    section_name: str  # название секции внутри которой находится факт

    def __repr__(self) -> str:
        return f"Text: {self.text}, DocName: {self.doc_name}, SectionName: {self.section_name}"


@dataclass
class FactComparison:
    factA: str  # первый факт для сравнения
    factB: str  # второй факт для сравнения
    has_diff: bool  # есть ли различие между фактами
    diff: Optional[str]  # описание различия, если оно есть
    severeness_level: int  # уровень серьезности различия

    def __repr__(self) -> str:
        return f"FactA: {self.factA}, FactB: {self.factB}, HasDiff: {self.has_diff}, Diff: {self.diff}, SeverenessLevel: {self.severeness_level}"
