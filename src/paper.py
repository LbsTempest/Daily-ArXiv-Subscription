from dataclasses import dataclass, field
from typing import List

@dataclass
class Paper:
    title: str
    link: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    comment: str = ""
    date: str = ""