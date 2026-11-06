from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IngestedDocument:
    content: str
    source: str
    modality: str

    metadata: dict = field(default_factory=dict)

    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None