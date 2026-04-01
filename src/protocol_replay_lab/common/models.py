from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(slots=True)
class RequestResponse:
    request: bytes
    response: Optional[bytes] = None
    error: Optional[str] = None


@dataclass(slots=True)
class SendSummary:
    host: str
    port: int
    sent_count: int = 0
    total_count: int = 0
    elapsed_seconds: float = 0.0
    failed_indices: List[int] = field(default_factory=list)
    exchanges: List[RequestResponse] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.sent_count == self.total_count and not self.failed_indices
