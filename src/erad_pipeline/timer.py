"""Timing helpers for the ERAD experiment."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator


@dataclass
class StageTimer:
    backend: str
    run_id: str
    n_samples: int | None
    dataset_name: str
    rows: list[dict[str, object]] = field(default_factory=list)

    @contextmanager
    def track(
        self,
        etapa: str,
        *,
        implementation_backend: str,
        observacoes: str = "",
    ) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.rows.append(
                {
                    "backend": self.backend,
                    "run_id": self.run_id,
                    "n_samples": self.n_samples,
                    "etapa": etapa,
                    "tempo_segundos": elapsed,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "nome_do_dataset": self.dataset_name,
                    "implementation_backend": implementation_backend,
                    "observacoes": observacoes,
                }
            )

    def add_manual(
        self,
        etapa: str,
        tempo_segundos: float,
        *,
        implementation_backend: str,
        observacoes: str = "",
    ) -> None:
        self.rows.append(
            {
                "backend": self.backend,
                "run_id": self.run_id,
                "n_samples": self.n_samples,
                "etapa": etapa,
                "tempo_segundos": tempo_segundos,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "nome_do_dataset": self.dataset_name,
                "implementation_backend": implementation_backend,
                "observacoes": observacoes,
            }
        )
