"""Iteration history tracking for the meta-agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from el_agente_bench.models import ConfusionMatrix


class IterationRecord(BaseModel):
    """Record of a single meta-agent iteration."""

    iteration: int
    hypothesis: str = ""
    changes_summary: str = ""
    diff: str = ""
    train_metrics: ConfusionMatrix = Field(default_factory=ConfusionMatrix)
    timestamp: str = ""
    cost_usd: float = 0.0
    reverted: bool = False


class MetaHistory(BaseModel):
    """Full history of all meta-agent iterations."""

    iterations: list[IterationRecord] = Field(default_factory=list)
    best_iteration: int = 0
    best_f1: float = 0.0

    def add(self, record: IterationRecord) -> None:
        self.iterations.append(record)
        if record.train_metrics.f1 > self.best_f1:
            self.best_f1 = record.train_metrics.f1
            self.best_iteration = record.iteration

    def summary(self) -> str:
        lines = []
        for r in self.iterations:
            marker = " *BEST*" if r.iteration == self.best_iteration else ""
            reverted = " (reverted)" if r.reverted else ""
            lines.append(
                f"  Iteration {r.iteration}: F1={r.train_metrics.f1:.4f} "
                f"TP={r.train_metrics.tp} FP={r.train_metrics.fp} "
                f"TN={r.train_metrics.tn} FN={r.train_metrics.fn} "
                f"— {r.hypothesis}{marker}{reverted}"
            )
        return "\n".join(lines)
