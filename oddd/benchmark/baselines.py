from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["ok", "blocked", "failed", "skipped"]


@dataclass
class BaselineResult:
    model: str
    strategy: str
    status: Status
    metrics: dict[str, Any] = field(default_factory=dict)
    batch: dict[str, Any] = field(default_factory=dict)
    bootstrap: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "strategy": self.strategy,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "metrics": self.metrics,
            "batch": self.batch,
            "bootstrap": self.bootstrap,
        }


class BaselineRegistry:
    """Canonical baseline slots for benchmark runs."""

    PREDICTION_MODELS = ("rf", "mlp", "xgboost", "lightgbm", "chemprop_v2")
    STRATEGIES = ("odca", "ensemble_ucb", "vanilla_conformal", "potency_only")

    @staticmethod
    def probe_chemprop_v2() -> BaselineResult:
        """Optional external baseline — never fabricate metrics when not trained."""
        try:
            import chemprop  # noqa: F401

            return BaselineResult(
                model="chemprop_v2",
                strategy="none",
                status="blocked",
                failure_reason="chemprop available but not trained in this run (use chemprop_only)",
                metrics={"chemprop_available": True},
            )
        except ImportError:
            return BaselineResult(
                model="chemprop_v2",
                strategy="none",
                status="blocked",
                failure_reason="chemprop package not installed locally",
                metrics={"chemprop_available": False},
            )

    @staticmethod
    def empty_slot(model: str, strategy: str, reason: str) -> BaselineResult:
        return BaselineResult(
            model=model,
            strategy=strategy,
            status="blocked",
            failure_reason=reason,
        )
