import json
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path
from .config import config
from .models import CostRecord

class CostTracker:
    """Singleton cost and token accounting engine."""
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._records: List[CostRecord] = []
                cls._instance._load_records()
        return cls._instance

    def _load_records(self) -> None:
        if config.COST_LOG_FILE.exists():
            try:
                with open(config.COST_LOG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._records = [CostRecord(**item) for item in data]
            except Exception:
                self._records = []

    def save_records(self) -> None:
        with self._lock:
            try:
                with open(config.COST_LOG_FILE, "w", encoding="utf-8") as f:
                    json.dump([r.model_dump() for r in self._records], f, indent=2)
            except Exception as e:
                print(f"[COST_TRACKER] Warning: failed to save cost log: {e}")

    def record_usage(
        self,
        provider: str,
        model: str,
        task: str,
        input_tokens: int,
        output_tokens: int,
        status: str = "SUCCESS"
    ) -> CostRecord:
        """Calculate exact USD cost based on token counts and log it.

        Pricing comes from config.get_pricing(), which raises
        PricingNotConfiguredError (rather than silently returning $0.00) for
        any real (non-MOCK_MODE) provider/model pair without a verified
        PRICING entry -- see config.py. This call is allowed to propagate:
        a real API call whose cost cannot be accounted for must be a loud,
        fixable configuration gap, not a fabricated zero in the cost log.
        """
        pricing_info = config.get_pricing(provider, model)
        input_price_per_m = pricing_info.get("input", 0.0)
        output_price_per_m = pricing_info.get("output", 0.0)
        request_fixed = pricing_info.get("request", 0.0)

        cost = ((input_tokens / 1_000_000.0) * input_price_per_m) + \
               ((output_tokens / 1_000_000.0) * output_price_per_m) + request_fixed

        rec = CostRecord(
            provider=provider,
            model=model,
            task=task,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=round(cost, 6),
            status=status
        )

        with self._lock:
            self._records.append(rec)
        self.save_records()
        return rec

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate total tokens and cost by provider and task, plus a
        per-(provider, model) breakdown (usage_by_model) that clearly
        identifies model / input tokens / output tokens / calls / estimated
        cost for each distinct model actually used -- this is what makes a
        newly-configured, correctly- or incorrectly-priced model visible at
        a glance instead of hiding inside a provider-level total."""
        total_cost = 0.0
        by_provider: Dict[str, float] = {}
        by_task: Dict[str, float] = {}
        tokens_by_provider: Dict[str, Dict[str, int]] = {}
        by_model: Dict[tuple, Dict[str, Any]] = {}
        by_task_usage: Dict[str, Dict[str, Any]] = {}

        for r in self._records:
            total_cost += r.estimated_cost
            by_provider[r.provider] = round(by_provider.get(r.provider, 0.0) + r.estimated_cost, 6)
            by_task[r.task] = round(by_task.get(r.task, 0.0) + r.estimated_cost, 6)

            if r.provider not in tokens_by_provider:
                tokens_by_provider[r.provider] = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
            tokens_by_provider[r.provider]["input_tokens"] += r.input_tokens
            tokens_by_provider[r.provider]["output_tokens"] += r.output_tokens
            tokens_by_provider[r.provider]["calls"] += 1

            model_key = (r.provider, r.model)
            if model_key not in by_model:
                by_model[model_key] = {
                    "provider": r.provider, "model": r.model, "calls": 0,
                    "input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0,
                }
            entry = by_model[model_key]
            entry["calls"] += 1
            entry["input_tokens"] += r.input_tokens
            entry["output_tokens"] += r.output_tokens
            entry["estimated_cost"] = round(entry["estimated_cost"] + r.estimated_cost, 6)

            if r.task not in by_task_usage:
                by_task_usage[r.task] = {"task": r.task, "calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0}
            task_entry = by_task_usage[r.task]
            task_entry["calls"] += 1
            task_entry["input_tokens"] += r.input_tokens
            task_entry["output_tokens"] += r.output_tokens
            task_entry["estimated_cost"] = round(task_entry["estimated_cost"] + r.estimated_cost, 6)

        return {
            "total_estimated_cost_usd": round(total_cost, 6),
            "cost_by_provider": by_provider,
            "cost_by_task": by_task,
            "usage_by_provider": tokens_by_provider,
            "usage_by_model": sorted(by_model.values(), key=lambda e: (e["provider"], e["model"])),
            # Distinguishes research/synthesis/generation/validation model
            # usage from RAG-embedding, question-bank-embedding, and
            # deduplication-embedding usage -- each is logged under its own
            # task name (see task_name= call sites in generator_service.py,
            # synthesis_service.py, validator_service.py, rag_service.py,
            # db_loader.py, deduplicator_service.py) so this needed no new
            # tagging, only this aggregation.
            "usage_by_task": sorted(by_task_usage.values(), key=lambda e: e["task"]),
            "total_calls": len(self._records)
        }

    def clear(self) -> None:
        with self._lock:
            self._records = []
        self.save_records()

cost_tracker = CostTracker()
