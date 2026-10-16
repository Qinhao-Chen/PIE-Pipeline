"""Task-D occurrence-atomic, bounded fixed-write exchange search.

The scorer is injected. This module neither loads a subject model nor infers graph
edges. Graph evidence must already be measured and mapped to occurrence keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from hashlib import sha256
from itertools import combinations
import json
import math
import random
import sqlite3
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence


CONTEXT_FIELDS = (
    "model_revision", "clt_revision", "pair_dataset_sha256",
    "prompt_pair_id", "protocol_label", "evaluator_sha256",
    "evaluator_config_sha256", "intervention_sha256",
)


def compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def occurrence_key(value: Sequence[object]) -> str:
    """Validate and canonicalize the pie-parallel-v1 occurrence-key array."""
    if len(value) != 4 or not isinstance(value[3], (list, tuple)) or len(value[3]) != 4:
        raise ValueError("occurrence key must be [dataset_hash, pair_id, position, feature_key]")
    dataset, pair_id, position, feature = value
    if not isinstance(dataset, str) or not dataset or not isinstance(pair_id, str) or not pair_id:
        raise ValueError("dataset hash and pair ID must be nonempty strings")
    if not isinstance(position, int) or isinstance(position, bool) or position < 0:
        raise ValueError("aligned source position must be a nonnegative integer")
    model, clt, layer, index = feature
    if not isinstance(model, str) or not model or not isinstance(clt, str) or not clt:
        raise ValueError("model and CLT revisions must be nonempty strings")
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in (layer, index)):
        raise ValueError("layer and feature index must be nonnegative integers")
    return compact([dataset, pair_id, position, [model, clt, layer, index]])


def key_array(key: str) -> list[object]:
    value = json.loads(key)
    if occurrence_key(value) != key:
        raise ValueError("noncanonical occurrence key")
    return value


@dataclass(frozen=True)
class SelectionView:
    retained_occurrences: frozenset[str]
    retained_write_ids: frozenset[str]


@dataclass(frozen=True)
class GraphEvidence:
    context: Mapping[str, str]
    # Pair keys are sorted canonical occurrence-key strings. Values describe
    # measured connected or convergent evidence; no inferred edges are accepted.
    measured_pairs: Mapping[tuple[str, str], str]
    provenance: str

    def validate(self, context: Mapping[str, str]) -> None:
        for name in ("model_revision", "clt_revision", "pair_dataset_sha256", "prompt_pair_id"):
            if self.context.get(name) != context.get(name):
                raise ValueError(f"graph context mismatch: {name}")
        if not self.provenance:
            raise ValueError("graph provenance is required")
        for pair, kind in self.measured_pairs.items():
            if len(pair) != 2 or pair[0] == pair[1] or tuple(sorted(pair)) != pair:
                raise ValueError("graph pair must contain two sorted, distinct occurrence keys")
            for key in pair:
                key_array(key)
            if kind not in ("connected", "convergent"):
                raise ValueError("graph evidence must be measured connected or convergent")


@dataclass(frozen=True)
class SearchConfig:
    max_candidate_evaluations: int = 32
    max_rounds: int = 2
    incoming_pool_size: int = 8
    outgoing_pool_size: int = 6
    outgoing_pair_limit: int = 6
    max_proposal_attempts: int = 256
    random_fraction: float = 0.25
    acceptance_tolerance: float = 1e-6
    verification_tolerance: float = 1e-7
    seed: int = 0

    def validate(self) -> None:
        if self.max_candidate_evaluations < 0 or self.max_rounds < 1:
            raise ValueError("invalid evaluation or round limit")
        if not (1 <= self.incoming_pool_size <= 16 and 1 <= self.outgoing_pool_size <= 16):
            raise ValueError("pool sizes must be in [1, 16]")
        if self.outgoing_pair_limit < 1 or self.max_proposal_attempts < 1:
            raise ValueError("invalid proposal limit")
        if not 0 <= self.random_fraction < 1:
            raise ValueError("random fraction must be in [0, 1)")
        if self.acceptance_tolerance < 0 or self.verification_tolerance < 0:
            raise ValueError("negative numerical tolerance")


class SQLiteScoreCache:
    """Task-local durable score cache. Caller owns the path and protocol context."""

    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL NOT NULL)")
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def get(self, key: str) -> float | None:
        row = self.db.execute("SELECT score FROM scores WHERE key=?", (key,)).fetchone()
        return None if row is None else float(row[0])

    def put(self, key: str, score: float) -> None:
        self.db.execute("INSERT OR REPLACE INTO scores(key, score) VALUES (?, ?)", (key, score))
        self.db.commit()


def score_cache_key(context: Mapping[str, str], selection: frozenset[str]) -> str:
    missing = [name for name in CONTEXT_FIELDS if not context.get(name)]
    if missing:
        raise ValueError(f"incomplete evaluator context: {missing}")
    payload = {"context": {name: context[name] for name in CONTEXT_FIELDS},
               "retained_occurrences": sorted(selection)}
    return sha256(compact(payload).encode()).hexdigest()


@dataclass
class SearchResult:
    mode: str
    status: str
    initial: frozenset[str]
    final: frozenset[str]
    initial_j: float | None = None
    final_j: float | None = None
    candidate_evaluations: int = 0
    cache_hits: int = 0
    uncached_candidate_calls: int = 0
    initialization_calls: int = 0
    verification_calls: int = 0
    proposal_attempts: int = 0
    accepted_moves: int = 0
    elapsed_seconds: float = 0.0
    trace: list[dict] = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> dict:
        data = asdict(self)
        data["initial"] = [key_array(k) for k in sorted(self.initial)]
        data["final"] = [key_array(k) for k in sorted(self.final)]
        return data


def _view(selection: frozenset[str], universe: Mapping[str, Sequence[str]]) -> SelectionView:
    # Atomic occurrence choice: every decoder-write ID attached to an occurrence
    # enters the evaluator together.
    writes = frozenset(write for key in selection for write in universe[key])
    return SelectionView(selection, writes)


def _score(
    selection: frozenset[str], context: Mapping[str, str],
    universe: Mapping[str, Sequence[str]], scorer: Callable[[SelectionView], float],
    cache: SQLiteScoreCache | None, force: bool = False,
) -> tuple[float, bool]:
    cache_key = score_cache_key(context, selection)
    if cache is not None and not force:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit, True
    value = float(scorer(_view(selection, universe)))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid KL score: {value}")
    if cache is not None:
        cache.put(cache_key, value)
    return value, False


def _candidate_pools(
    selection: frozenset[str], attribution: Mapping[str, float],
    config: SearchConfig,
) -> tuple[list[str], list[str]]:
    # Legacy global FAP ranks by absolute attribution, including negative scores.
    outgoing = sorted(selection, key=lambda k: (abs(attribution[k]), k))[:config.outgoing_pool_size]
    incoming = sorted(set(attribution) - selection,
                      key=lambda k: (-abs(attribution[k]), k))[:config.incoming_pool_size]
    return outgoing, incoming


def _proposals(
    mode: str, selection: frozenset[str], attribution: Mapping[str, float],
    config: SearchConfig, graph: GraphEvidence | None, round_index: int, quota: int,
) -> list[tuple[tuple[str, ...], tuple[str, ...], str]]:
    outgoing, incoming = _candidate_pools(selection, attribution, config)
    size = 1 if mode == "single" else 2
    if len(outgoing) < size or len(incoming) < size:
        return []
    # These schedules depend only on state/round/seed, so pair controls use
    # exactly the same outgoing group at a given proposal slot. Incoming pair
    # choice is the controlled difference.
    def stream(kind: str):
        payload = compact([config.seed, round_index, sorted(selection), kind])
        return random.Random(int(sha256(payload.encode()).hexdigest()[:16], 16))

    out_groups = list(combinations(outgoing, size))
    stream(f"out-{size}").shuffle(out_groups)
    if size == 2:
        out_groups = out_groups[:config.outgoing_pair_limit]
    in_groups = list(combinations(incoming, size))
    stream(f"in-{size}").shuffle(in_groups)
    guided = []
    if mode == "guided_pair":
        assert graph is not None
        guided = [pair for pair in in_groups if tuple(sorted(pair)) in graph.measured_pairs]
        if not guided:
            return []
    n_random = round(quota * config.random_fraction)
    proposals = []
    limit = min(config.max_proposal_attempts, max(quota * 4, quota))
    for slot in range(limit):
        drop = out_groups[slot % len(out_groups)]
        incoming_slot = slot // len(out_groups) + slot % len(out_groups)
        random_slot = (mode == "guided_pair" and n_random > 0 and
                       ((slot + 1) * n_random // quota) != (slot * n_random // quota))
        if mode == "guided_pair" and not random_slot:
            add = guided[incoming_slot % len(guided)]
            source = graph.measured_pairs[tuple(sorted(add))]
        else:
            add = in_groups[incoming_slot % len(in_groups)]
            source = "single" if size == 1 else "random_pair"
        proposals.append((drop, add, source))
    return proposals


def run_exchange(
    *, mode: str, initial: Sequence[str], universe: Mapping[str, Sequence[str]],
    attribution: Mapping[str, float], context: Mapping[str, str],
    scorer: Callable[[SelectionView], float], config: SearchConfig,
    graph: GraphEvidence | None = None, cache: SQLiteScoreCache | None = None,
) -> SearchResult:
    """Run a best-of-round search with a cap over unique candidate sets.

    Initialization and forced final verification are separate from the extra
    candidate cap. One accepted move ends a round and regenerates proposals.
    """
    start = time.perf_counter()
    config.validate()
    if mode not in ("fap", "single", "random_pair", "guided_pair"):
        raise ValueError("unknown mode")
    selected = frozenset(initial)
    if len(selected) != len(initial) or not selected <= universe.keys():
        raise ValueError("initial set has duplicates or unknown occurrences")
    if set(universe) != set(attribution):
        raise ValueError("every eligible occurrence needs its verified base attribution")
    for key, writes in universe.items():
        arr = key_array(key)
        if arr[0] != context.get("pair_dataset_sha256") or arr[1] != context.get("prompt_pair_id"):
            raise ValueError("occurrence/prompt context mismatch")
        if arr[3][0] != context.get("model_revision") or arr[3][1] != context.get("clt_revision"):
            raise ValueError("occurrence/checkpoint context mismatch")
        if not writes or not all(isinstance(w, str) and w for w in writes):
            raise ValueError("each occurrence needs explicit decoder-write IDs")
        if not math.isfinite(float(attribution[key])):
            raise ValueError("nonfinite base attribution")
    score_cache_key(context, selected)
    if mode == "guided_pair":
        if graph is None:
            return SearchResult(mode, "blocked_missing_graph", selected, selected,
                                elapsed_seconds=time.perf_counter() - start)
        graph.validate(context)
    result = SearchResult(mode, "running", selected, selected)
    try:
        current_j, hit = _score(selected, context, universe, scorer, cache)
        result.initial_j = current_j
        result.initialization_calls = 0 if hit else 1
        result.cache_hits += int(hit)
        if mode == "fap":
            result.status = "baseline"
        else:
            seen = {selected}
            per_round = math.ceil(config.max_candidate_evaluations / config.max_rounds)
            for round_index in range(config.max_rounds):
                remaining = config.max_candidate_evaluations - result.candidate_evaluations
                if remaining <= 0:
                    result.status = "budget_exhausted"
                    break
                quota = min(per_round, remaining)
                proposals = _proposals(mode, selected, attribution, config, graph,
                                       round_index, quota)
                if not proposals:
                    result.status = "no_eligible_proposals"
                    break
                best = None
                round_evaluations = 0
                trace_start = len(result.trace)
                for outgoing, incoming, guidance in proposals:
                    if round_evaluations >= quota or result.proposal_attempts >= config.max_proposal_attempts:
                        break
                    result.proposal_attempts += 1
                    candidate = (selected - frozenset(outgoing)) | frozenset(incoming)
                    if len(candidate) != len(selected) or candidate in seen:
                        continue
                    seen.add(candidate)
                    result.candidate_evaluations += 1
                    round_evaluations += 1
                    row = {"round": round_index + 1, "before": [key_array(k) for k in sorted(selected)],
                           "after": [key_array(k) for k in sorted(candidate)],
                           "outgoing": [key_array(k) for k in outgoing],
                           "incoming": [key_array(k) for k in incoming],
                           "guidance": guidance, "j_before": current_j,
                           "j_after": None, "accepted": False,
                           "candidate_evaluation": result.candidate_evaluations,
                           "cache_hit": False, "seconds": None, "error": None}
                    t0 = time.perf_counter()
                    try:
                        value, hit = _score(candidate, context, universe, scorer, cache)
                        row["j_after"] = value
                        row["cache_hit"] = hit
                        result.cache_hits += int(hit)
                        result.uncached_candidate_calls += int(not hit)
                        if value < current_j - config.acceptance_tolerance:
                            if best is None or value < best[0]:
                                best = (value, candidate, row)
                    except Exception as exc:  # record failed forwards; never accept them
                        result.uncached_candidate_calls += 1
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    finally:
                        row["seconds"] = time.perf_counter() - t0
                        result.trace.append(row)
                if best is None:
                    if not round_evaluations:
                        result.status = "no_new_candidates"
                    elif all(row["error"] for row in result.trace[trace_start:]):
                        result.status = "candidate_errors"
                    else:
                        result.status = "no_improvement"
                    break
                current_j, selected, accepted_row = best
                accepted_row["accepted"] = True
                result.accepted_moves += 1
                result.final = selected
                result.final_j = current_j
                result.status = "improved"
                if result.proposal_attempts >= config.max_proposal_attempts:
                    result.status = "proposal_attempt_limit"
                    break
            else:
                if result.status == "improved":
                    result.status = "round_limit"
        # Force one deterministic-protocol final check even when a warm cache is
        # present. This is an extra call outside the candidate budget.
        if result.accepted_moves and result.status in (
                "no_improvement", "no_new_candidates", "no_eligible_proposals",
                "candidate_errors"):
            result.status = "improved_then_" + result.status
        verification, _ = _score(selected, context, universe, scorer, cache, force=True)
        result.verification_calls = 1
        if abs(verification - current_j) > config.verification_tolerance:
            result.status = "verification_mismatch"
            result.error = f"cached/search J={current_j}, verified J={verification}"
        result.final = selected
        result.final_j = verification
    except Exception as exc:
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
    result.elapsed_seconds = time.perf_counter() - start
    return result
