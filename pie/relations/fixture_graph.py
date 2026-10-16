"""Synthetic graph constructor for the offline software fixture."""

MODEL = "fixture-model@sha"
CLT = "fixture-clt@sha"
DATASET_HASH = "a" * 64


def graph(pair_id: int, graph_id: str, values: list[float]) -> dict:
    nodes = [{"id": f"s{i}", "type": "feature", "layer": 1,
              "position": 2, "position_role": "name", "feature_index": 10 + i}
             for i in range(5)]
    nodes.append({"id": "t", "type": "feature", "layer": 4,
                  "position": 6, "feature_index": 91})
    nodes.append({"id": "error", "type": "error", "layer": 4, "position": 6})
    edges = [{"source": f"s{i}", "target": "t", "value": value,
              "measurement_status": "computed_nonzero"}
             for i, value in enumerate(values)]
    edges.append({"source": "error", "target": "t", "value": 0.5,
                  "measurement_status": "computed_nonzero"})
    edges.append({"source": "s0", "target": "error", "value": None,
                  "measurement_status": "unmeasured_omitted"})
    return {"graph_id": graph_id, "prompt_pair_id": pair_id,
            "prompt": "Alex handed a book to Sam." if pair_id == 1 else "Taylor handed a note to Morgan.",
            "model_revision": MODEL, "clt_revision": CLT,
            "pair_dataset_sha256": DATASET_HASH,
            "edge_semantics": "synthetic_signed_feature_attribution",
            "normalization": "synthetic_shared_target_units",
            "position_policy": "synthetic_token_indices_no_bos",
            "support_policy": "explicit_computed_entries",
            "nodes": nodes, "edges": edges}

