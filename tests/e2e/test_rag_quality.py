import datetime as dt
import json
import os
import tempfile
import uuid
from dataclasses import dataclass

import httpx
import pytest


@dataclass
class QueryEval:
    query: str
    expected_file_id: str
    rank: int | None
    hit: bool
    evidence_hit: bool | None


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_auth_headers() -> dict[str, str]:
    token = os.getenv("RAG_EVAL_AUTH_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}

    secret = os.getenv("RAG_EVAL_JWT_SECRET")
    if not secret:
        return {}

    import jwt

    user_id = os.getenv("RAG_EVAL_USER_ID", "rag-eval-user")
    payload = {
        "id": user_id,
        "exp": dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1),
    }
    encoded = jwt.encode(payload, secret, algorithm="HS256")
    return {"Authorization": f"Bearer {encoded}"}


def _extract_doc_and_score(item):
    # Expected shape from FastAPI serialization of tuple(Document, score):
    # [ {"page_content": "...", "metadata": {...}}, 0.123 ]
    if not isinstance(item, list) or len(item) < 2:
        return None, None
    doc = item[0] if isinstance(item[0], dict) else None
    score = item[1]
    return doc, score


def _rank_of_file_id(results, target_file_id: str) -> int | None:
    for idx, item in enumerate(results, start=1):
        doc, _ = _extract_doc_and_score(item)
        if not doc:
            continue
        metadata = doc.get("metadata") or {}
        if metadata.get("file_id") == target_file_id:
            return idx
    return None


def _contains_expected_snippet(results, target_file_id: str, expected_snippets: list[str]) -> bool:
    lowered = [snippet.lower() for snippet in expected_snippets]
    for item in results:
        doc, _ = _extract_doc_and_score(item)
        if not doc:
            continue
        metadata = doc.get("metadata") or {}
        if metadata.get("file_id") != target_file_id:
            continue
        content = (doc.get("page_content") or "").lower()
        if any(snippet in content for snippet in lowered):
            return True
    return False


@pytest.mark.integration
def test_rag_quality_benchmark():
    if not _to_bool(os.getenv("RAG_EVAL_ENABLED"), default=False):
        pytest.skip(
            "Set RAG_EVAL_ENABLED=1 to run end-to-end RAG quality benchmark against a live API."
        )

    base_url = os.getenv("RAG_EVAL_BASE_URL", "http://localhost:8000").rstrip("/")
    dataset_path = os.getenv(
        "RAG_EVAL_DATASET", "tests/e2e/rag_eval_dataset.json"
    )
    k_default = int(os.getenv("RAG_EVAL_K", "5"))
    min_hit_rate = float(os.getenv("RAG_EVAL_MIN_HIT_RATE", "0.90"))
    min_mrr = float(os.getenv("RAG_EVAL_MIN_MRR", "0.70"))
    min_context_precision = float(os.getenv("RAG_EVAL_MIN_CONTEXT_PRECISION", "0.80"))
    timeout = float(os.getenv("RAG_EVAL_TIMEOUT_SECONDS", "120"))

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    documents = dataset.get("documents", [])
    queries = dataset.get("queries", [])
    assert documents, "Dataset must include at least one document."
    assert queries, "Dataset must include at least one query."

    headers = _build_auth_headers()

    run_id = uuid.uuid4().hex[:8]
    id_map: dict[str, str] = {
        d["id"]: f"rag_eval_{run_id}_{d['id']}" for d in documents if "id" in d
    }
    upload_ids = list(id_map.values())

    with httpx.Client(timeout=timeout) as client:
        health = client.get(f"{base_url}/health", headers=headers)
        assert health.status_code == 200, (
            f"Health check failed at {base_url}/health with status {health.status_code}: "
            f"{health.text}"
        )

        try:
            # Best-effort pre-cleanup in case prior run used same IDs.
            client.request(
                "DELETE", f"{base_url}/documents", headers=headers, json=upload_ids
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                for doc in documents:
                    source_id = doc["id"]
                    file_id = id_map[source_id]
                    filename = doc.get("filename", f"{source_id}.txt")
                    content = doc["content"]
                    filepath = os.path.join(tmpdir, filename)

                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)

                    with open(filepath, "rb") as f:
                        response = client.post(
                            f"{base_url}/embed",
                            headers=headers,
                            data={"file_id": file_id},
                            files={"file": (filename, f, "text/plain")},
                        )
                    assert response.status_code == 200, (
                        f"Embed failed for {filename} ({file_id}) with status "
                        f"{response.status_code}: {response.text}"
                    )

            evaluations: list[QueryEval] = []
            evidence_applicable = 0

            for q in queries:
                query_text = q["query"]
                expected_file_id = id_map[q["expected_file_id"]]
                k = int(q.get("k", k_default))

                payload = {
                    "query": query_text,
                    "file_ids": upload_ids,
                    "k": k,
                }
                response = client.post(
                    f"{base_url}/query_multiple", headers=headers, json=payload
                )

                if response.status_code == 404:
                    results = []
                else:
                    assert response.status_code == 200, (
                        f"Query failed for '{query_text}' with status "
                        f"{response.status_code}: {response.text}"
                    )
                    results = response.json()

                rank = _rank_of_file_id(results, expected_file_id)
                hit = rank is not None and rank <= k

                expected_snippets = q.get("expected_snippets") or []
                if expected_snippets:
                    evidence_applicable += 1
                    evidence_hit = _contains_expected_snippet(
                        results, expected_file_id, expected_snippets
                    )
                else:
                    evidence_hit = None

                evaluations.append(
                    QueryEval(
                        query=query_text,
                        expected_file_id=expected_file_id,
                        rank=rank,
                        hit=hit,
                        evidence_hit=evidence_hit,
                    )
                )

        finally:
            # Cleanup uploaded docs from vector DB.
            client.request(
                "DELETE", f"{base_url}/documents", headers=headers, json=upload_ids
            )

    total = len(evaluations)
    hit_rate = sum(1 for e in evaluations if e.hit) / total
    mrr = sum((1 / e.rank) if e.rank else 0 for e in evaluations) / total
    if evidence_applicable:
        context_precision = (
            sum(1 for e in evaluations if e.evidence_hit is True) / evidence_applicable
        )
    else:
        context_precision = 1.0

    misses = [e for e in evaluations if not e.hit]
    context_misses = [e for e in evaluations if e.evidence_hit is False]

    assert hit_rate >= min_hit_rate, (
        f"Hit@k below threshold: got {hit_rate:.3f}, expected >= {min_hit_rate:.3f}. "
        f"Misses: {[{'query': m.query, 'expected_file_id': m.expected_file_id, 'rank': m.rank} for m in misses]}"
    )
    assert mrr >= min_mrr, f"MRR below threshold: got {mrr:.3f}, expected >= {min_mrr:.3f}."
    assert context_precision >= min_context_precision, (
        f"Context precision below threshold: got {context_precision:.3f}, "
        f"expected >= {min_context_precision:.3f}. "
        f"Context misses: {[{'query': m.query, 'expected_file_id': m.expected_file_id, 'rank': m.rank} for m in context_misses]}"
    )
