"""Direct citations between papers in B's sample (paper A's reference list contains paper B).

Fetches each sampled paper's ``referenced_works`` from OpenAlex (100 ids per request,
``select=id,referenced_works``), caches them in data/cache/openalex/references.jsonl so
a rerun only asks for what is missing, and keeps the citations whose cited paper is also
in the sample.

Output: data/paper_citations.csv  (citing_id, cited_id, citing_year, cited_year)

Run: python -m src.topic_radar.citations
The API key is read by OpenAlexClient from the environment / .env and is never printed.
"""

from __future__ import annotations

import json

import pandas as pd

from src.common.contracts import DATA_DIR

REF_CACHE = DATA_DIR / "cache" / "openalex" / "references.jsonl"
CITATIONS_CSV = DATA_DIR / "paper_citations.csv"
BATCH = 100


def short_id(value: str) -> str:
    return str(value).rsplit("/", 1)[-1]


def load_refs(path=REF_CACHE) -> dict[str, list[str]]:
    refs = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                refs[row["id"]] = row["refs"]
    return refs


def fetch_refs(paper_ids: list[str], client=None, path=REF_CACHE) -> dict[str, list[str]]:
    """Reference lists for ``paper_ids``, from the cache where possible."""
    refs = load_refs(path)
    todo = [p for p in paper_ids if p not in refs]
    if not todo:
        return refs
    if client is None:
        from src.research_map.openalex import OpenAlexClient
        client = OpenAlexClient.from_env()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        for start in range(0, len(todo), BATCH):
            batch = todo[start:start + BATCH]
            payload = client._get("works", {"filter": f"openalex_id:{'|'.join(batch)}",
                                            "select": "id,referenced_works", "per_page": BATCH})
            found = set()
            for work in payload.get("results", []):
                pid = short_id(work["id"])
                found.add(pid)
                refs[pid] = [short_id(r) for r in work.get("referenced_works") or []]
                out.write(json.dumps({"id": pid, "refs": refs[pid]}) + "\n")
            for pid in set(batch) - found:  # not returned (merged or removed): record as empty
                refs[pid] = []
                out.write(json.dumps({"id": pid, "refs": []}) + "\n")
            out.flush()
            print(f"  {min(start + BATCH, len(todo))}/{len(todo)} papers", flush=True)
    return refs


def in_sample_citations(refs: dict[str, list[str]], years: dict[str, int]) -> pd.DataFrame:
    """Citations whose citing and cited papers are both in the sample."""
    rows = [(a, b) for a, cited in refs.items() if a in years for b in cited if b in years and b != a]
    out = pd.DataFrame(rows, columns=["citing_id", "cited_id"]).drop_duplicates()
    out["citing_year"] = out["citing_id"].map(years)
    out["cited_year"] = out["cited_id"].map(years)
    return out


def main() -> None:
    nodes = pd.read_csv(DATA_DIR / "network_nodes.csv", usecols=["paper_id", "year"])
    years = dict(zip(nodes["paper_id"], nodes["year"].astype(int)))
    refs = fetch_refs(list(years))
    cites = in_sample_citations(refs, years)
    cites.to_csv(CITATIONS_CSV, index=False)
    with_refs = sum(1 for p in years if refs.get(p))
    print(f"{len(years)} papers, {with_refs} with a reference list, "
          f"{int(sum(len(refs.get(p, [])) for p in years))} references in total")
    print(f"{len(cites)} citations inside the sample, touching "
          f"{len(set(cites['citing_id']) | set(cites['cited_id']))} papers -> {CITATIONS_CSV.name}")


if __name__ == "__main__":
    main()
