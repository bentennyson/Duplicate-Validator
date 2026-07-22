import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import Levenshtein



def _canonicalize(chunk: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Reduce a raw chunk to the four canonical columns using the user's mapping."""
    out = pd.DataFrame(index=chunk.index)
    out["title"] = chunk[mapping["title"]].astype(str) if mapping.get("title") else ""
    if mapping.get("authors"):
        a = chunk[mapping["authors"]].astype(str)
        a = (a.str.replace(r"\s*;\s*", "|", regex=True)
              .str.replace(r"\s+and\s+", "|", regex=True))
        out["authors"] = a
    else:
        out["authors"] = "Unknown"
    out["venue"] = chunk[mapping["venue"]].astype(str) if mapping.get("venue") else "Unknown"
    if mapping.get("year"):
        yr = pd.to_numeric(chunk[mapping["year"]], errors="coerce")
        if yr.isna().mean() > 0.5:
            yr = pd.to_numeric(
                chunk[mapping["year"]].astype(str).str.extract(r"(\d{4})")[0],
                errors="coerce")
        out["year"] = yr
    else:
        out["year"] = 2023
    return out


def read_records_chunked(path, mapping, is_json_lines=False,
                         chunksize=50_000, progress=None, max_records=None):
    """
    Stream a CSV / JSON / JSON-lines file from disk into a compact dataframe
    containing only the mapped columns. Memory use is proportional to the
    number of records kept, never to the raw file size.

    progress: optional callable(total_records_so_far)
    max_records: optional hard cap (safety valve on small machines)
    """
    frames, total = [], 0

    if str(path).lower().endswith(".csv"):
        usecols = [v for v in mapping.values() if v]  # read ONLY mapped columns
        reader = pd.read_csv(path, chunksize=chunksize, dtype=str,
                             usecols=usecols, on_bad_lines="skip")
        for chunk in reader:
            frames.append(_canonicalize(chunk, mapping))
            total += len(chunk)
            if progress:
                progress(total)
            if max_records and total >= max_records:
                break

    elif is_json_lines:
        keep = {v for v in mapping.values() if v}
        rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append({k: obj.get(k) for k in keep})  # keep only mapped keys
                if len(rows) >= chunksize:
                    frames.append(_canonicalize(pd.DataFrame(rows), mapping))
                    total += len(rows)
                    rows = []
                    if progress:
                        progress(total)
                    if max_records and total >= max_records:
                        break
        if rows and not (max_records and total >= max_records):
            frames.append(_canonicalize(pd.DataFrame(rows), mapping))
            total += len(rows)
            if progress:
                progress(total)

    else:
        # Plain JSON array — no true streaming possible; suitable for
        # moderate files only. Huge datasets should be CSV or JSON-lines.
        df_raw = pd.read_json(path)
        if max_records:
            df_raw = df_raw.head(max_records)
        frames.append(_canonicalize(df_raw, mapping))
        total = len(df_raw)
        if progress:
            progress(total)

    if not frames:
        return pd.DataFrame(columns=["title", "authors", "venue", "year"])

    df = pd.concat(frames, ignore_index=True)
    df["venue"] = df["venue"].astype("category")
    return df




_STOPWORDS = frozenset("""
a an the of on in for and to with using via from by at is are as its
based new study analysis approach method system model data learning
""".split())

_token_re = re.compile(r"[a-z0-9]+")


def _title_tokens(title: str, max_tokens=12):
    toks = [t for t in _token_re.findall(title.lower())
            if len(t) > 2 and t not in _STOPWORDS]
    # Cap tokens per title so index building and pair-refinement stay
    # linear even for very long titles.
    return set(toks[:max_tokens])


def generate_candidate_pairs_scalable(df, threshold=0.5, max_pairs=500,
                                      progress=None, max_block_size=100,
                                      pair_budget=3_000_000):
    """
    Find likely-duplicate pairs without all-pairs comparison, in bounded time.

    - Inverted index: title token -> record indices.
    - Blocks larger than max_block_size are refined on rare token *pairs*;
      truly enormous blocks are skipped (no discriminative signal).
    - EVERY inner iteration (including skipped pairs) counts toward
      pair_budget, so runtime is hard-capped regardless of dataset size.
    - progress(fraction) is called with a 0..1 float as work proceeds.

    Returns a list of (idx_a, idx_b, similarity), best first.
    """
    titles = df["title"].fillna("").astype(str).tolist()
    n = len(titles)

    if progress:
        progress(0.02)


    blocks = defaultdict(list)
    for idx in range(n):
        for tok in _title_tokens(titles[idx]):
            blocks[tok].append(idx)

    if progress:
        progress(0.10)

   
    refined = []
    refine_cap = max_block_size * 20
    for ids in blocks.values():
        L = len(ids)
        if L < 2:
            continue
        if L <= max_block_size:
            refined.append(ids)
        elif L <= refine_cap:
            sub = defaultdict(list)
            for idx in ids:
                toks = sorted(_title_tokens(titles[idx], max_tokens=8))
                for a in range(len(toks)):
                    for b in range(a + 1, len(toks)):
                        sub[(toks[a], toks[b])].append(idx)
            for ids2 in sub.values():
                if 2 <= len(ids2) <= max_block_size:
                    refined.append(ids2)
       

    blocks = None  # free memory
    # Rarest (most discriminative) blocks first
    refined.sort(key=len)
    total_blocks = len(refined)

    if progress:
        progress(0.20)

    
    seen = set()
    results = {}
    work = 0                     
    prune_at = max_pairs * 20

    def _prune():
        nonlocal results
        best = sorted(results.items(), key=lambda kv: kv[1],
                      reverse=True)[:max_pairs * 5]
        results = dict(best)

    done = False
    for bi, ids in enumerate(refined):
        k = len(ids)
        for a in range(k):
            i = ids[a]
            ti = titles[i]
            li = len(ti)
            for b in range(a + 1, k):
                work += 1
                if work >= pair_budget:
                    done = True
                    break
                j = ids[b]
                key = (i, j) if i < j else (j, i)
                if key in seen:
                    continue
                seen.add(key)

                tj = titles[j]
                lj = len(tj)
                m = max(li, lj)
                if m == 0:
                    continue
                # Length pre-filter: can't reach threshold if lengths differ
                # too much
                if abs(li - lj) / m > (1 - threshold):
                    continue
                sim = 1 - Levenshtein.distance(ti.lower(), tj.lower()) / m
                if sim > threshold:
                    results[key] = min(sim, 0.9999)
                    if len(results) >= prune_at:
                        _prune()
            if done:
                break
        if done:
            break
        if progress and bi % 2000 == 0 and total_blocks:
            progress(0.20 + 0.80 * max(bi / total_blocks, work / pair_budget))

    if progress:
        progress(1.0)

    ranked = sorted(results.items(), key=lambda kv: kv[1],
                    reverse=True)[:max_pairs]
    return [(i, j, s) for (i, j), s in ranked]
