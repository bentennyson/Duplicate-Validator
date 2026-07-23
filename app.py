import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein


st.set_page_config(
    page_title="Duplicate Validator",
    page_icon="◈",
    layout="wide",
)

st.markdown("""
<style>
    /* Typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
    }
    h1, h2, h3 { letter-spacing: -0.01em; color: #292524; }

    .block-container { padding-top: 2.5rem; max-width: 1100px; }

    /* App header */
    .app-title {
        font-size: 1.9rem; font-weight: 700; color: #1C1917;
        margin-bottom: 0.15rem; letter-spacing: -0.02em;
    }
    .app-subtitle {
        color: #78716C; font-size: 1rem; margin-bottom: 1.75rem;
    }

    /* Paper cards */
    .paper-card {
        background: #FFFFFF;
        border: 1px solid #E7E5E4;
        border-radius: 10px;
        padding: 1.25rem 1.4rem;
        height: 100%;
    }
    .paper-label {
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; color: #A8A29E; margin-bottom: 0.6rem;
    }
    .paper-title {
        font-size: 1.05rem; font-weight: 600; color: #1C1917;
        line-height: 1.4; margin-bottom: 0.75rem;
    }
    .paper-meta {
        font-size: 0.88rem; color: #57534E; line-height: 1.7;
    }
    .paper-meta span.k { color: #A8A29E; }

    /* Verdict banner */
    .verdict {
        border-radius: 10px; padding: 0.85rem 1.2rem; margin: 1.1rem 0;
        font-size: 0.95rem; border: 1px solid;
    }
    .verdict.dup   { background: #F4F3F0; border-color: #D6D3D1; color: #292524; }
    .verdict.nodup { background: #FAFAF8; border-color: #E7E5E4; color: #57534E; }
    .verdict.unsure { background: #FBF7EE; border-color: #E5DCC8; color: #57534E; }
    .verdict b { font-weight: 650; }

    /* Similarity bars */
    .sim-row { display: flex; align-items: center; margin: 0.45rem 0; }
    .sim-label { width: 90px; font-size: 0.85rem; color: #57534E; }
    .sim-track {
        flex: 1; height: 8px; background: #ECEAE7;
        border-radius: 4px; overflow: hidden;
    }
    .sim-fill { height: 100%; background: #57534E; border-radius: 4px; }
    .sim-val { width: 56px; text-align: right; font-size: 0.85rem;
               color: #292524; font-variant-numeric: tabular-nums; }

    .explain {
        background: #FFFFFF; border: 1px solid #E7E5E4; border-radius: 10px;
        padding: 1rem 1.25rem; font-size: 0.92rem; color: #44403C;
        line-height: 1.6;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px; font-weight: 550;
        border: 1px solid #D6D3D1;
    }
    .stButton > button[kind="primary"] {
        background: #292524; border-color: #292524;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1C1917; border-color: #1C1917;
    }

    /* Section headings */
    .section-head {
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; color: #A8A29E;
        margin: 1.6rem 0 0.6rem 0;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


for key, default in [
    ("current_pair_idx", 0),
    ("feedback_log", []),
    ("data_loaded", False),
    ("df", None),
    ("candidate_pairs", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


COLUMN_ALIASES = {
    "title":   ["title", "titles", "paper title", "article title", "publication title", "name"],
    "authors": ["authors", "author", "author(s)", "creator", "creators", "author names", "by"],
    "venue":   ["venue", "journal", "conference", "publication", "source", "booktitle",
                "proceedings", "publication venue", "container-title"],
    "year":    ["year", "pub_year", "pub year", "publication year", "date", "published",
                "pubdate", "issued"],
}

def auto_guess_mapping(columns):
    """Best-guess mapping from the file's columns to the four canonical fields."""
    lower = {str(c).lower().strip(): c for c in columns}
    mapping = {}
    for canon, aliases in COLUMN_ALIASES.items():
        match = None
        for a in aliases:
            if a in lower:
                match = lower[a]; break
        if match is None:
            for cl, orig in lower.items():
                if any(a in cl for a in aliases):
                    match = orig; break
        mapping[canon] = match
    return mapping

def apply_column_mapping(df, mapping):
    """Return a new dataframe with canonical columns built from the user's mapping."""
    out = pd.DataFrame(index=df.index)
    out["title"] = df[mapping["title"]].astype(str) if mapping.get("title") else ""
    if mapping.get("authors"):
        a = df[mapping["authors"]].astype(str)
        a = a.str.replace(r"\s*;\s*", "|", regex=True).str.replace(r"\s+and\s+", "|", regex=True)
        out["authors"] = a
    else:
        out["authors"] = "Unknown"
    out["venue"] = df[mapping["venue"]].astype(str) if mapping.get("venue") else "Unknown"
    if mapping.get("year"):
        yr = pd.to_numeric(df[mapping["year"]], errors="coerce")
        if yr.isna().mean() > 0.5:
            yr = pd.to_numeric(df[mapping["year"]].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
        out["year"] = yr
    else:
        out["year"] = 2023
    return out.reset_index(drop=True)

def load_raw_file(uploaded_file):
    """Load a CSV/JSON upload without assuming column names (mapping happens after)."""
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            return None, ["Unsupported file format. Please upload CSV or JSON."]
        if len(df.columns) == 0:
            return None, ["No columns found in file."]
        if len(df) < 2:
            return None, ["Dataset must have at least 2 records."]
        return df, []
    except Exception as e:
        return None, [f"Error loading file: {str(e)}"]


def generate_candidate_pairs(df, max_pairs=500):
    """Token blocking if the scalable module is available, else TF-IDF fallback."""
    try:
        from scalable_processing import generate_candidate_pairs_scalable
    except Exception:
        return _generate_candidate_pairs_legacy(df, max_pairs=max_pairs)

    n = len(df)
    bar = st.progress(0.0, text="Scanning for duplicates...") if n > 5000 else None
    def _p(frac):
        if bar is not None:
            bar.progress(min(1.0, float(frac)),
                         text=f"Scanning for duplicates... {min(1.0, float(frac)):.0%}")
    raw = generate_candidate_pairs_scalable(df, threshold=0.5,
                                            max_pairs=max_pairs, progress=_p)
    if bar is not None:
        bar.empty()
    return [{"idx_a": i, "idx_b": j, "tfidf_similarity": sim} for (i, j, sim) in raw]

def _generate_candidate_pairs_legacy(df, max_pairs=100):
    """Generate candidate duplicate pairs using TF-IDF similarity (memory-efficient)."""
    n_records = len(df)

    if n_records > 10000:
        st.warning(f"Large dataset ({n_records:,} records). Sampling to keep memory bounded.")
        sample_size = min(5000, n_records)
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df

    titles = df_sample["title"].fillna("").tolist()
    max_features = min(500, len(titles) * 2)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(titles)

    candidate_pairs = []
    batch_size = 1000
    for i in range(0, len(df_sample), batch_size):
        batch_end = min(i + batch_size, len(df_sample))
        batch_similarities = cosine_similarity(tfidf_matrix[i:batch_end], tfidf_matrix)

        for batch_idx, row_idx in enumerate(range(i, batch_end)):
            for col_idx in range(row_idx + 1, len(df_sample)):
                sim = batch_similarities[batch_idx, col_idx]
                if 0.5 < sim < 1.0:
                    if n_records > 10000:
                        orig_idx_a = df_sample.index[row_idx]
                        orig_idx_b = df_sample.index[col_idx]
                    else:
                        orig_idx_a = row_idx
                        orig_idx_b = col_idx
                    candidate_pairs.append({
                        "idx_a": orig_idx_a,
                        "idx_b": orig_idx_b,
                        "tfidf_similarity": sim,
                    })
                    if len(candidate_pairs) >= max_pairs * 2:
                        break
            if len(candidate_pairs) >= max_pairs * 2:
                break
        if len(candidate_pairs) >= max_pairs * 2:
            break

    candidate_pairs.sort(key=lambda x: x["tfidf_similarity"], reverse=True)
    return candidate_pairs[:max_pairs]


_name_token_re = re.compile(r"[a-zA-Z]{2,}")
_paren_re = re.compile(r"\([^)]*\)")   # affiliations: "(Brown University)", "(for the CDF...)"
_GENERIC_NAME_TOKENS = frozenset("""
and et al the for on behalf of university univ institute inst college
department dept laboratory lab collaboration collaborations team group
center centre national school academy division faculty
""".split())

def _author_tokens(s):
    """Name tokens from an author string, robust to separators (| ; , 'and')
    and ordering. Parenthesized affiliations and institutional words are
    stripped so only actual person names are compared."""
    s = _paren_re.sub(" ", str(s))
    return {t.lower() for t in _name_token_re.findall(s)
            if t.lower() not in _GENERIC_NAME_TOKENS}

def _is_missing(v):
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    s = str(v).strip().lower()
    return s in {"", "nan", "none", "null", "unknown", "-", "—"}

def compute_field_similarities(rec_a, rec_b):
    """Returns dict of field -> similarity in [0,1], or None if the field is
    missing on either side (missing data is no evidence either way)."""
    sims = {}

    if not _is_missing(rec_a.get("title")) and not _is_missing(rec_b.get("title")):
        title_a = str(rec_a["title"]).lower().strip()
        title_b = str(rec_b["title"]).lower().strip()
        sims["title"] = 1 - (Levenshtein.distance(title_a, title_b) / max(len(title_a), len(title_b)))
    else:
        sims["title"] = None

    if not _is_missing(rec_a.get("authors")) and not _is_missing(rec_b.get("authors")):
        ta, tb = _author_tokens(rec_a["authors"]), _author_tokens(rec_b["authors"])
        union = len(ta | tb)
        sims["authors"] = len(ta & tb) / union if union > 0 else None
    else:
        sims["authors"] = None

    if not _is_missing(rec_a.get("venue")) and not _is_missing(rec_b.get("venue")):
        va = str(rec_a["venue"]).lower().strip()
        vb = str(rec_b["venue"]).lower().strip()
        sims["venue"] = 1.0 if va == vb else 0.0
    else:
        sims["venue"] = None

    if not _is_missing(rec_a.get("year")) and not _is_missing(rec_b.get("year")):
        try:
            sims["year"] = 1.0 if int(float(rec_a["year"])) == int(float(rec_b["year"])) else 0.0
        except (TypeError, ValueError):
            sims["year"] = None
    else:
        sims["year"] = None
    return sims

# Title is the strongest evidence; venue/year are weak corroboration
_FIELD_WEIGHTS = {"title": 0.55, "authors": 0.25, "venue": 0.10, "year": 0.10}

def _author_conflict(field_sims):
    """Identical titles with clearly different authors usually means two
    distinct works (separate proceedings, talks, or review papers on the
    same topic) — not a duplicate record."""
    return (field_sims.get("title") is not None
            and field_sims.get("authors") is not None
            and field_sims["title"] > 0.85
            and field_sims["authors"] < 0.2)

def _republication(field_sims):
    """Same title AND same authors, but different venue and year: usually
    the same work published twice (e.g. conference proceedings + journal
    version). Whether that counts as a duplicate is a policy decision."""
    return (field_sims.get("title") is not None and field_sims["title"] > 0.85
            and field_sims.get("authors") is not None and field_sims["authors"] > 0.7
            and field_sims.get("venue") == 0.0
            and field_sims.get("year") == 0.0)

def predict_duplicate(rec_a, rec_b):
    field_sims = compute_field_similarities(rec_a, rec_b)
    available = {k: v for k, v in field_sims.items() if v is not None}
    if not available:
        return False, 0.0, field_sims
    total_w = sum(_FIELD_WEIGHTS[k] for k in available)
    confidence = sum(_FIELD_WEIGHTS[k] * v for k, v in available.items()) / total_w
    if _author_conflict(field_sims):
        # Strong disagreement on real author names overrides a title match
        confidence *= 0.45
    return confidence > 0.7, confidence, field_sims

def save_feedback(feedback_log):
    output_path = Path("data/outputs/user_feedback.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_types(feedback_log), f, indent=2)


def paper_card(label, rec):
    def fmt(v):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else v
    authors = str(fmt(rec.get("authors"))).replace("|", ", ")
    year = rec.get("year")
    try:
        year = int(year) if pd.notna(year) else "—"
    except Exception:
        year = fmt(year)
    st.markdown(f"""
    <div class="paper-card">
        <div class="paper-label">{label}</div>
        <div class="paper-title">{fmt(rec.get('title'))}</div>
        <div class="paper-meta">
            <span class="k">Authors</span> &nbsp;{authors}<br>
            <span class="k">Venue</span> &nbsp;&nbsp;&nbsp;{fmt(rec.get('venue'))}<br>
            <span class="k">Year</span> &nbsp;&nbsp;&nbsp;&nbsp;{year}
        </div>
    </div>
    """, unsafe_allow_html=True)

def sim_bar(label, value):
    if value is None:
        st.markdown(f"""
        <div class="sim-row">
            <div class="sim-label">{label}</div>
            <div class="sim-track"></div>
            <div class="sim-val" style="color:#A8A29E">n/a</div>
        </div>
        """, unsafe_allow_html=True)
        return
    st.markdown(f"""
    <div class="sim-row">
        <div class="sim-label">{label}</div>
        <div class="sim-track"><div class="sim-fill" style="width:{value*100:.0f}%"></div></div>
        <div class="sim-val">{value:.0%}</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div class="app-title">Duplicate Validator</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Human-in-the-loop review for bibliographic data cleaning</div>',
            unsafe_allow_html=True)


if not st.session_state.data_loaded:
    tab1, tab2, tab3 = st.tabs(["Upload file", "Sample data", "Format guide"])

    with tab1:
        st.markdown("Upload a CSV or JSON file with your bibliographic records. "
                    "Columns can have any names — you'll map them after upload.")

        uploaded_file = st.file_uploader("Choose a file", type=["csv", "json"],
                                         label_visibility="collapsed")

        if uploaded_file is not None:
            import tempfile, os

            # Spill the upload to a temp file ONCE, then stream it back from
            # disk in chunks — working memory stays small even for huge files.
            fname = uploaded_file.name
            fid = getattr(uploaded_file, "file_id", None) or f"{fname}:{getattr(uploaded_file, 'size', '')}"
            if st.session_state.get("upload_tmp_id") != fid:
                old = st.session_state.get("upload_tmp_path")
                if old and os.path.exists(old):
                    try: os.remove(old)
                    except Exception: pass
                suffix = ".json" if fname.endswith(".json") else ".csv"
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tf.write(uploaded_file.getbuffer()); tf.close()
                st.session_state.upload_tmp_id = fid
                st.session_state.upload_tmp_path = tf.name
            tmp_path = st.session_state.upload_tmp_path

            # Detect JSON-lines (one object per line, e.g. the arXiv snapshot)
            is_jl = False
            if fname.endswith(".json"):
                try:
                    with open(tmp_path, encoding="utf-8") as fh:
                        head = fh.readline().strip()
                    json.loads(head); is_jl = head.startswith("{")
                except Exception:
                    is_jl = False

            # Small preview for column detection — never loads the whole file
            raw_df, errors = None, []
            try:
                if fname.endswith(".csv"):
                    raw_df = pd.read_csv(tmp_path, nrows=200, dtype=str)
                elif is_jl:
                    rows = []
                    with open(tmp_path, encoding="utf-8") as fh:
                        for line in fh:
                            if line.strip():
                                rows.append(json.loads(line))
                            if len(rows) >= 200:
                                break
                    raw_df = pd.DataFrame(rows)
                else:
                    raw_df = pd.read_json(tmp_path).head(200)
                if raw_df is None or len(raw_df.columns) == 0:
                    errors = ["No columns found in file."]
            except Exception as e:
                errors = [f"Could not read file: {e}"]

            if errors:
                for e in errors:
                    st.error(e)
            else:
                st.caption("Preview (first rows) · columns: " +
                           ", ".join(str(c) for c in raw_df.columns))
                st.dataframe(raw_df.head(10), use_container_width=True)

                st.markdown('<div class="section-head">Map your columns</div>',
                            unsafe_allow_html=True)
                st.caption("Only Title is required. Authors, Venue, and Year improve accuracy. "
                           "Best guesses are pre-filled — adjust if needed.")

                guess = auto_guess_mapping(list(raw_df.columns))
                cols = [str(c) for c in raw_df.columns]
                NONE = "— none —"
                fid = uploaded_file.name

                def _sel(label, canon, required):
                    options = cols if required else [NONE] + cols
                    default = guess.get(canon)
                    index = options.index(default) if default in options else 0
                    return st.selectbox(label, options, index=index, key=f"map_{canon}_{fid}")

                c1, c2 = st.columns(2)
                with c1:
                    m_title = _sel("Title (required)", "title", True)
                    m_authors = _sel("Authors", "authors", False)
                with c2:
                    m_venue = _sel("Venue / Journal", "venue", False)
                    m_year = _sel("Year", "year", False)

                mapping = {
                    "title":   m_title,
                    "authors": None if m_authors == NONE else m_authors,
                    "venue":   None if m_venue == NONE else m_venue,
                    "year":    None if m_year == NONE else m_year,
                }

                if st.button("Start validation", type="primary"):
                    try:
                        from scalable_processing import read_records_chunked
                        status = st.empty()
                        with st.spinner("Reading file in chunks (low memory)..."):
                            df = read_records_chunked(
                                tmp_path, mapping, is_json_lines=is_jl,
                                chunksize=50_000,
                                progress=lambda k: status.text(f"Loaded {k:,} records..."))
                        status.text(f"Loaded {len(df):,} records.")
                        with st.spinner("Finding candidate duplicates (token blocking)..."):
                            candidate_pairs = generate_candidate_pairs(df, max_pairs=500)
                        st.session_state.df = df
                        st.session_state.candidate_pairs = candidate_pairs
                        st.session_state.data_loaded = True
                        st.session_state.current_pair_idx = 0
                        st.session_state.feedback_log = []
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing file: {e}")

        # ---- Very large files: read straight from disk by path ----
        with st.expander("Very large file (multi-GB)? Load it from a local path"):
            st.caption(
                "Browser uploads hold the whole file in memory once, so for "
                "files of several GB it's faster and safer to read straight "
                "from disk. Paste the file's full path below. "
                "(Local use only — this can't reach your disk when the app "
                "runs on a hosting server.)")
            big_path = st.text_input(
                "Full path to file",
                placeholder=r"C:\Users\you\Downloads\arxiv-metadata-oai-snapshot.json",
                key="big_path_input")
            if st.button("Load from path", key="load_from_path"):
                import os
                big_path = (big_path or "").strip().strip('"')
                if not big_path:
                    st.warning("Paste a file path first.")
                elif not os.path.exists(big_path):
                    st.error(f"File not found: {big_path}")
                else:
                    try:
                        from scalable_processing import read_records_chunked
                        is_jl2 = False
                        if big_path.lower().endswith(".json"):
                            try:
                                with open(big_path, encoding="utf-8") as fh:
                                    head = fh.readline().strip()
                                json.loads(head); is_jl2 = head.startswith("{")
                            except Exception:
                                is_jl2 = False
                        if big_path.lower().endswith(".csv"):
                            preview = pd.read_csv(big_path, nrows=200, dtype=str)
                        elif is_jl2:
                            rows = []
                            with open(big_path, encoding="utf-8") as fh:
                                for line in fh:
                                    if line.strip():
                                        rows.append(json.loads(line))
                                    if len(rows) >= 200:
                                        break
                            preview = pd.DataFrame(rows)
                        else:
                            preview = pd.read_json(big_path).head(200)

                        mapping2 = auto_guess_mapping(list(preview.columns))
                        st.caption("Auto-detected columns: " +
                                   ", ".join(f"{k} → {v}" for k, v in mapping2.items() if v))
                        status = st.empty()
                        with st.spinner("Reading file in chunks (low memory)..."):
                            df = read_records_chunked(
                                big_path, mapping2, is_json_lines=is_jl2,
                                chunksize=50_000,
                                progress=lambda k: status.text(f"Loaded {k:,} records..."))
                        status.text(f"Loaded {len(df):,} records.")
                        with st.spinner("Finding candidate duplicates (token blocking)..."):
                            candidate_pairs = generate_candidate_pairs(df, max_pairs=500)
                        st.session_state.df = df
                        st.session_state.candidate_pairs = candidate_pairs
                        st.session_state.data_loaded = True
                        st.session_state.current_pair_idx = 0
                        st.session_state.feedback_log = []
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error reading file: {e}")

        # ---- Deployed / cloud: load a large file from a public URL ----
        with st.expander("Load from a URL (works on the deployed site)"):
            st.caption(
                "The hosted server can't read files off your computer, but it "
                "can download from a public link. Host your dataset somewhere "
                "public — Hugging Face, Zenodo, a GitHub release, an S3/HTTP "
                "link, or a Google Drive direct-download link — and paste the "
                "URL. The server streams it in chunks, so memory stays bounded. "
                "CSV and JSON-lines formats stream best.")
            url = st.text_input(
                "Public file URL",
                placeholder="https://huggingface.co/datasets/you/data/resolve/main/records.jsonl",
                key="url_input")
            if st.button("Load from URL", key="load_from_url"):
                import os, tempfile, urllib.request
                url = (url or "").strip().strip('"')
                if not url.lower().startswith(("http://", "https://")):
                    st.warning("Paste a full http(s) URL.")
                else:
                    try:
                        # Stream the download to a temp file (bounded memory)
                        lower = url.lower().split("?")[0]
                        suffix = (".csv" if lower.endswith(".csv")
                                  else ".jsonl" if lower.endswith((".jsonl", ".ndjson"))
                                  else ".json")
                        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                        status = st.empty()
                        req = urllib.request.Request(
                            url, headers={"User-Agent": "duplicate-validator"})
                        with urllib.request.urlopen(req) as resp:
                            total = resp.length or 0
                            got = 0
                            while True:
                                block = resp.read(1024 * 1024)  # 1 MB chunks
                                if not block:
                                    break
                                tf.write(block)
                                got += len(block)
                                if total:
                                    status.text(f"Downloading... {got/1e6:.0f} / {total/1e6:.0f} MB")
                                else:
                                    status.text(f"Downloading... {got/1e6:.0f} MB")
                        tf.close()
                        url_path = tf.name

                        # Detect JSON-lines
                        is_jl3 = False
                        if suffix in (".json", ".jsonl"):
                            try:
                                with open(url_path, encoding="utf-8") as fh:
                                    head = fh.readline().strip()
                                json.loads(head); is_jl3 = head.startswith("{")
                            except Exception:
                                is_jl3 = suffix == ".jsonl"

                        if suffix == ".csv":
                            preview = pd.read_csv(url_path, nrows=200, dtype=str)
                        elif is_jl3:
                            rows = []
                            with open(url_path, encoding="utf-8") as fh:
                                for line in fh:
                                    if line.strip():
                                        rows.append(json.loads(line))
                                    if len(rows) >= 200:
                                        break
                            preview = pd.DataFrame(rows)
                        else:
                            preview = pd.read_json(url_path).head(200)

                        mapping3 = auto_guess_mapping(list(preview.columns))
                        st.caption("Auto-detected columns: " +
                                   ", ".join(f"{k} → {v}" for k, v in mapping3.items() if v))
                        from scalable_processing import read_records_chunked
                        with st.spinner("Reading file in chunks (low memory)..."):
                            df = read_records_chunked(
                                url_path, mapping3, is_json_lines=is_jl3,
                                chunksize=50_000,
                                progress=lambda k: status.text(f"Loaded {k:,} records..."))
                        status.text(f"Loaded {len(df):,} records.")
                        try:
                            os.remove(url_path)
                        except Exception:
                            pass
                        with st.spinner("Finding candidate duplicates (token blocking)..."):
                            candidate_pairs = generate_candidate_pairs(df, max_pairs=500)
                        st.session_state.df = df
                        st.session_state.candidate_pairs = candidate_pairs
                        st.session_state.data_loaded = True
                        st.session_state.current_pair_idx = 0
                        st.session_state.feedback_log = []
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error downloading or reading file: {e}")

    with tab2:
        st.markdown("Load the built-in DBLP sample dataset (599 records) to try the tool.")
        if st.button("Load sample data"):
            try:
                df = pd.read_csv("data/processed/dblp_10k_clean.csv")
                with st.spinner("Finding candidate duplicates..."):
                    candidate_pairs = generate_candidate_pairs(df)
                st.session_state.df = df
                st.session_state.candidate_pairs = candidate_pairs
                st.session_state.data_loaded = True
                st.session_state.current_pair_idx = 0
                st.session_state.feedback_log = []
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample data: {e}")

    with tab3:
        st.markdown("""
**Required column**

- `title` — paper or article title

**Optional columns**

- `authors` — names separated by `|` (e.g. `John Smith|Jane Doe`)
- `venue` — conference or journal name
- `year` — publication year

**Supported formats:** CSV and JSON. Minimum 2 records; the tool works best with 100–10,000.

**Example CSV**
```csv
title,authors,venue,year
"Deep Learning for NLP","John Smith|Jane Doe","ACL",2023
"Machine Learning Basics","Alice Wong","ICML",2022
```
        """)
        template_csv = (
            'title,authors,venue,year\n'
            '"Example Paper 1","Author A|Author B","Conference Name",2023\n'
            '"Example Paper 2","Author C","Journal Name",2023'
        )
        st.download_button("Download CSV template", data=template_csv,
                           file_name="template_bibliographic_data.csv", mime="text/csv")

    st.stop()


df = st.session_state.df
candidate_pairs = st.session_state.candidate_pairs

with st.sidebar:
    st.markdown("### Progress")
    total_pairs = len(candidate_pairs)
    validated = len(st.session_state.feedback_log)
    st.metric("Pairs reviewed", f"{validated} / {total_pairs}")
    st.progress(validated / total_pairs if total_pairs > 0 else 0)

    if validated > 0:
        accepted = sum(1 for f in st.session_state.feedback_log if f["user_decision"] == "accept")
        c1, c2 = st.columns(2)
        c1.metric("Accepted", accepted)
        c2.metric("Rejected", validated - accepted)

    st.divider()
    st.markdown("### How it works")
    st.caption("Review each pair, check the similarity breakdown, then accept "
               "if they're duplicates or reject if they're not. Your decisions "
               "are logged and can be downloaded at the end.")
    st.divider()
    if st.button("Load a different dataset", use_container_width=True):
        st.session_state.data_loaded = False
        st.session_state.df = None
        st.session_state.candidate_pairs = None
        st.session_state.current_pair_idx = 0
        st.session_state.feedback_log = []
        st.rerun()

if st.session_state.current_pair_idx >= len(candidate_pairs):
    st.success("All pairs reviewed.")
    st.markdown('<div class="section-head">Summary</div>', unsafe_allow_html=True)
    feedback_df = pd.DataFrame(st.session_state.feedback_log)

    if len(feedback_df) > 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total reviewed", len(feedback_df))
        col2.metric("Accepted as duplicates", int(sum(feedback_df["user_decision"] == "accept")))
        col3.metric("Rejected", int(sum(feedback_df["user_decision"] == "reject")))
        st.dataframe(feedback_df, use_container_width=True)

        st.download_button(
            "Download feedback log (JSON)",
            data=json.dumps(st.session_state.feedback_log, indent=2, default=str),
            file_name="user_feedback.json",
            mime="application/json",
        )
    else:
        st.caption("No decisions were recorded.")

else:
    pair = candidate_pairs[st.session_state.current_pair_idx]
    rec_a = df.iloc[pair["idx_a"]].to_dict()
    rec_b = df.iloc[pair["idx_b"]].to_dict()

    is_dup, confidence, field_sims = predict_duplicate(rec_a, rec_b)

    st.markdown(f'<div class="section-head">Pair {st.session_state.current_pair_idx + 1} '
                f'of {len(candidate_pairs)}</div>', unsafe_allow_html=True)

    uncertain = _republication(field_sims) or (0.55 < confidence <= 0.82)
    if uncertain:
        st.markdown(f'<div class="verdict unsure">Prediction: <b>uncertain — needs your '
                    f'judgment</b> · duplicate score {confidence:.0%}</div>',
                    unsafe_allow_html=True)
    elif is_dup:
        st.markdown(f'<div class="verdict dup">Prediction: <b>likely duplicate</b> '
                    f'· confidence {confidence:.0%}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict nodup">Prediction: <b>likely not a duplicate</b> '
                    f'· confidence {1-confidence:.0%}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        paper_card("Paper A", rec_a)
    with col2:
        paper_card("Paper B", rec_b)

    st.markdown('<div class="section-head">Field similarity</div>', unsafe_allow_html=True)
    sim_bar("Title", field_sims["title"])
    sim_bar("Authors", field_sims["authors"])
    sim_bar("Venue", field_sims["venue"])
    sim_bar("Year", field_sims["year"])

    # Explanation
    fs = field_sims
    missing = [k for k, v in fs.items() if v is None]
    repub_note = ""
    if _republication(fs):
        repub_note = ("Same title and same authors, but different venue and year — "
                      "this looks like the same work published twice (e.g. a journal "
                      "version and a conference-proceedings version). Whether that "
                      "counts as a duplicate depends on your deduplication policy. ")
    if is_dup:
        reasons = []
        if fs["title"] is not None and fs["title"] > 0.9:
            reasons.append(f"nearly identical titles ({fs['title']:.0%} match)")
        if fs["authors"] is not None and fs["authors"] > 0.7:
            reasons.append(f"overlapping authors ({fs['authors']:.0%})")
        if fs["venue"] == 1.0:
            reasons.append("same venue")
        if fs["year"] == 1.0:
            reasons.append("same year")
        explanation = (f"Likely duplicates: {', '.join(reasons)}."
                       if reasons else "High overall semantic similarity.")
    else:
        differences = []
        if _author_conflict(fs):
            differences.append(
                "same title but clearly different authors — likely two distinct "
                "works on the same topic (e.g. separate talks or proceedings)")
        if fs["title"] is not None and fs["title"] < 0.5:
            differences.append(f"different titles ({fs['title']:.0%} match)")
        if fs["authors"] is not None and fs["authors"] < 0.3:
            differences.append(f"different authors ({fs['authors']:.0%} overlap)")
        if fs["venue"] is not None and fs["venue"] < 0.5:
            differences.append("different venues")
        if fs["year"] == 0.0:
            differences.append("different years")
        explanation = (f"Likely not duplicates: {', '.join(differences)}."
                       if differences else "Low overall semantic similarity.")
    if missing:
        explanation += (f" ({', '.join(missing).capitalize()} missing on one side — "
                        "excluded from the score.)")
    explanation = repub_note + explanation

    st.markdown('<div class="section-head">Explanation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="explain">{explanation}</div>', unsafe_allow_html=True)

    st.write("")
    bcol1, bcol2, bcol3 = st.columns([2, 2, 1])

    def _log(decision):
        st.session_state.feedback_log.append({
            "pair_index": st.session_state.current_pair_idx,
            "idx_a": pair["idx_a"],
            "idx_b": pair["idx_b"],
            "ai_prediction": "duplicate" if is_dup else "not_duplicate",
            "ai_confidence": float(confidence),
            "user_decision": decision,
            "field_similarities": {k: (None if v is None else float(v)) for k, v in field_sims.items()},
            "timestamp": datetime.now().isoformat(),
        })
        try:
            save_feedback(st.session_state.feedback_log)
        except Exception:
            pass  # read-only filesystem on some hosts; log stays in session
        st.session_state.current_pair_idx += 1
        st.rerun()

    with bcol1:
        if st.button("Accept — these are duplicates", use_container_width=True, type="primary"):
            _log("accept")
    with bcol2:
        if st.button("Reject — not duplicates", use_container_width=True):
            _log("reject")
    with bcol3:
        if st.button("Skip", use_container_width=True):
            st.session_state.current_pair_idx += 1
            st.rerun()

st.markdown("")
st.caption("Human-in-the-loop duplicate detection · TF-IDF + Levenshtein similarity")
