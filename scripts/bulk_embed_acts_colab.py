# ============================================================
# Bulk Embed & Ingest — Kenya Acts of Parliament
# Run this in Google Colab (Pro recommended for long sessions)
#
# RESUMABLE: safe to stop/restart at any time.
#   - Checkpoint saved to Drive every CHECKPOINT_EVERY acts.
#   - Errors saved to Drive (not lost on crash).
#   - Ctrl+C / runtime interrupt triggers a clean flush first.
#   - Duplicate inserts are silently ignored on re-run.
# ============================================================

# ── CELL 1: Install dependencies ────────────────────────────
# !pip install -q pymongo langchain-google-genai langchain-text-splitters tenacity tqdm pandas google-api-python-client

# ── CELL 2: Authenticate & mount Drive ──────────────────────
# from google.colab import auth, drive
# auth.authenticate_user()
# drive.mount('/content/drive')

# ── CELL 3: Configuration — only section you need to edit ───

GOOGLE_API_KEY  = "YOUR_GOOGLE_API_KEY"
MONGO_URI       = "YOUR_ATLAS_MONGO_URI"   # include db name: .../haki?retryWrites=true
MONGO_DB        = "YOUR_DB_NAME"
COLLECTION_NAME = "legal_docs"             # same collection as cases

# Paths on mounted Drive
CSV_PATH  = "/content/drive/MyDrive/Haki/scraping_progress.csv"
ACTS_DIR  = "/content/drive/MyDrive/Haki/Acts"   # fallback if path in CSV doesn't resolve

# State files — saved to Drive so they survive runtime restarts
CHECKPOINT_FILE = "/content/drive/MyDrive/Haki/acts_checkpoint.json"
ERRORS_FILE     = "/content/drive/MyDrive/Haki/acts_errors.json"

# Chunking — section-aware, slightly larger for structured legislative text
CHUNK_SIZE    = 2000
CHUNK_OVERLAP = 150

# Flush + checkpoint every N acts
CHECKPOINT_EVERY = 50

# Gemini embedding API max texts per call
EMBED_BATCH = 100

# MongoDB insert_many batch size
MONGO_BATCH = 500

# ── CELL 4: Imports ──────────────────────────────────────────
import os, json
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

from pymongo import MongoClient
from pymongo.errors import BulkWriteError

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tenacity import (
    retry, stop_after_attempt,
    wait_exponential, retry_if_exception_type,
)
import google.api_core.exceptions

print("Imports OK")

# ── CELL 5: Clients ──────────────────────────────────────────

embedder = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=GOOGLE_API_KEY,
    task_type="retrieval_document",
    output_dimensionality=1536,
)

mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
mongo_client.admin.command("ping")
collection = mongo_client[MONGO_DB][COLLECTION_NAME]

# Section-aware splitter for legislative text.
# Tries to keep whole sections together before falling back to paragraphs.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    is_separator_regex=True,
    separators=[
        r"\n#{1,3} ",       # Markdown headers: # ## ###
        r"\n\d+\. ",        # Numbered sections: 1. 2. 3.
        r"\n\(\d+\) ",      # Sub-sections: (1) (2)
        r"\n\([a-z]\) ",    # Sub-subsections: (a) (b)
        r"\n[A-Z][A-Z ]+\n",# ALL-CAPS headings (PART I, SCHEDULE, etc.)
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)

print(f"✓ MongoDB connected → {MONGO_DB}.{COLLECTION_NAME}")

# ── CELL 6: Load CSV ─────────────────────────────────────────
print("\nLoading CSV…")
df = pd.read_csv(CSV_PATH, dtype=str).fillna("")

# Only process completed rows with no errors
df = df[(df["status"] == "completed") & (df["error_message"] == "")]
print(f"  {len(df):,} completed acts")

def parse_details(json_str: str) -> dict:
    """Parse document_details_json safely."""
    try:
        return json.loads(json_str) if json_str else {}
    except Exception:
        return {}

def resolve_path(csv_path: str) -> Path:
    """
    Convert the path stored in the CSV (which uses /content/drive/My Drive/)
    to the mounted Drive path (which uses /content/drive/MyDrive/).
    Falls back to ACTS_DIR + filename if the path doesn't exist.
    """
    p = Path(csv_path.replace("/content/drive/My Drive/", "/content/drive/MyDrive/"))
    if p.exists():
        return p
    # Fallback: look in ACTS_DIR by filename
    fallback = Path(ACTS_DIR) / p.name
    return fallback

# Build record list
records = []
for _, row in df.iterrows():
    act_id   = row.get("uuid", "")
    filepath = resolve_path(row.get("markdown_text_path", ""))
    details  = parse_details(row.get("document_details_json", ""))
    records.append({
        "act_id":       act_id,
        "filepath":     filepath,
        "title":        row.get("title", ""),
        "url":          row.get("url", ""),
        "act_type":     row.get("type", ""),            # Main Act / Subsidiary Legislation
        "main_act_title": row.get("main_act_title", ""),
        "main_act_url": row.get("main_act_url", ""),
        "citation":     details.get("citation", "").replace("Copy", "").strip(),
        "current_date": (details.get("date") or {}).get("current_date", ""),
        "language":     details.get("language", "English"),
        "sub_type":     details.get("type", ""),        # e.g. "Legal Notice"
    })

print(f"  {len(records):,} records ready")

# ── CELL 7: Checkpoint & error helpers ───────────────────────

def load_checkpoint() -> set[str]:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f))
    return set()

def save_checkpoint(done: set[str]):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(sorted(done), f)

def load_errors() -> list:
    if os.path.exists(ERRORS_FILE):
        with open(ERRORS_FILE) as f:
            return json.load(f)
    return []

def save_errors(errors: list):
    with open(ERRORS_FILE, "w") as f:
        json.dump(errors, f, indent=2)

# ── CELL 8: Document builder ─────────────────────────────────

def make_docs(record: dict, text: str) -> list[dict]:
    chunks = splitter.split_text(text)
    act_id = record["act_id"]
    return [
        {
            "_id":            f"{act_id}_{i}",
            "text":           chunk,
            "file_id":        act_id,
            "chunk_index":    i,
            "doc_type":       "act",
            # Act identity
            "title":          record["title"],
            "citation":       record["citation"],
            "act_type":       record["act_type"],       # Main Act / Subsidiary Legislation
            "sub_type":       record["sub_type"],       # Legal Notice, etc.
            # For subsidiary legislation — link to parent act
            "main_act_title": record["main_act_title"],
            "main_act_url":   record["main_act_url"],
            # Dates & links
            "current_date":   record["current_date"],
            "url":            record["url"],
            "language":       record["language"],
        }
        for i, chunk in enumerate(chunks)
    ]

# ── CELL 9: Embedding with retry ─────────────────────────────

@retry(
    retry=retry_if_exception_type((
        google.api_core.exceptions.ResourceExhausted,
        google.api_core.exceptions.ServiceUnavailable,
        google.api_core.exceptions.DeadlineExceeded,
    )),
    wait=wait_exponential(multiplier=2, min=5, max=120),
    stop=stop_after_attempt(8),
)
def _embed_sub_batch(texts: list[str]) -> list[list[float]]:
    return embedder.embed_documents(texts)

def embed_all(texts: list[str]) -> list[list[float]]:
    vecs = []
    for i in range(0, len(texts), EMBED_BATCH):
        vecs.extend(_embed_sub_batch(texts[i : i + EMBED_BATCH]))
    return vecs

# ── CELL 10: MongoDB insert (duplicate-safe) ──────────────────

def mongo_insert(docs: list[dict]):
    if not docs:
        return
    try:
        collection.insert_many(docs, ordered=False)
    except BulkWriteError as bwe:
        non_dup = [
            e for e in bwe.details.get("writeErrors", [])
            if e.get("code") != 11000
        ]
        if non_dup:
            raise RuntimeError(f"Unexpected write errors: {non_dup[:3]}") from bwe

# ── CELL 11: Flush helper ─────────────────────────────────────

buf_docs:    list[dict] = []
buf_texts:   list[str]  = []
pending_ids: set[str]   = set()

done_ids = load_checkpoint()
errors   = load_errors()

def flush():
    global buf_docs, buf_texts, pending_ids
    if not buf_texts:
        return
    vecs = embed_all(buf_texts)
    for doc, vec in zip(buf_docs, vecs):
        doc["embedding"] = vec
    for i in range(0, len(buf_docs), MONGO_BATCH):
        mongo_insert(buf_docs[i : i + MONGO_BATCH])
    done_ids.update(pending_ids)
    save_checkpoint(done_ids)
    buf_docs, buf_texts, pending_ids = [], [], set()

# ── CELL 12: Main ingestion loop ──────────────────────────────

remaining = [r for r in records if r["act_id"] not in done_ids]

print(f"\nAlready done : {len(done_ids):,}")
print(f"To process   : {len(remaining):,}")
print(f"Checkpoint   : every {CHECKPOINT_EVERY} acts\n")

# Quick sanity check on paths
missing = [r for r in remaining[:10] if not r["filepath"].exists()]
if missing:
    print(f"⚠ Warning: {len(missing)} of first 10 files not found. Check ACTS_DIR and CSV paths.")
    for r in missing[:3]:
        print(f"  {r['filepath']}")

pbar = tqdm(total=len(remaining), unit="act")
cases_since_flush = 0

try:
    for record in remaining:
        act_id   = record["act_id"]
        filepath = record["filepath"]
        try:
            if not filepath.exists():
                errors.append({"id": act_id, "title": record["title"], "error": f"file not found: {filepath}"})
                pbar.update(1)
                continue

            text = filepath.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                errors.append({"id": act_id, "title": record["title"], "error": "empty file"})
                pbar.update(1)
                continue

            docs = make_docs(record, text)
            for d in docs:
                buf_texts.append(d["text"])
                buf_docs.append(d)
            pending_ids.add(act_id)
            cases_since_flush += 1

            if len(buf_texts) >= MONGO_BATCH or cases_since_flush >= CHECKPOINT_EVERY:
                flush()
                save_errors(errors)
                cases_since_flush = 0

        except Exception as e:
            errors.append({"id": act_id, "title": record["title"], "error": str(e)})
            save_errors(errors)

        pbar.update(1)

except KeyboardInterrupt:
    print("\n⚠ Interrupted — saving progress…")

finally:
    try:
        flush()
        save_errors(errors)
    except Exception as e:
        print(f"  Final flush failed: {e}")

pbar.close()

print(f"\n{'='*50}")
print(f"  Done    : {len(done_ids):,} acts")
print(f"  Errors  : {len(errors)}")
if errors:
    print(f"  See     : {ERRORS_FILE}")
    for e in errors[:5]:
        print(f"    [{e.get('title','')}] {e['error']}")
