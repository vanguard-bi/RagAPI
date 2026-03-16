# ============================================================
# Bulk Embed & Ingest — Kenya Law Judgments
# Run this in Google Colab (Pro recommended for long sessions)
#
# RESUMABLE: safe to stop/restart at any time.
#   - Checkpoint saved to Drive every CHECKPOINT_EVERY cases.
#   - Errors saved to Drive (not lost on crash).
#   - Ctrl+C / runtime interrupt triggers a clean flush first.
#   - Duplicate inserts are silently ignored on re-run.
#
# Drive IDs are used directly — no need to know folder paths.
# ============================================================

# ── CELL 1: Install dependencies ────────────────────────────
# !pip install -q pymongo langchain-google-genai langchain-text-splitters tenacity tqdm pandas gdown google-api-python-client

# ── CELL 2: Authenticate & mount Drive ──────────────────────
# from google.colab import auth, drive
# auth.authenticate_user()      # needed for Drive API (CSV download)
# drive.mount('/content/drive') # needed for local file access

# ── CELL 3: Configuration — only section you need to edit ───

GOOGLE_API_KEY  = "YOUR_GOOGLE_API_KEY"
MONGO_URI       = "YOUR_ATLAS_MONGO_URI"   # include db name: .../haki?retryWrites=true
MONGO_DB        = "YOUR_DB_NAME"           # e.g. "haki"
COLLECTION_NAME = "legal_docs"

# Google Drive CSV file ID (from share link)
CSV_FILE_ID = "1gSGMJYqvIlbDm61yFdhBeRiWFOVAa-A5"

# Path to markdown files via mounted Drive
FILES_DIR = "/content/drive/MyDrive/Haki Cases/files_batched"

# State files — saved to Drive so they survive runtime restarts
CHECKPOINT_FILE = "/content/drive/MyDrive/Haki Cases/embed_checkpoint.json"
ERRORS_FILE     = "/content/drive/MyDrive/Haki Cases/embed_errors.json"

# Chunking
CHUNK_SIZE    = 1500   # characters
CHUNK_OVERLAP = 150

# Flush + checkpoint every N cases (safety net for slow batches)
CHECKPOINT_EVERY = 50

# Gemini embedding API: max texts per call
EMBED_BATCH = 100

# MongoDB insert_many batch size
MONGO_BATCH = 500

# ── CELL 4: Imports ──────────────────────────────────────────
import os, io, json
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
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

print("Imports OK")

# ── CELL 5: Download CSV from Drive ──────────────────────────

drive_service = build("drive", "v3")

def drive_download_bytes(file_id: str) -> bytes:
    req = drive_service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl  = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()

print("Downloading CSV…")
csv_bytes  = drive_download_bytes(CSV_FILE_ID)
df         = pd.read_csv(io.BytesIO(csv_bytes), dtype=str).fillna("")
meta_index = df.set_index("id").to_dict("index")
print(f"  {len(meta_index):,} cases loaded")

# ── CELL 6: Scan local markdown files (via mounted Drive) ────
print(f"\nScanning {FILES_DIR} …")
md_files = list(Path(FILES_DIR).rglob("*.md"))
print(f"  Found {len(md_files):,} .md files")

def extract_id(filepath: Path) -> str | None:
    """Return the UUID after '___' in the filename stem."""
    stem = filepath.stem
    return stem.split("___")[-1] if "___" in stem else None

# Map case_id → local Path
file_index: dict[str, Path] = {}
unmatched = 0
for f in md_files:
    cid = extract_id(f)
    if cid and cid in meta_index:
        file_index[cid] = f
    else:
        unmatched += 1

print(f"  Matched      : {len(file_index):,}")
print(f"  No CSV match : {unmatched}")

# ── CELL 8: Clients ──────────────────────────────────────────

embedder = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=GOOGLE_API_KEY,
    task_type="retrieval_document",
    output_dimensionality=1536,
)

mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
mongo_client.admin.command("ping")
collection = mongo_client[MONGO_DB][COLLECTION_NAME]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

print(f"\n✓ MongoDB connected → {MONGO_DB}.{COLLECTION_NAME}")

# ── CELL 9: Checkpoint & error helpers ───────────────────────

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

# ── CELL 10: Document builder ────────────────────────────────

def make_docs(case_id: str, text: str, meta: dict) -> list[dict]:
    chunks = splitter.split_text(text)
    return [
        {
            "_id":           f"{case_id}_{i}",
            "text":          chunk,
            "file_id":       case_id,
            "chunk_index":   i,
            "title":         meta.get("title", ""),
            "court":         meta.get("court", ""),
            "court_code":    meta.get("param_court", ""),
            "year":          meta.get("param_year", ""),
            "judgment_date": meta.get("judgment_date", ""),
            "case_number":   meta.get("case_number", ""),
            "citation":      meta.get("citation", ""),
            "judges":        meta.get("judges", ""),
            "outcome":       meta.get("outcome", ""),
            "url":           meta.get("url", ""),
            "type":          meta.get("type", "Judgment"),
            "language":      meta.get("language", "English"),
        }
        for i, chunk in enumerate(chunks)
    ]

# ── CELL 11: Embedding with retry ────────────────────────────

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

# ── CELL 12: MongoDB insert (duplicate-safe) ─────────────────

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

# ── CELL 13: Flush helper ────────────────────────────────────

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

# ── CELL 14: Main ingestion loop ─────────────────────────────

remaining = [
    (cid, fid)
    for cid, fid in file_index.items()
    if cid not in done_ids
]

print(f"Already done : {len(done_ids):,}")
print(f"To process   : {len(remaining):,}")
print(f"Checkpoint   : every {CHECKPOINT_EVERY} cases\n")

pbar = tqdm(total=len(remaining), unit="case")
cases_since_flush = 0

try:
    for case_id, filepath in remaining:
        try:
            text = filepath.read_text(encoding="utf-8", errors="ignore").strip()

            if not text:
                errors.append({"id": case_id, "error": "empty file"})
                pbar.update(1)
                continue

            docs = make_docs(case_id, text, meta_index[case_id])
            for d in docs:
                buf_texts.append(d["text"])
                buf_docs.append(d)
            pending_ids.add(case_id)
            cases_since_flush += 1

            if len(buf_texts) >= MONGO_BATCH or cases_since_flush >= CHECKPOINT_EVERY:
                flush()
                save_errors(errors)
                cases_since_flush = 0

        except Exception as e:
            errors.append({"id": case_id, "error": str(e)})
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
print(f"  Done    : {len(done_ids):,} cases")
print(f"  Errors  : {len(errors)}")
if errors:
    print(f"  See     : {ERRORS_FILE}")
    for e in errors[:5]:
        print(f"    {e['id']}: {e['error']}")
