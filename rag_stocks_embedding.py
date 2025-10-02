import io
import os
import json
import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
import unicodedata
import html
import streamlit as st
import pdfplumber
import numpy as np
import requests

import chromadb
from chromadb.config import Settings

# NLP
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Preprocessing utilities
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
STOP = set(stopwords.words('english'))

# Logging setup (JSON lines)
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("rag_app")
logger.setLevel(logging.INFO)
log_handler = logging.FileHandler(os.path.join(LOG_DIR, "rag_events.log"))
log_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(log_handler)

# Chroma persistent client
CHROMA_DIR = "./chroma_store_stocks"
client = chromadb.PersistentClient(path=CHROMA_DIR)
CHROMA_COLLECTION = "bid_collection"
# Settings
#EMBED_MODEL = "ollama/mxbai-embed-large"  # replace with your ollama embed model name
EMBED_MODEL = "mxbai-embed-large"  # replace with your ollama embed model name
LLM_MODEL = "llama3.1:8b"         # replace with your ollama llm model name


# Initialize collection
#collection = client.get_or_create_collection(name=CHROMA_COLLECTION, metadata={})
collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
# Helpers: Ollama embed & chat calls (adjust if you use ollama SDK)
OLLAMA_BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def ollama_embed(
    texts: List[str],
    model: str = "mxbai-embed-large",
    timeout: int = 120,
    max_retries: int = 3,
    backoff: float = 0.5,
    debug_raw: bool = False,
) -> np.ndarray:
    """
    Get embeddings from Ollama, robust to several response shapes.

    Supported response shapes (handled):
      - {"embedding": [...]}  # single item
      - {"data":[ {"embedding":[...]} ]}  # batch style
      - {"result": {"embedding":[...]}}  # some wrappers
      - {"embeddings": [..., ...]}  # list-of-embeddings

    Raises a clear error with raw response text if the shape is unknown.
    Returns numpy.ndarray (float32) with shape (len(texts), dim).
    """
    print("ollama_embed method called")
    if not isinstance(texts, list):
        raise ValueError("texts must be a list of strings")

    url = f"{OLLAMA_BASE.rstrip('/')}/api/embeddings"

    embeddings: List[List[float]] = []

    for t in texts:
        payload = {"model": model, "prompt": t}
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                resp = requests.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()

                if debug_raw:
                    print("OLLAMA EMBED RAW RESPONSE:", data)

                # Try common shapes in order of likelihood
                # 1) top-level "embedding"
                if isinstance(data, dict) and "embedding" in data and isinstance(data["embedding"], list):
                    emb = data["embedding"]

                # 2) {"data": [ {"embedding": [...]}, ... ] }  (batch shape)
                elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                    # take the first element's embedding if present
                    first = data["data"][0] if data["data"] else None
                    if isinstance(first, dict) and "embedding" in first:
                        emb = first["embedding"]
                    else:
                        # maybe the array is directly embeddings
                        if all(isinstance(x, list) for x in data["data"]):
                            # pick first list
                            emb = data["data"][0]
                        else:
                            raise ValueError("No 'embedding' found in 'data' entry")

                # 3) {"result": {"embedding": [...]}}
                elif isinstance(data, dict) and "result" in data and isinstance(data["result"], dict) and "embedding" in data["result"]:
                    emb = data["result"]["embedding"]

                # 4) {"embeddings": [[...],[...]]}  (list of embeddings)
                elif isinstance(data, dict) and "embeddings" in data and isinstance(data["embeddings"], list):
                    # choose first embedding
                    emb = data["embeddings"][0]

                # 5) If the response is already a plain list (unlikely for POST per-text)
                elif isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                    emb = data

                else:
                    # Unknown format — help the user by raising with details
                    raise ValueError(f"Unexpected response format from embedding API: {type(data)} / keys: {list(data.keys()) if isinstance(data, dict) else 'non-dict'}; full: {data}")

                # validate embedding shape and type
                if not isinstance(emb, (list, tuple)) or not all(isinstance(x, (float, int)) for x in emb):
                    raise ValueError("Embedding found but has unexpected contents or type")

                embeddings.append([float(x) for x in emb])
                last_exc = None
                break

            except Exception as e:
                last_exc = e
                # retry only on network-related or 5xx errors; stop on 4xx client errors
                status = None
                try:
                    status = resp.status_code  # type: ignore
                except Exception:
                    status = None
                # If it's a client error (400-499) don't retry
                if status and 400 <= status < 500:
                    raise
                # otherwise wait and retry
                time.sleep(backoff * (attempt + 1))

        if last_exc is not None:
            # raise a helpful error showing last response text if available
            raw_text = None
            try:
                raw_text = resp.text  # type: ignore
            except Exception:
                raw_text = "<no response text available>"
            raise RuntimeError(f"Failed to get embedding for a text after {max_retries} attempts. last error: {last_exc}. raw response: {raw_text}")

    # final array
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.array(embeddings, dtype=np.float32)
    return arr


def ollama_embed2(texts: List[str], model: str = "mxbai-embed-large", retry: int = 3) -> np.ndarray:
    print("ollama_embed method called")
    url = f"{OLLAMA_BASE}/api/embeddings"
    vecs: List[List[float]] = []
    for t in texts:
        payload = {"model": model, "prompt": t}
        for attempt in range(retry):
            try:
                r = requests.post(url, json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
                v = data.get("embedding")
                if not v:
                    raise ValueError("Empty embedding from Ollama")
                vecs.append(v)
                break
            except Exception:
                if attempt == retry - 1:
                    raise
                time.sleep(0.4 * (attempt + 1))
    return np.array(vecs, dtype=np.float32)


def ollama_embed3(texts: List[str], model: str = "mxbai-embed-large", batch_size: int = 32) -> np.ndarray:
    print("ollama_embed2 method called")
    """Call local Ollama embedding endpoint. Returns numpy array shape (len(texts), dim)."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(batch)
        resp = requests.post(f"{OLLAMA_BASE}/api/embeddings", json={"model": model, "texts": batch})
        resp.raise_for_status()
        data = resp.json()

        # Expecting data to be {'embeddings': [[...], [...]]}
        batch_emb = data.get('embedding') or data.get('embeddings')
        if isinstance(batch_emb, list):
            embeddings.extend(batch_emb)
        else:
            raise ValueError("Unexpected response format from embedding API.")

    return np.array(embeddings)

def parse_ollama_ndjson(ndjson_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse NDJSON streaming response from Ollama and return:
      - full_text: concatenated assistant content
      - info: metadata from the last parsed object (done flag, done_reason, timings, etc.)
    Handles partial content fields across lines and joins them in order.
    """
    full_parts = []
    last_obj: Dict[str, Any] = {}
    if not ndjson_text:
        return "", {"raw": ""}

    # splitlines() preserves order, ignore empty lines
    for line in ndjson_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # ignore non-json lines (unlikely) or break
            continue
        # extract assistant content if present
        # common shape: {"message": {"role":"assistant", "content":"..."}}
        # some variants may present nested structures; be defensive
        msg = None
        if isinstance(obj, dict):
            if "message" in obj and isinstance(obj["message"], dict):
                msg = obj["message"]
            # older shapes: {"response": "..."} or {"text": "..."}
            elif "response" in obj and isinstance(obj["response"], str):
                full_parts.append(obj["response"])
            elif "text" in obj and isinstance(obj["text"], str):
                full_parts.append(obj["text"])
        if msg:
            content = msg.get("content")
            if isinstance(content, str):
                full_parts.append(content)
        last_obj = obj

    # join parts without adding extra spaces (they already include spacing in most streams)
    full_text = "".join(full_parts)
    return full_text, last_obj

def ollama_chat(
    prompt: str,
    model: str = LLM_MODEL,
    system: Optional[str] = None,
    max_tokens: int = 512,
    stream_timeout: int = 300,
    debug: bool = False,
) -> str:
    """
    Call Ollama /api/chat and gracefully handle streamed NDJSON or plain JSON responses.
    Returns concatenated assistant message (string). Raises on HTTP errors.
    """
    payload = {"model": model, "messages": []}
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].append({"role": "user", "content": prompt})
    payload["options"] = {"num_predict": max_tokens}

    # request - don't set stream=True here; Ollama returns NDJSON even if not using chunked transfer.
    resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=stream_timeout)
    if debug:
        print("HTTP status:", resp.status_code)
        print("Raw response head (first 1000 chars):")
        print(resp.text[:1000])

    resp.raise_for_status()

    text = resp.text
    # Fast path: single JSON object
    try:
        parsed = resp.json()
        # common single-object responses
        #  - {"message": {"role":"assistant","content":"..."}}
        if isinstance(parsed, dict):
            if "message" in parsed and isinstance(parsed["message"], dict):
                content = parsed["message"].get("content", "")
                if isinstance(content, str):
                    return content
            if "response" in parsed and isinstance(parsed["response"], str):
                return parsed["response"]
        # if we fall through, maybe it's a list or other shape; try NDJSON below
    except Exception:
        # json() failed -> likely NDJSON or multiple JSON objects; parse lines
        pass

    # NDJSON handling: parse each JSON line and concatenate
    full_text, last_obj = parse_ollama_ndjson(text)
    if debug:
        print("Parsed NDJSON last obj keys:", list(last_obj.keys()))
        print("Done flag in last obj:", last_obj.get("done"), "done_reason:", last_obj.get("done_reason"))

    return full_text

def ollama_chat3(prompt: str, model: str = LLM_MODEL, system: Optional[str] = None, max_tokens: int = 512) -> str:
    print("ollama_chat method called")

    payload = {
        "model": model,
        "messages": []
    }
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].append({"role": "user", "content": prompt})
    payload["options"] = {"num_predict": max_tokens}

    resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=300)
    print("RAW RESPONSE:", resp.text)
    resp.raise_for_status()
    data = resp.json()

    # /api/chat returns something like: {"message": {"role":"assistant","content":"..."}}
    if "message" in data and isinstance(data["message"], dict):
        return data["message"].get("content", "").strip()

    # fallback: some builds wrap messages in a list
    if "messages" in data and isinstance(data["messages"], list):
        for m in data["messages"]:
            if m.get("role") == "assistant":
                return m.get("content", "").strip()

    # fallback: old generate-like response
    if "response" in data:
        return data["response"].strip()

    return json.dumps(data)



def ollama_chat2(prompt: str, model: str = LLM_MODEL, system: Optional[str] = None, max_tokens: int = 512) -> str:
    print("ollama_chat method called")
    payload = {
        "model": model,
        "messages": []
    }
    if system:
        payload['messages'].append({"role": "system", "content": system})
    payload['messages'].append({"role": "user", "content": prompt})

    resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload)
    print(resp.text)
    resp.raise_for_status()
    data = resp.json()
    # adapt based on the exact shape of ollama response
    # try to read message -> content
    choices = data.get('choices') or data.get('generations') or data
    if isinstance(choices, list) and len(choices) > 0:
        # try several common keys
        text = None
        c = choices[0]
        for k in ("message", "content", "text", "output"):
            if isinstance(c, dict) and k in c:
                text = c[k]
                break
        if text is None:
            # fallback: stringify
            text = json.dumps(choices)
        return text
    return str(data)




def remove_header_footer_by_repetition(pages: List[List[str]]) -> List[List[str]]:
    """Simple heuristic: find lines that appear in >=50% of pages at top/bottom and remove them."""
    print("remove_header_footer_by_repetition method called")
    top_lines = [p[0] if p else "" for p in pages]
    bottom_lines = [p[-1] if p else "" for p in pages]
    def frequent(lines):
        counts = {}
        for l in lines:
            counts[l] = counts.get(l, 0) + 1
        threshold = max(1, int(0.5 * len(lines)))
        return {k for k,v in counts.items() if v >= threshold}
    tops = frequent(top_lines)
    bottoms = frequent(bottom_lines)
    cleaned = []
    for p in pages:
        print("=======")
        print(p)
        print("=======")
        if not p:
            cleaned.append(p)
            continue
        start = 1 if p[0] in tops else 0
        end = -1 if p[-1] in bottoms else len(p)
        cleaned.append(p[start:end if end!=len(p) else None])
    return cleaned

# Basic HTML/text cleaning for table cells
def _clean_cell(text: Optional[str]) -> str:
    if not text:
        return ""
    t = html.unescape(str(text))
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Normalize header labels to simple keys
def _normalize_header_label(label: str, fallback: str) -> str:
    if not label:
        return fallback
    lab = str(label).lower().strip()
    lab = re.sub(r"[^\w\s]", " ", lab)          # remove punctuation
    lab = re.sub(r"\s+", "_", lab)              # spaces -> underscore
    lab = re.sub(r"_+", "_", lab)               # collapse underscores
    lab = lab.strip("_")
    return lab or fallback

# PDF text + table extraction (robust version)
def extract_text_and_tables_from_pdf_bytes(
    file_bytes: bytes,
    source_name: str = "NCC.pdf",
    remove_header_from_tables: bool = True,
    remove_footer_from_tables: bool = True,
    header_ratio: float = 0.08,
    footer_ratio: float = 0.08,
    max_sent_len: int = 400,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (full_text, tables_converted).

    full_text: cleaned page text joined (header/footer removed via remove_header_footer_by_repetition)
    tables_converted: list of dicts:
        {
            "source": source_name,
            "page": page_number,
            "sentences": [ ... human sentences ... ],
            "raw": [ [row1], [row2], ... ]   # cleaned raw rows (strings)
        }
    """
    tables_converted: List[Dict[str, Any]] = []
    pages_lines: List[List[str]] = []
    text_pages: List[str] = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for pnum, page in enumerate(pdf.pages, start=1):
            # extract page text
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.debug("extract_text error on page %d: %s", pnum, e)
                text = ""
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            pages_lines.append(lines)
            text_pages.append(text)

            # compute bbox for table cropping
            y0 = int(page.height * (header_ratio if remove_header_from_tables else 0.0))
            y1 = int(page.height * (1.0 - (footer_ratio if remove_footer_from_tables else 0.0)))
            cropped = page.within_bbox((0, y0, page.width, y1))

            # extract tables robustly
            try:
                page_tables = cropped.extract_tables() or []
            except Exception as e:
                logger.debug("extract_tables error on page %d: %s", pnum, e)
                page_tables = []

            for tbl in page_tables:
                # cleaned raw rows (strings), keep only rows with at least one non-empty cell
                cleaned_rows = []
                for row in tbl or []:
                    cleaned = [_clean_cell(c) for c in (row or [])]
                    if any(cell for cell in cleaned):
                        cleaned_rows.append(cleaned)

                if not cleaned_rows:
                    continue
                # Header detection like earlier: find header in first up-to-3 rows
                header = None
                header_row_index: Optional[int] = None
                for ridx in range(min(3, len(cleaned_rows))):
                    row = cleaned_rows[ridx]
                    nonempty = [c for c in row if c]
                    threshold = max(2, int(0.5 * max(1, len(row))))
                    if len(nonempty) >= threshold:
                        header = row
                        header_row_index = ridx
                        break
                if header_row_index is None:
                    # fallback: use first row as header (but we will still keep raw rows)
                    header = cleaned_rows[0]
                    header_row_index = 0

                header_len = len(header)
                normalized_header = [
                    _normalize_header_label(h, f"col_{i+1}") for i, h in enumerate(header)
                ]

                # Build sentences from data rows (skip header row)
                data_rows = [r for i, r in enumerate(cleaned_rows) if i != header_row_index]
                sentences = []
                for row in data_rows:
                    pairs = []
                    for i in range(header_len):
                        h = normalized_header[i] if i < len(normalized_header) else f"col_{i+1}"
                        v = _clean_cell(row[i]) if i < len(row) else ""
                        if v:
                            if len(v) > max_sent_len // 2:
                                v = v[: (max_sent_len // 2) - 1] + "…"
                            pairs.append(f"{h} is {v}")
                    if not pairs:
                        continue
                    sent = "; ".join(pairs)
                    sent = f"On page {pnum} of {source_name}, {sent}."
                    print(sent)
                    if len(sent) > max_sent_len:
                        sent = sent[: max_sent_len - 1] + "…"                        
                    sentences.append(sent)

                # Append one table entry: keep original cleaned raw (not the pdfplumber raw which may contain None)
                tables_converted.append({
                    "source": source_name,
                    "page": pnum,
                    "sentences": sentences,           # human-readable sentences
                    "raw": cleaned_rows,             # cleaned raw rows (strings)
                })

    # Remove header/footer across pages for the textual content (assumes this helper exists)
    try:
        cleaned_pages = remove_header_footer_by_repetition(pages_lines)
    except Exception as e:
        logger.debug("remove_header_footer_by_repetition failed: %s", e)
        cleaned_pages = pages_lines

    # Join pages into paragraphs/sections
    paragraphs: List[str] = []
    for lines in cleaned_pages:
        if not lines:
            continue
        para = "\n".join(lines)
        paragraphs.extend([p.strip() for p in para.split('\n\n') if p.strip()])
    full_text = "\n\n".join(paragraphs)

    return full_text, tables_converted

# PDF text + table extraction (simple version)
def extract_text_and_tables_from_pdf(file_bytes: bytes) -> Tuple[str, List[Dict[str, Any]]]:
    """Return plain text (joined pages) and a list of converted tables (as sentence strings and raw rows).
    Each table dict: {"sentences": [...], "raw": [[...], ...]}
    """
    print("extract_text_and_tables_from_pdf method called") 
    tables_converted = []
    pages_lines = []
    text_pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            pages_lines.append(lines)
            text_pages.append(text)
            # extract tables
            try:
                page_tables = page.extract_tables()
            except Exception:
                page_tables = []
            for tbl in page_tables:
                # tbl is list of rows; convert to simple sentences
                sentences = []
                for row in tbl:
                    # join non-empty cells
                    cells = [str(c).strip() for c in row if c and str(c).strip()]
                    if not cells:
                        continue
                    # create a sentence like "<col1> is <col2>, <col3> ..." if possible
                    if len(cells) == 2:
                        sentences.append(f"{cells[0]} is {cells[1]}")
                    else:
                        sentences.append("; ".join(cells))
                if sentences:
                    tables_converted.append({"sentences": sentences, "raw": tbl})
    # remove headers/footers heuristically
    cleaned_pages = remove_header_footer_by_repetition(pages_lines)
    # join pages into paragraphs
    paragraphs = []
    for lines in cleaned_pages:
        if not lines:
            continue
        para = "\n".join(lines)
        paragraphs.extend([p.strip() for p in para.split('\n\n') if p.strip()])
    full_text = "\n\n".join(paragraphs)
    print(full_text)
    return full_text, tables_converted


def normalize_and_tokenize(text: str, do_lemmatize: bool = True, do_stem: bool = True) -> str:
    print("normalize_and_tokenize method called")
    txt = text.lower()
    tokens = word_tokenize(txt)
    tokens = [t for t in tokens if t.isalnum() and t not in STOP]
    if do_lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    if do_stem:
        tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# Simple paragraph splitter and merger (to avoid tiny chunks)
def split_paragraphs(text: str) -> List[str]:
    print("split_paragraphs method called")
    parts = re.split(r"\n\s*\n+", text.strip())
    merged: List[str] = []
    buf: List[str] = []
    for p in parts:
        if len(p.split()) < 20:
            buf.append(p)
        else:
            if buf:
                merged.append(" ".join(buf))
                buf = []
            merged.append(p)
    if buf:
        merged.append(" ".join(buf))
    return [p.strip() for p in merged if p.strip()]

# Word-based chunking with overlap
def chunk_text_by_words(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    print("chunk_text_by_words method called")
    words = text.split()
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks



# Contract-domain hints to bias retrieval
CONTRACT_KEYWORDS = [        
    "revenue","revenue from operations", "total income", "total expense", "profit before tax", 
    "segment revenue", "sales","topline","growth","ebitda","operating profit","operating margin",
    "reasearch and development expenses","r&d expense","selling and marketing expense",
    "ebitda margin","operating expense","cost of revenue","cost of goods sold","dividend","profit after tax",
    "dividend yield","dividend payout","cash dividend","interim dividend","total assets",
    "depreciation","amortization","capital expenditure","cash and cash equivalents","interim dividend",
    "cash generated from operations","payments to acquire investments","investments",
    "borrowings","long-term borrowings","short-term borrowings","income tax","tax expense",
    "financial instruments","trade receivables","trade payables","current liabilities",
    "securities","equity share capital","reserves","shareholder fund","security deposit",
    "loan to employees","loan to subsidiaries","loan from subsidiaries","deferred tax",
    "deferred contract cost","other current assets","other non-current assets","inter corporate deposits",
    "interest income","other borrowing costs","finance costs","other current liabilities",
    "withholding tax","short-term investments","long-term investments","other bank balances",
    "total liabilities","earnings per share","eps","net profit","net loss","net sales","other income",
    "final dividend","dividend per share","dps","share buyback","buyback","share repurchase",
    "operating cash flow","operating activities","investing activities","financing activities",
    "gross margin","net income","pat","eps","guidance","outlook","capex","opex","cash flow",
    "other equity","reserves","retained earnings","net worth","book value",
    "free cash flow","fiscal","fy","qoq","yoy","tax","interest","debt","leverage","working capital",
    "inventory","order book","backlog","pipeline","headwinds","tailwinds","risk","sensitivity","valuation",
    "multiple","pe","ev/ebitda","roc","roce","roe","buyback","cost","pricing","mix","gross profit",
    "units","volume","utilization","forecast","estimate","consensus","beat","miss","attrition","hiring",
    "hiring freeze","restructuring","layoff","reduction in force","ramp down","ramp up", "lease liabilities",     
]

SEED_TONE_PROMPTS = {
    "formal": "formal tone, legal/contract analysis style, precise clause references",
    "executive": "executive summary tone, risks/opportunities, recommended actions",
    "technical": "technical tone, security/privacy compliance and control details",
    "concise": "concise tone, short bullet points, plain language",
}
# Keyword injection
KEYWORD_PROMPT_MAP = {
    'compare': 'Please structure the answer as a comparison with pros/cons and a final recommendation.',
    'analyze': 'Please provide an analytical response, include causes, implications, and a short conclusion.'
}

# contract query expansion

def expand_contract_query(q: str) -> str:
    print("expand_contract_query method called")
    ql = q.lower()
    #extra = [w for w in CONTRACT_KEYWORDS if w not in ql]
    extra = ""
    return q + ("\n\nFocus on: " + ", ".join(extra) if extra else "")


def detect_keywords_and_inject(prompt: str) -> Tuple[str, List[str]]:
    print("detect_keywords_and_inject method called")
    found = [k for k in KEYWORD_PROMPT_MAP.keys() if k in prompt.lower()]
    injections = [KEYWORD_PROMPT_MAP[k] for k in found]
    injection_text = "\n".join(injections)
    new_prompt = (injection_text + "\n" + prompt) if injection_text else prompt
    return new_prompt, found

def normalize_rows(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / n

# tone detection from embeddings
class ToneDetector:
    def __init__(self, embed_model: str, tone_vectors: Dict[str, np.ndarray]):
        self.embed_model = embed_model
        self.tone_vectors = tone_vectors
    @classmethod
    def build(cls, embed_model: str):
        texts = list(SEED_TONE_PROMPTS.values())
        E = normalize_rows(ollama_embed(texts))
        tone_vectors = {k: E[i] for i, k in enumerate(SEED_TONE_PROMPTS.keys())}
        return cls(embed_model, tone_vectors)
    def detect(self, prompt: str) -> str:
        q = normalize_rows(ollama_embed([prompt]))[0]
        best, best_s = None, -1.0
        for tone, vec in self.tone_vectors.items():
            s = float(vec @ q)
            if s > best_s:
                best, best_s = tone, s
        return best or "formal"


# Tone detection using simple heuristics (VADER could be used if available)
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
_sia = SentimentIntensityAnalyzer()

def detect_tone(text: str) -> str:
    print("detect_tone method called")
    scores = _sia.polarity_scores(text)
    if scores['compound'] >= 0.4:
        return 'positive'
    elif scores['compound'] <= -0.4:
        return 'negative'
    else:
        return 'neutral'

# Reranking: combine cosine similarity and keyword matches and tone match
def rerank_results(query_embedding: np.ndarray, candidate_embeddings: np.ndarray, candidates: List[Dict[str, Any]], prompt_keywords: List[str], prompt_tone: str) -> List[Dict[str, Any]]:
    # cosine similarities
    print("rerank_results method called")
    sims = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings).flatten()
    results = []
    for i, c in enumerate(candidates):
        weight = float(sims[i])
        # keyword boost
        text = c.get('document', '')
        kw_count = sum(1 for k in prompt_keywords if k in text.lower())
        weight += 0.05 * kw_count
        # tone boost if tone words present (simple)
        if prompt_tone == 'positive' and any(w in text.lower() for w in ['good', 'improve', 'strength']):
            weight += 0.02
        if prompt_tone == 'negative' and any(w in text.lower() for w in ['risk', 'fail', 'issue']):
            weight += 0.02
        results.append({**c, 'score': float(weight), 'raw_sim': float(sims[i])})
    # sort desc
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    return results_sorted

def _safe_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms

def mmr(
    qvec: np.ndarray,
    doc_vecs: np.ndarray,
    candidates: List[int],
    lambda_mult: float = 0.55,
    k: int = 10,
    metric: str = "cosine",            # "cosine" or "dot"
    normalize: bool = True,            # if True and metric=="cosine", normalize vectors
    random_seed: Optional[int] = None, # for tie-break randomness (deterministic if provided)
) -> List[Dict[str, Any]]:
    """
    Vectorized MMR that returns selection metadata as a list of dicts.

    Returns:
        List[Dict[str, Any]] where each dict has:
          - "doc_id": int (index into doc_vecs)
          - "score": float (MMR score used to pick this doc)
          - "relevance": float (similarity of doc to query)
          - "redundancy": float (max similarity to previously selected docs)
          - "rank": int (1-based order of selection)
    """
    out: List[Dict[str, Any]] = []
    if len(candidates) == 0:
        return out

    rng = np.random.default_rng(random_seed) if random_seed is not None else None

    cand_idx = np.asarray(candidates, dtype=int)
    cand_vecs = doc_vecs[cand_idx]  # (m, D)
    q = np.asarray(qvec).reshape(-1)

    if metric not in {"cosine", "dot"}:
        raise ValueError("metric must be 'cosine' or 'dot'")

    if metric == "cosine" and normalize:
        cand_vecs = _safe_normalize_rows(cand_vecs)
        q = q / max(np.linalg.norm(q), 1e-12)

    # relevance: similarity of each candidate to query
    q_sims = cand_vecs @ q            # shape (m,)

    m = cand_vecs.shape[0]
    k = min(k, m)

    # candidate×candidate similarity matrix
    sims = cand_vecs @ cand_vecs.T    # shape (m, m)

    selected_pos: List[int] = []
    remaining = np.ones(m, dtype=bool)

    # tiny jitter for tie-breaking if rng provided
    noise_scale = 1e-12

    for _ in range(k):
        if not selected_pos:
            scores = q_sims.copy()
            redundancy = np.zeros_like(q_sims)
        else:
            redundancy = np.max(sims[:, selected_pos], axis=1)  # for each candidate
            scores = lambda_mult * q_sims - (1.0 - lambda_mult) * redundancy

        # mask already picked
        scores[~remaining] = -np.inf

        # jitter
        if rng is not None:
            scores = scores + rng.normal(scale=noise_scale, size=scores.shape)

        pick_pos = int(np.argmax(scores))
        if not remaining[pick_pos]:
            break

        # compute final numbers for this pick (without jitter)
        rel = float(q_sims[pick_pos])
        red = float(redundancy[pick_pos]) if selected_pos else 0.0
        score = float(lambda_mult * rel - (1.0 - lambda_mult) * red)

        out.append({
            "doc_id": int(cand_idx[pick_pos]),
            "score": score,
            "relevance": rel,
            "redundancy": red,
            "rank": len(out) + 1,
        })

        selected_pos.append(pick_pos)
        remaining[pick_pos] = False

        if not remaining.any():
            break

    return out


def mmr2(qvec: np.ndarray, doc_vecs: np.ndarray, candidates: List[int], lambda_mult: float = 0.55, k: int = 10) -> List[int]:
    selected: List[int] = []
    if not candidates:
        return selected
    q_sims = (doc_vecs[candidates] @ qvec).tolist()
    picked = set()
    while len(selected) < min(k, len(candidates)):
        if not selected:
            best_idx = int(np.argmax(q_sims))
            chosen = candidates[best_idx]
            selected.append(chosen)
            picked.add(chosen)
            continue
        scores = []
        for idx_i, i in enumerate(candidates):
            if i in picked:
                scores.append(-1e9)
                continue
            rel = q_sims[idx_i]
            red = 0.0
            for j in selected:
                red = max(red, float(doc_vecs[i] @ doc_vecs[j]))
            scores.append(lambda_mult * rel - (1 - lambda_mult) * red)
        best_idx = int(np.argmax(scores))
        chosen = candidates[best_idx]
        if chosen in picked:
            break
        selected.append(chosen)
        picked.add(chosen)
    return selected

#re-index chromadb
def reindex_chromadb(ids,chunks, embeddings, metadatas,chroma_dir="./chroma_store", collection_name=CHROMA_COLLECTION):
    print("reindex_chromadb method called")
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)
    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

# Logging helper

def log_event(event: Dict[str, Any]):
    logger.info(json.dumps({"ts": int(time.time()), **event}))

# Streamlit UI
st.set_page_config(page_title="MS Contracts KB", layout='wide')

st.title("Managed Services Contract Knowledge Base")

page = st.sidebar.selectbox("Page", ["Prompt", "Knowledge Base"])

if page == "Knowledge Base":
    st.header("Build Knowledge Base from PDFs")
    uploaded = st.file_uploader("Upload PDFs (multiple)", type=['pdf'], accept_multiple_files=True)
    embed_model = EMBED_MODEL #st.text_input("Embedding model (Ollama)", value=EMBED_MODEL)
    chunk_size = 200 #st.number_input("Chunk size (words)", value=200, min_value=50)
    overlap = 50 #st.number_input("Chunk overlap (words)", value=50, min_value=0)
    do_lemmatize = False #st.checkbox("Do lemmatization", value=True)
    do_stem = False #st.checkbox("Do stemming", value=False)
    
    if st.button("Process & Index"):
        if not uploaded:
            st.warning("Please upload at least one PDF")
        else:
            all_docs = []
            for f in uploaded:
                raw = f.read()
                #full_text, tables = extract_text_and_tables_from_pdf(raw)
                full_text, tables = extract_text_and_tables_from_pdf_bytes(raw, source_name=f.name)
                # log table conversions
                if tables:
                    for t in tables:
                        log_event({"type": "table_conversion", "filename": f.name, "sentences": t['sentences'], "raw": t['raw']})
                # normalize + tokenize + chunk
                normalized = normalize_and_tokenize(full_text, do_lemmatize=do_lemmatize, do_stem=do_stem)
                chunks = chunk_text_by_words(normalized, chunk_size=int(chunk_size), overlap=int(overlap))
                #chunks = split_paragraphs(normalized)
                # prepare docs for embedding
                texts_for_embed = chunks
                embeddings = ollama_embed(texts_for_embed,EMBED_MODEL,3)
                # insert into chroma
                ids = [str(uuid4()) for _ in chunks]
                metadatas = [{"source_file": f.name, "file_type":f.type, "file_size": f.size, "chunk_index": i} for i in range(len(chunks))]
                
                collection.add(ids=ids, documents=chunks, embeddings=embeddings.tolist(), metadatas=metadatas)
                #save_chunks_to_chromadb(collection, embeddings)
                st.success(f"Indexed {len(chunks)} chunks from {f.name}")
                log_event({"type": "indexed_file", "filename": f.name, "n_chunks": len(chunks)})
    if st.button("Re-index KB"):
        if not uploaded:
            st.warning("Please upload at least one PDF")
        else:
            all_docs = []
            for f in uploaded:
                raw = f.read()
                #full_text, tables = extract_text_and_tables_from_pdf(raw)
                full_text, tables = extract_text_and_tables_from_pdf_bytes(raw, source_name=f.name)
                # log table conversions
                if tables:
                    for t in tables:
                        log_event({"type": "table_conversion", "filename": f.name, "sentences": t['sentences'], "raw": t['raw']})
                # normalize + tokenize + chunk
                normalized = normalize_and_tokenize(full_text, do_lemmatize=do_lemmatize, do_stem=do_stem)
                #chunks = chunk_text_by_words(normalized, chunk_size=int(chunk_size), overlap=int(overlap))
                chunks = split_paragraphs(normalized)
                # prepare docs for embedding
                texts_for_embed = chunks
                embeddings = ollama_embed(texts_for_embed,EMBED_MODEL,3)
                # insert into chroma
                ids = [str(uuid4()) for _ in chunks]
                metadatas = [{"source_file": f.name, "file_type":f.type, "file_size": f.size, "chunk_index": i} for i in range(len(chunks))]
                reindex_chromadb(ids=ids,chunks=chunks, embeddings=embeddings,metadatas=metadatas)
                st.success("Re-indexed KB in ChromaDB.")
 
    
elif page == "Prompt":
    st.header("Ask the KB")
    prompt = st.text_area("Prompt", height=200)
    filename_filter = st.text_input("Optional: filter results by filename (exact match)")
    top_k = 5 #st.number_input("Top K to retrieve", value=10, min_value=1)
    rerank_n = 5 #st.number_input("Rerank top N (from retrieved)", value=10, min_value=1)
    #inject_keywords = st.checkbox("Enable keyword injection", value=True)
    inject_keywords = True
    #use_tone = st.checkbox("Use tone detection", value=True)
    use_tone = True

    if st.button("Run"):
        if not prompt.strip():
            st.warning("Please enter a prompt")
        else:
            prompt = expand_contract_query(prompt)
            # keyword injection
            processed_prompt, found_keywords = (detect_keywords_and_inject(prompt) if inject_keywords else (prompt, []))
            # tone
            #tone = detect_tone(prompt) if use_tone else 'neutral'
            td = ToneDetector.build(EMBED_MODEL) if use_tone else None
            tone = td.detect(prompt) if td else 'formal'
            # build embedding for query
            q_emb = ollama_embed([processed_prompt], model=EMBED_MODEL,debug_raw=False)[0]
            #q_emb = processed_prompt
            print(q_emb)    
            # Query chroma
            filter_metadata = {"source_file": filename_filter} if filename_filter else None
            # Chroma's query: try to use embeddings query if supported
            try:
                query_res = collection.query(query_embeddings=[q_emb.tolist()], n_results=int(top_k), where=filter_metadata)
                # query_res typically contains 'ids','metadatas','documents','distances' keys
                docs = []
                docs_list = query_res.get('documents') or query_res.get('results') or []
                # normalize into candidates
                ids = query_res.get('ids', [[]])[0]
                documents = query_res.get('documents', [[]])[0]
                metadatas = query_res.get('metadatas', [[]])[0]
                distances = query_res.get('distances', [[]])[0]
                # fetch embeddings for rerank (we'll ask chroma for embeddings per id if API supports, otherwise store embedding in metadata)
                # Here we assume embeddings aren't retrievable; a workaround would be to re-embed the documents (cheap for a few)
                candidate_embeddings = np.array([ollama_embed([d], model=EMBED_MODEL)[0] for d in documents])
                candidates = []
                for i, doc in enumerate(documents):
                    candidates.append({"id": ids[i], "document": doc, "metadata": metadatas[i], "distance": distances[i] if i < len(distances) else None})
            except Exception as e:
                st.error(f"Chroma query failed: {e}")
                candidates = []
                candidate_embeddings = np.zeros((0, q_emb.shape[0]))

            # Rerank
            if len(candidates) > 0:
                reranked = rerank_results(q_emb, candidate_embeddings, candidates, found_keywords, tone)
                #reranked = mmr(q_emb, candidate_embeddings, [i for i in range(len(candidates))], k=int(rerank_n))
                # log rerank data
                log_event({"type": "rerank", "prompt": prompt, "found_keywords": found_keywords, "tone": tone, "reranked_ids": [r['id'] for r in reranked]})
                # prepare context from top M
                top_ctx = "\n\n".join([f"Source: {r['metadata'].get('source_file','?')}\n{r['document']}" for r in reranked[:int(rerank_n)]])
            else:
                top_ctx = ''
                reranked = []

            # Create final prompt for LLM
            system_instructions = "You are an expert in contract document review with keen focus on ITIL processes. Use the provided context to answer concisely. If context is insufficient, say so."
            final_prompt = f"Context:\n{top_ctx}\n\nUser Prompt:\n{processed_prompt}"
            print(final_prompt)
            # Call LLM
            try:
                answer = ollama_chat(final_prompt, model=LLM_MODEL, system=system_instructions)
            except Exception as e:
                st.error(f"LLM call failed: {e}")
                answer = ""

            # Display
            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved & Reranked Chunks")
            for r in reranked[:int(rerank_n)]:
                st.markdown(f"**Source:** {r['metadata'].get('source_file','?')} — **Score:** {r['score']:.4f}")
                st.write(r['document'])
                st.write(r['metadata'])

            # log detailed chunk data
            log_event({
                "type": "query_complete",
                "prompt": prompt,
                "processed_prompt": processed_prompt,
                "found_keywords": found_keywords,
                "tone": tone,
                "retrieved_count": len(candidates),
                "reranked_top_ids": [r['id'] for r in reranked[:int(rerank_n)]],
            })

# Footer note
#st.sidebar.markdown("---")
#st.sidebar.markdown("Make sure Ollama server is reachable and models are configured. Update EMBED_MODEL and LLM_MODEL at the top of the script.")
