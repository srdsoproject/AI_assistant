"""
Safety Branch AI Assistant (Streamlit) - Groq + Gemini
Flow for every question:
  news question  -> live headlines
  small talk     -> normal chat
  otherwise      -> 1) search YOUR letters/documents (local, Drive & uploaded)  2) if not found -> search the internet
"""
import os, re, io, glob, json, time, shutil, hashlib, zipfile, tempfile, threading, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from urllib.parse import urlencode, quote_plus
from email.utils import parsedate_to_datetime
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Safety Branch AI", page_icon="🚆", layout="centered")

DOC_DIR = os.path.join(os.path.dirname(__file__), "documents")
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "safety_uploaded_docs")
INDEX_CACHE_DIR = os.path.join(tempfile.gettempdir(), "safety_index_cache")
DRIVE_SYNC_TTL = 300  # seconds to skip redundant Drive API checks on quick reloads
TOP_K, MIN_SCORE, MAX_OCR_PAGES = 6, 0.06, 400


def cfg(key, default=""):
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


PROVIDER = cfg("LLM_PROVIDER", "groq").lower()
FALLBACK = cfg("LLM_FALLBACK", "gemini").lower()
GROQ_MODEL = cfg("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = cfg("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_BACKUPS = ["openai/gpt-oss-20b"]
GEMINI_BACKUPS = ["gemini-flash-latest"]

# ---------------- prompts ----------------
RAG_SYSTEM = """You are the Safety Branch Assistant for Indian Railways officials and staff.
You get CONTEXT: excerpts of official letters/documents, each labeled with letter number, date and subject.
Answer ONLY from the CONTEXT. Never invent rules, limits, letter numbers, designations or dates.
Format your reply as:
**Answer:** short, direct.
**Applicable letter(s):** number, date, subject (cite as [1], [2]).
**Recommended action:** clear steps the user should take.
**Note:** if several letters cover the subject, the latest-dated one normally prevails unless a letter says otherwise; mention any letter that modifies/supersedes another.
Reply in the user's language (Hindi, English, Marathi or Hinglish). Be concise.
If the CONTEXT does not contain the answer, reply with exactly: NOT_FOUND_IN_DOCUMENTS"""

WEB_SYSTEM = """You are the Safety Branch Assistant for Indian Railways staff. The department's own letters had no
answer to this question, so use web search. Prefer official sources (indianrailways.gov.in, railwayboard.gov.in,
pib.gov.in, cr.indianrailways.gov.in, other .gov.in sites). Give a concise answer and a recommended action.
Start with one line: "🌐 Not found in the department letters - this is from the internet."
Never invent rule/letter numbers. End by telling the user to verify against the latest official instructions.
Reply in the user's language."""

WEB_SNIPPET_SYSTEM = """You are the Safety Branch Assistant. You get numbered web search results (title, snippet).
Answer ONLY from them, cite [1], [2]. Start with: "🌐 Not found in the department letters - this is from the internet."
If the results don't answer, say so. Tell the user to verify against official instructions. Reply in the user's language."""

CHAT_SYSTEM = """You are the Safety Branch Assistant, a friendly AI for Indian Railways Safety Branch staff.
Talk naturally: greet back, small talk, explanations and drafting help are fine. Warm, quick, a bit casual
(light emoji ok), short unless detail is needed. Reply in the user's language (Hindi, English, Marathi, Hinglish).
For specific railway rules, rule numbers, limits or SOP steps, never invent them: give general guidance only,
say it is not from department documents, and tell them to verify with the official manual/letter."""

LETTER_SYSTEM = """You are the Safety Branch Assistant. Summarize this official letter for end users, using ONLY its text.
Give: **Letter no. / date / issued by**, **Subject**, **Key instructions** (bullets), **Who must act (designations / departments)**,
**Deadlines / compliance**, **Recommended action for end users**. If something isn't stated, write "not stated".
If the text looks garbled (bad scan), say so."""

COMPARE_SYSTEM = """You are the Safety Branch Assistant for Indian Railways officials and staff.
You are given 2 or more official letters/documents (each with letter no., date, subject, and text).
Compare them thoroughly using ONLY the provided text. Never invent rules, limits, letter numbers, designations, or dates.
Format your comparison clearly:
**1. Overview Table:** Markdown table comparing Document, Letter No., Date, Issued By, and Subject.
**2. Key Differences & Changes:** Highlight what changed between earlier and later letters (rules, limits, procedures, responsibilities, deadlines).
**3. Superseding / Prevailing Status:** Identify the latest instruction and whether any document modifies, clarifies, or supersedes another.
**4. Common Instructions:** Rules or requirements that remain consistent across the documents.
**5. Recommended Action for Staff:** Clear, consolidated steps for end users based on the prevailing instructions.
Reply in the user's requested language (English, हिंदी, or मराठी). Be concise and accurate."""

NEWS_SYSTEM = """You are the Safety Branch Assistant for Indian Railways staff. You get recent NEWS HEADLINES
(numbered, with source and date). Summarize based ONLY on these headlines; cite [1], [2]; group related items;
mention dates; never add details not in the headlines (you only see titles) - tell the user to open the link for
details. Reply in the user's language, short and clear. News is not official instructions."""

SMALLTALK = re.compile(r"^\s*(hi+|hello+|hey+|hii+|yo|namaste|namaskar|नमस्ते|हाय|हैलो|good (morning|afternoon|evening|night)|"
                       r"thanks?|thank you|thx|ok(ay)?|cool|nice|bye|who are you|what can you do|how are you|"
                       r"what'?s up|help)\b[\s\S]{0,40}$", re.I)

SUGGESTIONS = ["📰 Latest Solapur Division news", "👋 Hi, what can you do?",
               "📑 Which letter covers SPAD prevention?", "📝 Help me draft a safety circular"]

# ---------------- styling ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Space Grotesk', sans-serif !important; }
.stApp {
  background: radial-gradient(1200px 600px at 10% -10%, #3b1d8f55, transparent),
              radial-gradient(900px 500px at 100% 0%, #ff2e9a33, transparent),
              radial-gradient(800px 500px at 50% 120%, #00e5ff22, transparent), #0a0b14;
  color: #eceefc; }
header[data-testid="stHeader"], footer, #MainMenu {visibility: hidden; height: 0;}
.block-container {padding-top: 1.5rem; max-width: 840px;}
.hero {text-align:center; padding: 8px 0 14px;}
.hero h1 {font-size: 2.3rem; font-weight: 700; margin: 0;
  background: linear-gradient(90deg,#b6ff3c,#00e5ff,#ff2e9a); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.hero p {color:#9aa0c3; margin: 4px 0 0; font-size: .95rem;}
.pill {display:inline-block; padding: 4px 12px; border-radius: 999px; font-size:.75rem;
  border:1px solid #ffffff22; background:#ffffff0d; margin: 4px 4px 0;}
.pill.ok {color:#b6ff3c; border-color:#b6ff3c55;} .pill.warn {color:#ffd23c; border-color:#ffd23c55;}
[data-testid="stChatMessage"] {background:#ffffff0a; border:1px solid #ffffff14; border-radius:18px;
  backdrop-filter: blur(10px); padding:12px 16px; margin-bottom:10px; box-shadow:0 6px 30px #00000040;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: linear-gradient(135deg,#6d3bff44,#ff2e9a33); border-color:#ff2e9a44;}
.badge {font-size:.7rem; padding:2px 9px; border-radius:999px; margin-bottom:6px; display:inline-block; border:1px solid #ffffff22;}
.badge.doc {color:#b6ff3c; background:#b6ff3c14;} .badge.chat {color:#00e5ff; background:#00e5ff14;}
.badge.web {color:#ffb03c; background:#ffb03c14;}
[data-testid="stChatInput"] {background:#ffffff0d; border:1px solid #ffffff22; border-radius:16px;}
[data-testid="stChatInput"] textarea {color:#fff !important;}
.stButton>button {background:#ffffff0d; color:#eceefc; border:1px solid #ffffff22; border-radius:14px;
  padding:.55rem .8rem; font-weight:500; transition: all .2s; width:100%;}
.stButton>button:hover {border-color:#b6ff3c; color:#b6ff3c; transform: translateY(-2px); box-shadow:0 6px 20px #b6ff3c22;}
[data-testid="stSidebar"] {background:#0d0e1acc; border-right:1px solid #ffffff14;}
.newscard {background:#ffffff0a; border:1px solid #ffffff14; border-radius:16px; padding:12px 16px; margin-bottom:10px; transition: all .2s;}
.newscard:hover {border-color:#00e5ff66; transform: translateY(-2px);}
.newscard a {color:#eceefc; text-decoration:none; font-weight:500; line-height:1.35;} .newscard a:hover {color:#b6ff3c;}
.newsmeta {color:#9aa0c3; font-size:.75rem; margin-top:5px;}
.spot {border:1px solid #b6ff3c55; background:linear-gradient(135deg,#b6ff3c12,#00e5ff10); border-radius:18px; padding:6px 16px 2px; margin-bottom:14px;}
div[role="radiogroup"] {gap:.4rem; flex-wrap:wrap;}
.foot {text-align:center; color:#6b7094; font-size:.72rem; margin-top:10px;}
</style>""", unsafe_allow_html=True)


# ---------------- password gate ----------------
def check_password():
    pw = cfg("APP_PASSWORD")
    if not pw or st.session_state.get("auth"):
        return True
    st.markdown('<div class="hero"><h1>🚆 Safety Branch AI</h1><p>Locked in 🔒 Enter the password</p></div>', unsafe_allow_html=True)
    entered = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Password…")
    if st.button("Let me in 🚀"):
        if entered == pw:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")
    return False


# ---------------- letter splitting & metadata ----------------
_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]
MONTH_NUM = {m: i for i, m in enumerate(_MONTHS, 1)}
MONTH_NUM.update({m[:3]: i for m, i in list(MONTH_NUM.items())})

SUB_RE = re.compile(r"(?<![A-Za-z])(sub(?:ject)?|विषय)\s*[:\-–.]+", re.I)
NO_RE = re.compile(r"(?<![A-Za-z])(?:letter\s*)?(?:file\s*)?no\.?\s*[:\-–]?\s*([A-Z0-9][^\n]{2,50})", re.I)
DATE_NUM_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})\b")
DATE_TXT_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?[ ,.\-]+([A-Za-z]{3,9})[ ,.\-]+(\d{4})\b")
HDR_RE = re.compile(
    r"(government of india|ministry of railways|railway board|central railway|"
    r"office of the|divisional railway manager|additional divisional railway manager|"
    r"senior divisional(?:\s+safety)?\s+officer|sr\.?\s*dso|divisional safety officer|"
    r"principal chief safety officer|pcso|chief safety officer|commissioner of railway safety|"
    r"safety branch|solapur division|भारत सरकार|रेल मंत्रालय|मध्य रेल|मंडल रेल|"
    r"वरिष्ठ मंडल संरक्षा अधिकारी|मुख्य संरक्षा अधिकारी|संरक्षा शाखा)",
    re.I,
)
ISSUER_RE = re.compile(
    r"(railway|division|zone|board|ministry|office|safety|sanraksha|officer|manager|"
    r"counsellor|inspector|superintendent|sr\.?\s*dso|drm|adrm|pcso|cso|crs|"
    r"रेल|मंडल|कार्यालय|संरक्षा|अधिकारी|प्रबंधक|निरीक्षक|शाखा)",
    re.I,
)


def parse_date(s):
    """Return datetime.date or None from the first date found in s."""
    if not s:
        return None
    m = DATE_NUM_RE.search(s)
    try:
        if m:
            d, mo, y = int(m[1]), int(m[2]), int(m[3])
            y += 2000 if y < 100 else 0
            return dt.date(y, mo, d)
        m = DATE_TXT_RE.search(s)
        if m and m[2].lower() in MONTH_NUM:
            return dt.date(int(m[3]), MONTH_NUM[m[2].lower()], int(m[1]))
    except ValueError:
        pass
    return None


def _head(text, n=14):
    return [l.strip() for l in (text or "").splitlines() if l.strip()][:n]


def is_letter_start(text):
    head = " ".join(_head(text, 12))
    if not SUB_RE.search(head):
        return False
    return bool(NO_RE.search(head) or DATE_NUM_RE.search(head) or DATE_TXT_RE.search(head) or HDR_RE.search(head))


def extract_meta(text):
    lines = _head(text, 16)
    head = "\n".join(lines)
    no = ""
    m = NO_RE.search(head)
    if m:
        no = re.split(r"\s{2,}|\bdate[d]?\b|\bdt\b", m.group(1), flags=re.I)[0].strip(" :.-,")
    d = parse_date(head)
    subject = ""
    m = SUB_RE.search(head)
    if m:
        tail = head[m.end():].split("\n")
        subject = tail[0].strip()
        if len(subject) >= 70 and len(tail) > 1 and not SUB_RE.search(tail[1]) \
                and not re.match(r"(ref|sir|madam|dear)", tail[1].strip(), re.I):
            subject += " " + tail[1].strip()
        subject = re.split(r"\b(?:ref|reference)\b[:.]", subject, flags=re.I)[0].strip(" :.-")[:200]
    issuer = next((l[:90] for l in lines[:6] if ISSUER_RE.search(l)), "")
    return {"no": no, "date": d.isoformat() if d else "", "subject": subject, "issuer": issuer}


def split_letters(pages, source="", path=""):
    """pages: list of (page_no, text). Returns list of letter dicts."""
    letters, cur = [], None
    for pno, text in pages:
        if not (text or "").strip():
            continue
        if cur is None or is_letter_start(text):
            if cur:
                letters.append(cur)
            cur = {"source": source, "path": path, "pages": [], **extract_meta(text)}
        cur["pages"].append((pno, text))
    if cur:
        letters.append(cur)
    for i, l in enumerate(letters, 1):
        l["id"] = f"{source}#{i}"
        l["page_start"], l["page_end"] = l["pages"][0][0], l["pages"][-1][0]
        l["text"] = "\n".join(t for _, t in l["pages"])
    return letters


def chunk_letters(letters, words=220, overlap=40):
    chunks = []
    for l in letters:
        prefix = f"File: {l['source']} Letter No. {l['no']} dated {l['date']} Sub: {l['subject']}. "
        for pno, text in l["pages"]:
            w = re.sub(r"\s+", " ", text or "").split()
            for i in range(0, max(len(w), 1), words - overlap):
                piece = " ".join(w[i:i + words])
                if len(piece) > 40:
                    chunks.append({"letter_id": l["id"], "source": l["source"], "path": l.get("path", ""),
                                   "page": f"p.{pno}", "no": l["no"], "date": l["date"],
                                   "subject": l["subject"], "text": piece, "search": prefix + piece})
    return chunks


# ---------------- Google Drive (private folder via service account) ----------------
DRIVE_DIR = os.path.join(tempfile.gettempdir(), "safety_drive_docs")
DRIVE_SA = os.path.join(DRIVE_DIR, "private")   # service-account downloads
DRIVE_PUB = os.path.join(DRIVE_DIR, "public")   # public-link downloads
GDOC = "application/vnd.google-apps.document"


def _drive_list(svc, folder_id, depth=0):
    files, token = [], None
    while True:
        r = svc.files().list(q=f"'{folder_id}' in parents and trashed=false",
                             fields="nextPageToken, files(id,name,mimeType,modifiedTime)", pageSize=200,
                             pageToken=token, supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
        for f in r.get("files", []):
            if f["mimeType"] == "application/vnd.google-apps.folder":
                if depth < 3:
                    files += _drive_list(svc, f["id"], depth + 1)
            else:
                files.append(f)
        token = r.get("nextPageToken")
        if not token:
            return files


def _sync_public(folder):
    """Simple mode: folder shared as 'Anyone with the link - Viewer'. No credentials needed."""
    try:
        import gdown
        os.makedirs(DRIVE_PUB, exist_ok=True)
        gdown.download_folder(id=folder, output=DRIVE_PUB, quiet=True, use_cookies=False,
                              remaining_ok=True, resume=True)
        n = sum(len(fs) for _, _, fs in os.walk(DRIVE_PUB))
        if not n:
            return {"enabled": True, "files": 0, "mode": "public",
                    "error": "No files downloaded. Share the folder as 'Anyone with the link - Viewer' (or set up the private service-account mode)."}
        return {"enabled": True, "files": n, "mode": "public", "error": None}
    except Exception as e:
        return {"enabled": True, "files": 0, "mode": "public",
                "error": f"{str(e)[:250]} - check the folder is shared as 'Anyone with the link - Viewer'."}


def clear_drive_sync_cache():
    sync_stamp = os.path.join(DRIVE_DIR, "_last_sync.json")
    if os.path.exists(sync_stamp):
        os.remove(sync_stamp)


def _get_drive_cfg():
    folder = cfg("GDRIVE_FOLDER_ID")
    sa_info = None
    try:
        if "gcp_service_account" in st.secrets:
            sa_info = dict(st.secrets["gcp_service_account"])
    except Exception:
        pass
    return folder, sa_info


def sync_drive(folder=None, sa_info=None):
    """Download new/changed PDF, DOCX, TXT, MD and Google Docs from the Drive folder. Thread-safe."""
    if folder is None:
        folder, sa_info = _get_drive_cfg()
    os.makedirs(DRIVE_DIR, exist_ok=True)
    sync_stamp = os.path.join(DRIVE_DIR, "_last_sync.json")
    if folder and os.path.exists(sync_stamp):
        try:
            st_data = json.load(open(sync_stamp))
            if st_data.get("folder") == folder and (time.time() - st_data.get("ts", 0)) < DRIVE_SYNC_TTL:
                return st_data.get("status", {"enabled": True, "files": 0, "error": None, "mode": st_data.get("mode")})
        except Exception:
            pass
    if not folder:
        return {"enabled": False, "files": 0, "error": None, "mode": None}
    if not sa_info:
        status = _sync_public(folder)
        if not status.get("error"):
            try:
                with open(sync_stamp, "w") as fh:
                    json.dump({"folder": folder, "ts": time.time(), "mode": "public", "status": status}, fh)
            except Exception:
                pass
        return status
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        info = dict(sa_info)
        info["private_key"] = info["private_key"].replace("\\n", "\n")
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/drive.readonly"])
        svc = build("drive", "v3", credentials=creds, cache_discovery=False)
        os.makedirs(DRIVE_SA, exist_ok=True)
        mpath = os.path.join(DRIVE_SA, "_manifest.json")
        manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}

        to_dl, n = [], 0
        for f in _drive_list(svc, folder):
            name = re.sub(r"[^\w.\-() ]", "_", f["name"])
            export = f["mimeType"] == GDOC
            if not export and not name.lower().endswith((".pdf", ".docx", ".txt", ".md")):
                continue
            if export and not name.lower().endswith(".pdf"):
                name += ".pdf"
            local = os.path.join(DRIVE_SA, f"{f['id'][:6]}_{name}")
            if manifest.get(f["id"]) == f["modifiedTime"] and os.path.exists(local):
                n += 1
            else:
                to_dl.append((f, export, local))

        def _dl_one(item):
            f, export, local = item
            s_local = build("drive", "v3", credentials=creds, cache_discovery=False)
            req = (s_local.files().export_media(fileId=f["id"], mimeType="application/pdf") if export
                   else s_local.files().get_media(fileId=f["id"], supportsAllDrives=True))
            buf, done = io.BytesIO(), False
            dl = MediaIoBaseDownload(buf, req, chunksize=5 * 1024 * 1024)
            while not done:
                _, done = dl.next_chunk()
            with open(local, "wb") as fh:
                fh.write(buf.getvalue())
            return f["id"], f["modifiedTime"]

        if to_dl:
            with ThreadPoolExecutor(max_workers=min(4, len(to_dl))) as pool:
                for fid, mtime in pool.map(_dl_one, to_dl):
                    manifest[fid] = mtime
                    n += 1

        with open(mpath, "w") as fh:
            json.dump(manifest, fh)
        status = {"enabled": True, "files": n, "error": None, "mode": "private"}
        with open(sync_stamp, "w") as fh:
            json.dump({"folder": folder, "ts": time.time(), "mode": "private", "status": status}, fh)
        return status
    except Exception as e:
        return {"enabled": True, "files": 0, "error": str(e)[:300], "mode": "private"}


# ---------------- documents: read (with OCR fallback), split into letters, index ----------------
def ocr_available():
    try:
        import pytesseract, pypdfium2  # noqa
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def read_pdf_pages(path, stats):
    from pypdf import PdfReader
    pages, pdf = [], None
    for i, p in enumerate(PdfReader(path).pages, 1):
        text = (p.extract_text() or "").strip()
        if len(text) < 30:                       # probably a scanned image page
            if stats["ocr_ok"] and stats["ocr_done"] < MAX_OCR_PAGES:
                try:
                    import pypdfium2 as pdfium, pytesseract
                    pdf = pdf or pdfium.PdfDocument(path)
                    img = pdf[i - 1].render(scale=1.5).to_pil()
                    try:
                        text = pytesseract.image_to_string(img, lang="eng+hin+mar")
                    except Exception:
                        text = pytesseract.image_to_string(img, lang="eng")
                    stats["ocr_done"] += 1
                except Exception:
                    pass
            if len(text.strip()) < 30:
                stats["unread"] += 1
        pages.append((i, text))
    return pages


def read_other(path):
    ext = path.lower().rsplit(".", 1)[-1]
    if ext == "docx":
        import docx
        return [(1, "\n".join(p.text for p in docx.Document(path).paragraphs))]
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [(1, f.read())]


def _parse_file_cached(path, name, ext, stats, lock):
    os.makedirs(INDEX_CACHE_DIR, exist_ok=True)
    st_info = os.stat(path)
    sig = f"{path}|{st_info.st_mtime_ns}|{st_info.st_size}"
    cpath = os.path.join(INDEX_CACHE_DIR, hashlib.md5(sig.encode()).hexdigest() + ".json")
    if os.path.exists(cpath):
        try:
            with open(cpath, "r", encoding="utf-8") as fh:
                pages = [tuple(x) for x in json.load(fh)]
            return split_letters(pages, name, path=path)
        except Exception:
            pass

    local_stats = {"ocr_ok": stats["ocr_ok"], "ocr_done": 0, "unread": 0}
    skip_ocr = ext == "pdf" and not stats.get("allow_ocr", True)
    if ext == "pdf":
        run_stats = {**local_stats, "ocr_ok": False} if skip_ocr else local_stats
        pages = read_pdf_pages(path, run_stats)
        local_stats = run_stats
    else:
        pages = read_other(path)
    with lock:
        stats["ocr_done"] += local_stats["ocr_done"]
        stats["unread"] += local_stats["unread"]
    if not skip_ocr:
        try:
            with open(cpath, "w", encoding="utf-8") as fh:
                json.dump([[pno, txt] for pno, txt in pages], fh)
        except Exception:
            pass
    return split_letters(pages, name, path=path)


@st.cache_resource
def get_bg_state():
    return {"started": False, "running": False, "done": False, "needs_reload": False, "drive": None}


def _bg_sync_worker(folder, sa_info, bg):
    try:
        dv = sync_drive(folder=folder, sa_info=sa_info)
        bg["drive"] = dv
        # Pre-parse & OCR all downloaded files into disk cache in the background
        stats = {"ocr_ok": ocr_available(), "ocr_done": 0, "unread": 0}
        lock = threading.Lock()
        targets = []
        for root in (DOC_DIR, DRIVE_DIR, UPLOAD_DIR):
            if os.path.exists(root):
                for path in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
                    if not os.path.isfile(path) or path.endswith("README.txt") or os.path.basename(path).startswith("_"):
                        continue
                    ext = path.lower().rsplit(".", 1)[-1]
                    if ext in ("pdf", "docx", "txt", "md"):
                        name = os.path.basename(path)
                        if path.startswith(DRIVE_SA):
                            name = name[7:]
                        targets.append((path, name, ext))
        if targets:
            with ThreadPoolExecutor(max_workers=min(4, len(targets))) as pool:
                list(pool.map(lambda t: _parse_file_cached(t[0], t[1], t[2], stats, lock), targets))
        bg["needs_reload"] = True
    except Exception as e:
        bg["drive"] = {"enabled": bool(folder), "files": 0, "error": str(e)[:250], "mode": None}
    finally:
        bg["running"] = False
        bg["done"] = True


def start_bg_sync_if_needed(force=False):
    folder, sa_info = _get_drive_cfg()
    bg = get_bg_state()
    if not folder:
        return bg
    if force:
        clear_drive_sync_cache()
        bg["started"] = False
    if not bg["started"] and not bg["running"]:
        bg["started"] = True
        bg["running"] = True
        bg["done"] = False
        threading.Thread(target=_bg_sync_worker, args=(folder, sa_info, bg), daemon=True).start()
    return bg


@st.cache_resource(show_spinner="Loading available documents…")
def build_index():
    bg = get_bg_state()
    folder, _ = _get_drive_cfg()
    if bg.get("drive"):
        drive = bg["drive"]
    elif not folder:
        drive = {"enabled": False, "files": 0, "error": None, "mode": None}
    else:
        n_drive = sum(
            1 for f in glob.glob(os.path.join(DRIVE_DIR, "**", "*"), recursive=True)
            if os.path.isfile(f) and not os.path.basename(f).startswith("_")
        )
        drive = {"enabled": True, "files": n_drive, "error": None, "mode": "syncing"}
    stats = {"ocr_ok": ocr_available(), "allow_ocr": not bg.get("running"), "ocr_done": 0, "unread": 0}
    letters, files, targets = [], [], []
    paths = []
    for root in (DOC_DIR, DRIVE_DIR, UPLOAD_DIR):
        if os.path.exists(root):
            paths += sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True))
    for path in paths:
        if not os.path.isfile(path) or path.endswith("README.txt") or os.path.basename(path).startswith("_"):
            continue
        ext = path.lower().rsplit(".", 1)[-1]
        if ext not in ("pdf", "docx", "txt", "md"):
            continue
        name = os.path.basename(path)
        if path.startswith(DRIVE_SA):
            name = name[7:]                      # drop the id prefix added when downloading
        files.append({"name": name, "path": path})
        targets.append((path, name, ext))

    lock = threading.Lock()
    if targets:
        workers = min(4, len(targets))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_parse_file_cached, p, n, e, stats, lock) for p, n, e in targets]
            for fut in as_completed(futs):
                try:
                    letters += fut.result()
                except Exception as e:
                    print(f"Could not read file: {e}")
    chunks = chunk_letters(letters)
    vec = matrix = None
    if chunks:
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        matrix = vec.fit_transform([c["search"] for c in chunks])
    return {"letters": letters, "files": files, "chunks": chunks, "vec": vec, "matrix": matrix, "stats": stats, "drive": drive}


def retrieve(query):
    idx = build_index()
    if not idx["chunks"]:
        return []
    try:
        scores = cosine_similarity(idx["vec"].transform([query]), idx["matrix"])[0]
    except Exception:
        return []
    return [idx["chunks"][i] for i in scores.argsort()[::-1][:TOP_K] if scores[i] > MIN_SCORE]


def find_relevant_files(query, chunks=None, limit=5):
    """Return unique matching files from retrieved chunks + filename/metadata matches."""
    idx = build_index()
    seen, matched = set(), []

    def add_file(path, name, no="", date="", subject=""):
        if not path or path in seen or not os.path.exists(path):
            return
        seen.add(path)
        matched.append({"path": path, "name": name, "no": no, "date": date, "subject": subject})

    for c in (chunks or []):
        add_file(c.get("path", ""), c.get("source", ""), c.get("no", ""), c.get("date", ""), c.get("subject", ""))

    tokens = [t for t in re.findall(r"[a-z0-9\u0900-\u097f]{3,}", query.lower())
              if t not in {"the", "and", "for", "with", "from", "this", "that", "file", "files", "letter", "letters", "document", "documents", "find", "search", "show", "give", "send", "uploaded"}]
    if tokens:
        for l in idx["letters"]:
            meta = f"{l['source']} {l['no']} {l['date']} {l['subject']} {l['issuer']}".lower()
            if any(t in meta for t in tokens):
                add_file(l.get("path", ""), l.get("source", ""), l.get("no", ""), l.get("date", ""), l.get("subject", ""))
        for f in idx.get("files", []):
            if any(t in f["name"].lower() for t in tokens):
                add_file(f["path"], f["name"])

    return matched[:limit]


def guess_mime(name):
    ext = (name or "").lower().rsplit(".", 1)[-1]
    return {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt": "text/plain",
        "md": "text/markdown",
        "zip": "application/zip",
    }.get(ext, "application/octet-stream")


def resolve_letter_path(letter, idx_files=None):
    path = letter.get("path", "")
    if path and os.path.exists(path):
        return path
    src_name = letter.get("source", "")
    for f in (idx_files or []):
        if f.get("name") == src_name and f.get("path") and os.path.exists(f["path"]):
            return f["path"]
    return ""


def get_letter_download(letter, idx_files=None):
    """Return (bytes, filename, mime) for a specific letter, extracting its pages if part of a multi-letter PDF."""
    path = resolve_letter_path(letter, idx_files)
    name = letter.get("source", "document") or os.path.basename(path)
    if not path:
        return None, None, None
    if path.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader, PdfWriter
            reader = PdfReader(path)
            total = len(reader.pages)
            ps = max(1, int(letter.get("page_start") or 1))
            pe = min(total, int(letter.get("page_end") or total))
            if ps <= pe and (ps > 1 or pe < total):
                writer = PdfWriter()
                for pg in range(ps - 1, pe):
                    writer.add_page(reader.pages[pg])
                out = io.BytesIO()
                writer.write(out)
                tag = re.sub(r"[^\w.-]+", "_", letter.get("no") or f"p{ps}-{pe}").strip("_")
                dl_name = f"{tag}_{name}" if tag else name
                return out.getvalue(), dl_name, "application/pdf"
        except Exception:
            pass
    with open(path, "rb") as fh:
        return fh.read(), name, guess_mime(name)


def build_zip_bytes(file_items):
    """Package unique existing files or (bytes, name) entries into an in-memory ZIP archive."""
    buf = io.BytesIO()
    seen_paths, seen_names, count = set(), set(), 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in file_items:
            data = item.get("data")
            path = item.get("path", "")
            name = item.get("name", "") or (os.path.basename(path) if path else "document")
            if data is None:
                if not path or path in seen_paths or not os.path.exists(path):
                    continue
                seen_paths.add(path)
            arcname = name
            n = 1
            while arcname in seen_names:
                stem, ext = os.path.splitext(name)
                arcname = f"{stem}_{n}{ext}"
                n += 1
            seen_names.add(arcname)
            if data is not None:
                zf.writestr(arcname, data)
            else:
                zf.write(path, arcname=arcname)
            count += 1
    return buf.getvalue() if count else b""


def label(c):
    return f"Letter No. {c['no'] or 'n/a'} dated {c['date'] or 'n/a'} | Sub: {c['subject'] or 'n/a'} | {c['source']} {c['page']}"


# ---------------- LLMs ----------------
@st.cache_resource
def groq_client(key):
    from groq import Groq
    return Groq(api_key=key)


@st.cache_resource
def gemini_client(key):
    from google import genai
    return genai.Client(api_key=key)


def _groq(system, msgs):
    key = cfg("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing in Secrets")
    client, errs = groq_client(key), []
    for model in dict.fromkeys([GROQ_MODEL] + GROQ_BACKUPS):
        try:
            kw = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
            r = client.chat.completions.create(model=model, temperature=0.3, max_tokens=2000,
                                               messages=[{"role": "system", "content": system}] + msgs, **kw)
            if r.choices[0].message.content:
                return r.choices[0].message.content
            errs.append(f"{model}: empty reply")
        except Exception as e:
            errs.append(f"{model}: {e}")
    raise RuntimeError("; ".join(errs))


def _gemini_contents(msgs):
    from google.genai import types
    return [types.Content(role="user" if m["role"] == "user" else "model",
                          parts=[types.Part(text=m["content"])]) for m in msgs]


def _gemini(system, msgs):
    from google.genai import types
    key = cfg("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing in Secrets")
    client, errs = gemini_client(key), []
    for model in dict.fromkeys([GEMINI_MODEL] + GEMINI_BACKUPS):
        try:
            r = client.models.generate_content(model=model, contents=_gemini_contents(msgs),
                config=types.GenerateContentConfig(system_instruction=system, temperature=0.3, max_output_tokens=2000))
            if r.text:
                return r.text
            errs.append(f"{model}: empty reply")
        except Exception as e:
            errs.append(f"{model}: {e}")
    raise RuntimeError("; ".join(errs))


def generate(system, msgs):
    providers, errors = {"groq": _groq, "gemini": _gemini}, []
    for p in dict.fromkeys([PROVIDER, FALLBACK]):
        if p in providers:
            try:
                return providers[p](system, msgs)
            except Exception as e:
                errors.append(f"{p}: {e}")
    raise RuntimeError(" | ".join(errors) or "No valid LLM provider configured")


# ---------------- internet search fallback ----------------
def web_answer(question, hist):
    """1) Gemini with Google Search grounding  2) DuckDuckGo snippets + LLM. Returns (text, sources)."""
    from google.genai import types
    errs = []
    key = cfg("GEMINI_API_KEY")
    if key:
        client = gemini_client(key)
        for model in dict.fromkeys([GEMINI_MODEL] + GEMINI_BACKUPS):
            try:
                r = client.models.generate_content(
                    model=model, contents=_gemini_contents(hist + [{"role": "user", "content": question}]),
                    config=types.GenerateContentConfig(system_instruction=WEB_SYSTEM, temperature=0.3,
                        max_output_tokens=2000, tools=[types.Tool(google_search=types.GoogleSearch())]))
                if r.text:
                    srcs = []
                    try:
                        for i, ch in enumerate(r.candidates[0].grounding_metadata.grounding_chunks or [], 1):
                            if ch.web:
                                srcs.append(f"[{i}] [{ch.web.title or ch.web.uri}]({ch.web.uri})")
                    except Exception:
                        pass
                    return r.text, srcs
            except Exception as e:
                errs.append(f"gemini-search {model}: {e}")
    try:
        from ddgs import DDGS
        res = DDGS().text(question + " Indian Railways", region="in-en", max_results=6)
        if res:
            ctx = "\n".join(f"[{i}] {x.get('title','')}: {x.get('body','')}" for i, x in enumerate(res, 1))
            text = generate(WEB_SNIPPET_SYSTEM, hist + [{"role": "user", "content": f"SEARCH RESULTS:\n{ctx}\n\nQUESTION: {question}"}])
            return text, [f"[{i}] [{x.get('title','link')}]({x.get('href','')})" for i, x in enumerate(res, 1)]
    except Exception as e:
        errs.append(f"ddgs: {e}")
    raise RuntimeError("; ".join(errs) or "web search unavailable")


# ---------------- NEWS ----------------
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NEWS_TOPICS = {
    "📍 Solapur Division": '("Solapur division" OR "Solapur railway" OR "Solapur railway station" OR "Solapur Central Railway")',
    "🚆 Central Railway": '"Central Railway" (Mumbai OR Pune OR Solapur OR Nagpur OR Bhusawal)',
    "🇮🇳 Indian Railways": '"Indian Railways"',
    "🛡️ Railway Safety": '("railway safety" OR "train accident" OR derailment OR Kavach) India',
}
NEWS_LANGS = {"English": ("en-IN", "IN:en"), "हिंदी": ("hi", "IN:hi"), "मराठी": ("mr", "IN:mr")}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(query, days=None, after=None, before=None, lang="English", limit=30):
    q = query + (f" when:{days}d" if days else "") + (f" after:{after}" if after else "") + (f" before:{before}" if before else "")
    hl, ceid = NEWS_LANGS.get(lang, NEWS_LANGS["English"])
    r = requests.get("https://news.google.com/rss/search?" + urlencode({"q": q, "hl": hl, "gl": "IN", "ceid": ceid}),
                     timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    items, seen = [], set()
    for it in ET.fromstring(r.content).iter("item"):
        title, link = it.findtext("title", ""), it.findtext("link", "")
        se = it.find("source")
        src = se.text if se is not None and se.text else ""
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        key = title.lower()[:70]
        if not title or key in seen:
            continue
        seen.add(key)
        try:
            when = parsedate_to_datetime(it.findtext("pubDate", "")).astimezone(IST)
        except Exception:
            when = None
        items.append({"title": title, "link": link, "source": src, "when": when})
    items.sort(key=lambda x: x["when"] or dt.datetime.min.replace(tzinfo=IST), reverse=True)
    return items[:limit]


def fmt_when(d):
    return d.strftime("%d %b %Y") if d else ""


def news_cards(items):
    for n in items:
        st.markdown(f'<div class="newscard"><a href="{n["link"]}" target="_blank">{n["title"]}</a>'
                    f'<div class="newsmeta">📰 {n["source"] or "news"} · {fmt_when(n["when"])}</div></div>', unsafe_allow_html=True)


NEWS_WORDS = re.compile(r"\b(news|latest|today|recent|recently|headlines?|happening|past news|old news|previous news)\b|"
                        r"खबर|समाचार|ताज़ा|ताजा|बातम्या|बातमी", re.I)


def pick_topic(q):
    ql = q.lower()
    if "solapur" in ql or "सोलापुर" in q or "सोलापूर" in q:
        return "📍 Solapur Division"
    if re.search(r"safety|accident|derail|kavach|collision|spad", ql):
        return "🛡️ Railway Safety"
    if "central railway" in ql or "मध्य रेल" in q:
        return "🚆 Central Railway"
    return "🇮🇳 Indian Railways"


def pick_days(q):
    ql = q.lower()
    if re.search(r"today|आज|24 hour", ql):
        return 1
    if re.search(r"year|past|old|previous|earlier|history|पिछले|मागील", ql):
        return 365
    if re.search(r"month|महीन|महिन", ql):
        return 30
    return 7


GENERAL_CHAT_RE = re.compile(
    r"\b(draft|write|compose|rewrite|translate|explain|summarize|idea|ideas|slogan|poem|story|essay|"
    r"email|message|speech|welcome|congratulat|advice|suggest|help me|how are you|tell me about|"
    r"what do you think|joke|grammar|correct this|improve this)\b",
    re.I,
)
DOC_INTENT_RE = re.compile(
    r"\b(letter|letters|circular|order|notification|document|documents|file|files|pdf|docx|upload|uploaded|"
    r"no\.?|dated|spad|kavach|g&sr|rule|rules|manual|sop|policy|guideline|directive|board|division|"
    r"which letter|find|search|show me)\b|पत्र|परिपत्रक|संरक्षा|नियम",
    re.I,
)


# ---------------- answer pipeline ----------------
def answer(question, history, chat_mode="Auto (Letters + Chat + Web)"):
    """Returns (text, sources, mode, files); mode in news | doc | web | chat."""
    hist = [{"role": m["role"], "content": m["content"]} for m in history[-8:]]

    if chat_mode == "💬 General Chat only":
        return generate(CHAT_SYSTEM, hist + [{"role": "user", "content": question}]), [], "chat", []

    if NEWS_WORDS.search(question) and chat_mode != "📚 Letters only":
        topic, days = pick_topic(question), pick_days(question)
        try:
            items = fetch_news(NEWS_TOPICS[topic], days=days)[:12] or fetch_news(NEWS_TOPICS[topic], days=365)[:12]
        except Exception as e:
            return f"⚠️ Couldn't fetch news right now ({e}). Try again in a bit.", [], "chat", []
        if items:
            ctx = "\n".join(f"[{i}] {n['title']} ({n['source']}, {fmt_when(n['when'])})" for i, n in enumerate(items, 1))
            text = generate(NEWS_SYSTEM, hist + [{"role": "user", "content": f"NEWS HEADLINES ({topic}):\n{ctx}\n\nQUESTION: {question}"}])
            return text, [f"[{i}] [{n['title']}]({n['link']}) - {n['source']}, {fmt_when(n['when'])}" for i, n in enumerate(items, 1)], "news", []

    if chat_mode != "📚 Letters only" and (
        SMALLTALK.match(question) or (GENERAL_CHAT_RE.search(question) and not DOC_INTENT_RE.search(question))
    ):
        return generate(CHAT_SYSTEM, hist + [{"role": "user", "content": question}]), [], "chat", []

    ctx = retrieve(question)                                   # 1) department & uploaded letters
    matched_files = find_relevant_files(question, ctx)
    if ctx:
        context = "\n\n".join(f"[{i}] {label(c)}\n{c['text']}" for i, c in enumerate(ctx, 1))
        text = generate(RAG_SYSTEM, hist + [{"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}])
        if "NOT_FOUND_IN_DOCUMENTS" not in text:
            return text, [f"[{i}] {label(c)}" for i, c in enumerate(ctx, 1)], "doc", matched_files

    if matched_files and (DOC_INTENT_RE.search(question) or chat_mode == "📚 Letters only"):
        file_list = "\n".join(f"- **{f['name']}**" + (f" (Letter No. {f['no']}, {f['date']})" if f.get("no") else "") for f in matched_files)
        return f"📑 I found relevant document(s) matching your query:\n\n{file_list}\n\nDownload them below or ask a specific question about their contents.", [], "doc", matched_files

    if chat_mode == "📚 Letters only":
        return "📭 Not found in your department or uploaded letters. Switch to **Auto** or **General Chat** to ask freely or search the web.", [], "doc", []

    if DOC_INTENT_RE.search(question):
        try:                                                    # 2) internet for rule/policy lookups
            text, srcs = web_answer(question, hist)
            return text, srcs, "web", []
        except Exception:
            pass

    return generate(CHAT_SYSTEM, hist + [{"role": "user", "content": question}]), [], "chat", []


def render_msg(m, msg_idx=0):
    if m["role"] == "assistant" and m.get("mode"):
        lab = {"doc": "📚 from your letters", "news": "📰 live news", "web": "🌐 from the internet"}.get(m["mode"], "💬 general chat")
        cls = {"doc": "doc", "news": "doc", "web": "web"}.get(m["mode"], "chat")
        st.markdown(f'<span class="badge {cls}">{lab}</span>', unsafe_allow_html=True)
    st.markdown(m["content"])
    if m.get("files"):
        st.markdown("**📎 Relevant file(s):**")
        cols = st.columns(min(len(m["files"]), 3))
        for j, f in enumerate(m["files"]):
            if os.path.exists(f["path"]):
                with open(f["path"], "rb") as fh:
                    cols[j % len(cols)].download_button(
                        label=f"⬇️ {f['name'][:28]}",
                        data=fh.read(),
                        file_name=f["name"],
                        key=f"dl_{msg_idx}_{j}_{f['name']}",
                    )
    if m.get("sources"):
        with st.expander("Sources"):
            st.markdown("\n\n".join(m["sources"]))


# ---------------- document viewer helper ----------------
def render_document_viewer(letter, key_suffix="", idx_files=None):
    path = resolve_letter_path(letter, idx_files)
    name = letter.get("source", "document")
    if not path:
        st.info("Original file is not available on disk.")
        return

    dl_bytes, dl_name, dl_mime = get_letter_download(letter, idx_files)
    if dl_bytes:
        st.download_button(
            f"⬇️ Download {dl_name}",
            data=dl_bytes,
            file_name=dl_name,
            mime=dl_mime,
            key=f"view_dl_{key_suffix or letter['id']}",
        )

    if path.lower().endswith(".pdf"):
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(path)
            total_pages = len(pdf)
            default_p = min(max(int(letter.get("page_start", 1)), 1), total_pages)
            pno = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=default_p,
                step=1,
                key=f"pg_{letter['id']}_{key_suffix}",
            )
            img = pdf[pno - 1].render(scale=2).to_pil()
            st.image(img, caption=f"{name} — Page {pno} of {total_pages}", use_container_width=True)
            return
        except Exception:
            pass
    st.text(letter["text"][:10000])


# ---------------- pages ----------------
def chat_page():
    st.markdown('<div class="hero"><h1>Safety Branch AI</h1><p>Letters, general chat & web · हिंदी / English / मराठी</p></div>', unsafe_allow_html=True)
    chat_mode = st.radio(
        "Chat mode",
        ["Auto (Letters + Chat + Web)", "💬 General Chat only", "📚 Letters only"],
        horizontal=True,
        label_visibility="collapsed",
        key="chat_mode",
    )
    st.session_state.setdefault("history", [])
    if not st.session_state.history:
        st.markdown("**Try one:**")
        cols = st.columns(2)
        for i, sg in enumerate(SUGGESTIONS):
            if cols[i % 2].button(sg, key=f"sug{i}"):
                st.session_state.pending = sg.split(" ", 1)[1]
                st.rerun()
    for idx_m, m in enumerate(st.session_state.history):
        with st.chat_message(m["role"]):
            render_msg(m, msg_idx=idx_m)
    q = st.chat_input("Ask a question… / कुछ पूछें…") or st.session_state.pop("pending", None)
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Checking your letters…"):
                try:
                    a, srcs, mode, files = answer(q, st.session_state.history, chat_mode=chat_mode)
                except Exception as e:
                    a, srcs, mode, files = f"⚠️ Error: {e}\n\nIf it names a provider, check that key/model name in Secrets.", [], None, []
            msg = {"role": "assistant", "content": a, "sources": srcs, "mode": mode, "files": files}
            render_msg(msg, msg_idx=len(st.session_state.history))
        st.session_state.history += [{"role": "user", "content": q}, msg]


def letters_page():
    import pandas as pd
    st.markdown('<div class="hero"><h1>Letter Register</h1><p>Every letter found in your documents</p></div>', unsafe_allow_html=True)
    idx = build_index()
    dv, stt = idx["drive"], idx["stats"]

    c_find, c_ref = st.columns([4, 1])
    find = c_find.text_input("Filter (number, subject, date, office)", placeholder="e.g. SPAD, 2024, Kavach", label_visibility="collapsed")
    if c_ref.button("🔄 Refresh", key="refresh_letters"):
        if dv.get("mode") == "public":
            shutil.rmtree(DRIVE_PUB, ignore_errors=True)
        start_bg_sync_if_needed(force=True)
        build_index.clear()
        st.rerun()

    L = sorted(idx["letters"], key=lambda l: l["date"] or "", reverse=True)
    if not L:
        st.warning("No letters found yet. Here's why:")
        if not dv["enabled"]:
            st.write("☁️ **Drive:** not connected - `GDRIVE_FOLDER_ID` is missing in Secrets (or the app hasn't reloaded since you added it).")
        elif dv["error"]:
            st.write(f"☁️ **Drive:** sync failed - {dv['error']}")
        else:
            st.write(f"☁️ **Drive ({dv.get('mode')}):** {dv['files']} file(s) downloaded.")
        st.write(f"📄 **Scanned/unreadable pages:** {stt['unread']} · **OCR available:** {stt['ocr_ok']}")
        if dv["enabled"] and not dv["error"] and dv["files"] and stt["unread"]:
            st.info("Files were downloaded but their pages are images and OCR isn't working. Make sure packages.txt is in the repo (top level) and redeploy, or upload text-searchable PDFs.")
        try:
            names = list(st.secrets.keys())
        except Exception:
            names = []
        st.write("🔑 **Secret names found** (values hidden):", names or "none")
        return

    rows = [{"Letter No": l["no"] or "—", "Date": l["date"] or "—", "Subject": l["subject"] or "(not detected)",
             "Issued by": l["issuer"] or "—", "Pages": f"{l['page_start']}-{l['page_end']}", "File": l["source"]} for l in L]
    df = pd.DataFrame(rows)
    if find:
        df = df[df.apply(lambda r: find.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
    st.caption(f"{len(df)} of {len(L)} letters")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### 📑 View, Download, Summarize & Compare")
    opts = {f"{l['no'] or 'n/a'} | {l['date'] or 'n/a'} | {(l['subject'] or l['text'][:60])[:70]}": l for l in L}
    opt_keys = list(opts.keys())
    picks = st.multiselect(
        "Select one or more letters / documents",
        opt_keys,
        default=[opt_keys[0]] if opt_keys else [],
        help="Select multiple documents to view them together, download them, or compare them with AI.",
    )
    selected = [opts[k] for k in picks]

    # ---- Download prompt: current/selected vs all documents ----
    dl_choice = st.radio(
        "What would you like to download?",
        ["Current / Selected document(s)", "All documents (ZIP)"],
        horizontal=True,
        key="dl_scope",
    )
    idx_files = idx.get("files", [])
    if dl_choice == "All documents (ZIP)":
        all_files = idx_files or [{"name": l["source"], "path": resolve_letter_path(l, idx_files)} for l in L]
        zip_data = build_zip_bytes(all_files)
        if zip_data:
            st.download_button(
                "📦 Download all documents (.zip)",
                data=zip_data,
                file_name="safety_branch_all_documents.zip",
                mime="application/zip",
                key="dl_all_zip",
            )
        else:
            st.info("No downloadable files found on disk.")
    else:
        if not selected:
            st.info("Select at least one document above to download.")
        elif len(selected) == 1:
            dl_bytes, dl_name, dl_mime = get_letter_download(selected[0], idx_files)
            if dl_bytes:
                st.download_button(
                    f"⬇️ Download current document ({dl_name})",
                    data=dl_bytes,
                    file_name=dl_name,
                    mime=dl_mime,
                    key="dl_curr_btn",
                )
            else:
                st.info("Original file is not available on disk.")
        else:
            sel_entries = []
            for s in selected:
                b, n, m = get_letter_download(s, idx_files)
                if b:
                    sel_entries.append({"name": n, "data": b, "mime": m})
            sel_zip = build_zip_bytes(sel_entries)
            if sel_zip:
                st.download_button(
                    f"📦 Download {len(sel_entries)} selected documents (.zip)",
                    data=sel_zip,
                    file_name="selected_documents.zip",
                    mime="application/zip",
                    key="dl_sel_zip",
                )
                cols_dl = st.columns(min(len(sel_entries), 3))
                for j, entry in enumerate(sel_entries):
                    cols_dl[j % len(cols_dl)].download_button(
                        f"⬇️ {entry['name'][:28]}",
                        data=entry["data"],
                        file_name=entry["name"],
                        mime=entry["mime"],
                        key=f"dl_sel_item_{j}",
                    )
            else:
                st.info("Selected files are not available on disk.")

    if not selected:
        return

    # ---- Multi-document viewer ----
    with st.expander(f"📄 View selected document(s) ({len(selected)})", expanded=False):
        for i, doc_item in enumerate(selected):
            st.markdown(f"**{i + 1}. {doc_item['source']}** — Letter No. `{doc_item['no'] or 'n/a'}` ({doc_item['date'] or 'n/a'})")
            render_document_viewer(doc_item, key_suffix=f"multi_{i}", idx_files=idx_files)
            if i < len(selected) - 1:
                st.markdown("<hr style='border:0;border-top:1px solid #ffffff22;margin:12px 0;'>", unsafe_allow_html=True)

    # ---- AI Summary & Compare ----
    lang = st.radio("AI output language", ["English", "हिंदी", "मराठी"], horizontal=True)
    c_sum, c_cmp = st.columns(2)

    if c_sum.button("✨ Summarize & recommend"):
        target = selected[0]
        body = " ".join(target["text"].split()[:5000])
        try:
            with st.spinner("Reading the letter…"):
                summary = generate(LETTER_SYSTEM, [{"role": "user", "content":
                    f"Write the summary in {lang}.\n\nLETTER (file {target['source']}, pages {target['page_start']}-{target['page_end']}):\n{body}"}])
                st.session_state["letter_summary"] = {"id": target["id"], "lang": lang, "text": summary, "letter": target}
        except Exception as e:
            st.error(f"Error: {e}")

    if c_cmp.button("⚖️ Compare selected (AI)", disabled=len(selected) < 2):
        per_doc_words = max(1200, 9000 // len(selected))
        docs_ctx = []
        for idx_s, s in enumerate(selected, 1):
            snippet = " ".join(s["text"].split()[:per_doc_words])
            docs_ctx.append(
                f"=== DOCUMENT {idx_s} ===\n"
                f"File: {s['source']} (pages {s['page_start']}-{s['page_end']})\n"
                f"Letter No.: {s['no'] or 'n/a'} | Date: {s['date'] or 'n/a'} | Issued by: {s['issuer'] or 'n/a'}\n"
                f"Subject: {s['subject'] or 'n/a'}\n\n{snippet}"
            )
        try:
            with st.spinner("Comparing selected documents…"):
                cmp_res = generate(COMPARE_SYSTEM, [{"role": "user", "content":
                    f"Write the comparison in {lang}.\n\n" + "\n\n".join(docs_ctx)}])
                st.session_state["letter_compare"] = {
                    "ids": tuple(s["id"] for s in selected),
                    "lang": lang,
                    "text": cmp_res,
                    "names": [s["source"] for s in selected],
                }
        except Exception as e:
            st.error(f"Error: {e}")

    if len(selected) < 2:
        st.caption("Tip: Select 2 or more documents above to enable AI comparison.")

    # ---- Display persisted Summary + WhatsApp share ----
    saved = st.session_state.get("letter_summary")
    if saved and saved.get("id") == selected[0]["id"]:
        st.markdown("#### 📝 Letter Summary")
        st.markdown(saved["text"])
        sl = saved.get("letter", selected[0])
        wa_msg = (
            f"*Safety Branch AI - Letter Summary*\n"
            f"Letter No: {sl['no'] or 'N/A'} | Date: {sl['date'] or 'N/A'}\n"
            f"Subject: {sl['subject'] or 'N/A'}\n\n"
            f"{saved['text']}"
        )
        st.link_button("💬 Share summary via WhatsApp", f"https://wa.me/?text={quote_plus(wa_msg)}")

    # ---- Display persisted Comparison + WhatsApp share ----
    saved_cmp = st.session_state.get("letter_compare")
    if saved_cmp and saved_cmp.get("ids") == tuple(s["id"] for s in selected):
        st.markdown("#### ⚖️ AI Document Comparison")
        st.markdown(saved_cmp["text"])
        wa_cmp = (
            f"*Safety Branch AI - Document Comparison*\n"
            f"Compared: {', '.join(saved_cmp['names'])}\n\n"
            f"{saved_cmp['text']}"
        )
        st.link_button("💬 Share comparison via WhatsApp", f"https://wa.me/?text={quote_plus(wa_cmp)}")

    with st.expander("View extracted text"):
        for idx_t, doc_item in enumerate(selected):
            st.markdown(f"**{idx_t + 1}. {doc_item['source']} (Letter No. {doc_item['no'] or 'n/a'})**")
            st.text(doc_item["text"][:4000])


def news_page():
    st.markdown('<div class="hero"><h1>Rail News</h1><p>Live + past headlines · Solapur Division first</p></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    topic = c1.selectbox("Topic", list(NEWS_TOPICS.keys()), index=0)
    lang = c2.selectbox("Language", list(NEWS_LANGS.keys()), index=0)
    rng = st.radio("Time range", ["24 hours", "7 days", "30 days", "1 year", "Custom dates"], index=1, horizontal=True)
    days = after = before = None
    if rng == "Custom dates":
        d1, d2 = st.columns(2)
        after = d1.date_input("From", dt.date.today() - dt.timedelta(days=180)).isoformat()
        before = d2.date_input("To", dt.date.today()).isoformat()
    else:
        days = {"24 hours": 1, "7 days": 7, "30 days": 30, "1 year": 365}[rng]
    if topic != "📍 Solapur Division":
        try:
            sp = fetch_news(NEWS_TOPICS["📍 Solapur Division"], days=30, lang=lang)[:4]
        except Exception:
            sp = []
        if sp:
            st.markdown('<div class="spot"><b>📍 Solapur Division spotlight</b> · last 30 days</div>', unsafe_allow_html=True)
            news_cards(sp)
    try:
        with st.spinner("Fetching headlines…"):
            items = fetch_news(NEWS_TOPICS[topic], days=days, after=after, before=before, lang=lang)
    except Exception as e:
        st.error(f"Couldn't fetch news: {e}")
        return
    st.markdown(f"**{topic}** · {len(items)} headlines")
    if not items:
        st.info("No headlines for this range. Try a longer range or another language.")
        return
    if st.button("✨ Summarize these headlines"):
        ctx = "\n".join(f"[{i}] {n['title']} ({n['source']}, {fmt_when(n['when'])})" for i, n in enumerate(items[:15], 1))
        try:
            with st.spinner("Summarizing…"):
                st.markdown(generate(NEWS_SYSTEM, [{"role": "user", "content": f"NEWS HEADLINES ({topic}):\n{ctx}\n\nSummarize the key developments."}]))
        except Exception as e:
            st.error(f"Error: {e}")
    news_cards(items)
    st.caption("Headlines via Google News. Open the link for the full story. News is not official instructions.")


# ---------------- main ----------------
if check_password():
    with st.sidebar:
        st.markdown("### 🚆 Safety Branch AI")
        uploaded = st.file_uploader("📤 Upload documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
        if uploaded:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            changed = False
            for uf in uploaded:
                dest = os.path.join(UPLOAD_DIR, uf.name)
                data = uf.getvalue()
                if not os.path.exists(dest) or os.path.getsize(dest) != len(data):
                    with open(dest, "wb") as fh:
                        fh.write(data)
                    changed = True
            if changed:
                build_index.clear()
                st.rerun()

    bg = start_bg_sync_if_needed()
    if bg.get("needs_reload"):
        bg["needs_reload"] = False
        build_index.clear()
    idx = build_index()
    stt = idx["stats"]
    with st.sidebar:
        if idx["letters"]:
            st.markdown(f'<span class="pill ok">📑 {len(idx["letters"])} letters · {len(idx["chunks"])} passages</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill warn">📭 no documents yet · chat, news & web mode</span>', unsafe_allow_html=True)
        if stt["unread"]:
            st.markdown(f'<span class="pill warn">⚠️ {stt["unread"]} scanned page(s) unreadable</span>', unsafe_allow_html=True)
            st.caption("Some pages are images and OCR isn't available. Add packages.txt (included) and redeploy, or upload a text-searchable PDF.")
        elif stt["ocr_done"]:
            st.markdown(f'<span class="pill ok">🔍 OCR read {stt["ocr_done"]} scanned page(s)</span>', unsafe_allow_html=True)
        dv = idx["drive"]
        if dv["enabled"] and dv["error"]:
            st.markdown('<span class="pill warn">☁️ Drive sync failed</span>', unsafe_allow_html=True)
            st.caption(dv["error"])
        elif bg.get("running"):
            st.markdown('<span class="pill warn">☁️ Syncing Drive in background…</span>', unsafe_allow_html=True)
        elif dv["enabled"]:
            kind = "public link" if dv.get("mode") == "public" else "private"
            st.markdown(f'<span class="pill ok">☁️ Drive ({kind}): {dv["files"]} file(s)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill warn">☁️ Drive not connected</span>', unsafe_allow_html=True)
            st.caption("GDRIVE_FOLDER_ID not found in Secrets. Add it, save, then press Sync / reload.")
        st.markdown(f'<span class="pill">⚡ {PROVIDER} → {FALLBACK or "none"}</span>', unsafe_allow_html=True)
        st.write("")
        if st.button("🔄 Sync / reload documents"):
            if dv.get("mode") == "public":
                shutil.rmtree(DRIVE_PUB, ignore_errors=True)   # force a fresh download
            start_bg_sync_if_needed(force=True)
            build_index.clear()
            st.rerun()
        with st.expander("🔧 Diagnostics"):
            try:
                names = list(st.secrets.keys())
            except Exception:
                names = []
            st.write("Secret names found (values hidden):", names or "none")
            st.write("GDRIVE_FOLDER_ID set:", bool(cfg("GDRIVE_FOLDER_ID")))
            st.write("Drive mode:", dv.get("mode") or "off")
            st.write("Files in documents/ folder:", len([f for f in glob.glob(os.path.join(DOC_DIR, "**", "*"), recursive=True) if os.path.isfile(f)]))
            st.write("Uploaded files:", len([f for f in glob.glob(os.path.join(UPLOAD_DIR, "*")) if os.path.isfile(f)]))
            st.write("OCR available:", stt["ocr_ok"])
        if st.button("🧹 Clear chat"):
            st.session_state.history = []
            st.rerun()
    page = st.radio("page", ["💬 Chat", "📑 Letters", "📰 News"], horizontal=True, label_visibility="collapsed")
    {"💬 Chat": chat_page, "📑 Letters": letters_page, "📰 News": news_page}[page]()
    st.markdown('<div class="foot">AI-generated. Verify with the latest official instructions before acting.</div>', unsafe_allow_html=True)
