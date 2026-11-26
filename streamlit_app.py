"""
Safety Branch AI Assistant - Streamlit (Groq + Gemini fallback)
- Chats normally (hi/hello/anything) even with NO documents.
- When documents are loaded, answers from them with citations.
"""
import os, re, glob
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Safety Branch AI", page_icon="🚆", layout="centered")

DOC_DIR = os.path.join(os.path.dirname(__file__), "documents")
CHUNK_WORDS, OVERLAP, TOP_K, MIN_SCORE = 220, 40, 6, 0.05


def cfg(key, default=""):
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


PROVIDER = cfg("LLM_PROVIDER", "groq").lower()
FALLBACK = cfg("LLM_FALLBACK", "gemini").lower()
GROQ_MODEL = cfg("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = cfg("GEMINI_MODEL", "gemini-2.5-flash")

# ---------------- prompts ----------------
RAG_SYSTEM = """You are the Safety Branch Assistant for Indian Railways officials and staff.
Rules:
1. Answer ONLY from the numbered CONTEXT passages. Never invent rules, limits, rule numbers or dates.
2. Cite every fact like [1], [2] matching the passage numbers.
3. If the context doesn't contain the answer, say so and suggest which manual/circular to check.
4. Reply in the same language as the user (Hindi, English or Hinglish).
5. Be concise and practical. Short bullets for procedures.
6. For accident/incident questions, remind that the latest Board/Zonal instructions prevail."""

CHAT_SYSTEM = """You are the Safety Branch Assistant, a friendly AI for Indian Railways Safety Branch staff.
Talk naturally: greet back, small talk, general questions, explanations, drafting help are all fine.
Style: warm, quick, a bit casual (light emoji ok, not spammy), short replies unless detail is needed.
Reply in the user's language (Hindi, English or Hinglish).
IMPORTANT: no official documents matched this question. For specific railway rules, rule numbers,
limits, SOP steps or dates, do NOT invent them - give general guidance only, say clearly it is not
from the department's documents, and tell them to verify with the official manual/circular.
Never fake citations."""

SUGGESTIONS = ["👋 Hi, what can you do?", "🚨 What to do after a derailment?",
               "📝 Help me draft a safety circular", "🔍 Explain SPAD in simple words"]

# ---------------- styling ----------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Space Grotesk', sans-serif !important; }
.stApp {
  background: radial-gradient(1200px 600px at 10% -10%, #3b1d8f55, transparent),
              radial-gradient(900px 500px at 100% 0%, #ff2e9a33, transparent),
              radial-gradient(800px 500px at 50% 120%, #00e5ff22, transparent), #0a0b14;
  color: #eceefc;
}
header[data-testid="stHeader"], footer, #MainMenu {visibility: hidden; height: 0;}
.block-container {padding-top: 1.5rem; max-width: 820px;}
.hero {text-align:center; padding: 8px 0 14px;}
.hero h1 {font-size: 2.3rem; font-weight: 700; margin: 0;
  background: linear-gradient(90deg,#b6ff3c,#00e5ff,#ff2e9a); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;}
.hero p {color:#9aa0c3; margin: 4px 0 0; font-size: .95rem;}
.pill {display:inline-block; padding: 4px 12px; border-radius: 999px; font-size:.75rem;
  border:1px solid #ffffff22; background:#ffffff0d; backdrop-filter: blur(8px); margin: 4px 4px 0;}
.pill.ok {color:#b6ff3c; border-color:#b6ff3c55;} .pill.warn {color:#ffd23c; border-color:#ffd23c55;}
[data-testid="stChatMessage"] {
  background: #ffffff0a; border: 1px solid #ffffff14; border-radius: 18px;
  backdrop-filter: blur(10px); padding: 12px 16px; margin-bottom: 10px;
  box-shadow: 0 6px 30px #00000040;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: linear-gradient(135deg,#6d3bff44,#ff2e9a33); border-color:#ff2e9a44;}
.badge {font-size:.7rem; padding:2px 9px; border-radius:999px; margin-bottom:6px; display:inline-block;
  border:1px solid #ffffff22;}
.badge.doc {color:#b6ff3c; background:#b6ff3c14;} .badge.chat {color:#00e5ff; background:#00e5ff14;}
[data-testid="stChatInput"] {background:#ffffff0d; border:1px solid #ffffff22; border-radius: 16px;}
[data-testid="stChatInput"] textarea {color:#fff !important;}
.stButton>button {background:#ffffff0d; color:#eceefc; border:1px solid #ffffff22; border-radius:14px;
  padding:.55rem .8rem; font-weight:500; transition: all .2s; width:100%;}
.stButton>button:hover {border-color:#b6ff3c; color:#b6ff3c; transform: translateY(-2px);
  box-shadow: 0 6px 20px #b6ff3c22;}
[data-testid="stSidebar"] {background:#0d0e1acc; border-right:1px solid #ffffff14;}
.streamlit-expanderHeader, [data-testid="stExpander"] {border-radius: 12px; border-color:#ffffff1a;}
.foot {text-align:center; color:#6b7094; font-size:.72rem; margin-top: 10px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------- password gate ----------------
def check_password():
    pw = cfg("APP_PASSWORD")
    if not pw or st.session_state.get("auth"):
        return True
    st.markdown('<div class="hero"><h1>🚆 Safety Branch AI</h1><p>locked in 🔒 enter the password</p></div>',
                unsafe_allow_html=True)
    entered = st.text_input("Password", type="password", label_visibility="collapsed",
                            placeholder="password…")
    if st.button("Let me in 🚀"):
        if entered == pw:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Nope, wrong password")
    return False


# ---------------- documents ----------------
def read_file(path):
    ext = path.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        from pypdf import PdfReader
        for i, p in enumerate(PdfReader(path).pages, 1):
            yield f"p.{i}", p.extract_text() or ""
    elif ext == "docx":
        import docx
        yield "", "\n".join(p.text for p in docx.Document(path).paragraphs)
    elif ext in ("txt", "md"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            yield "", f.read()


@st.cache_resource(show_spinner="Indexing documents…")
def build_index():
    chunks = []
    for path in glob.glob(os.path.join(DOC_DIR, "**", "*"), recursive=True):
        if not os.path.isfile(path) or path.endswith("README.txt"):
            continue
        name = os.path.basename(path)
        try:
            for page, text in read_file(path):
                words = re.sub(r"\s+", " ", text).split()
                for i in range(0, max(len(words), 1), CHUNK_WORDS - OVERLAP):
                    piece = " ".join(words[i:i + CHUNK_WORDS])
                    if len(piece) > 40:
                        chunks.append({"source": name, "page": page, "text": piece})
        except Exception as e:
            print(f"Could not read {name}: {e}")
    if not chunks:
        return [], None, None
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    return chunks, vec, vec.fit_transform([c["text"] for c in chunks])


def retrieve(query):
    chunks, vec, matrix = build_index()
    if not chunks:
        return []
    try:
        scores = cosine_similarity(vec.transform([query]), matrix)[0]
    except Exception:
        return []
    return [chunks[i] for i in scores.argsort()[::-1][:TOP_K] if scores[i] > MIN_SCORE]


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
    client = groq_client(key)
    r = client.chat.completions.create(
        model=GROQ_MODEL, temperature=0.4, max_tokens=1000,
        messages=[{"role": "system", "content": system}] + msgs)
    return r.choices[0].message.content


def _gemini(system, msgs):
    from google.genai import types
    key = cfg("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing in Secrets")
    client = gemini_client(key)
    contents = [types.Content(role="user" if m["role"] == "user" else "model",
                              parts=[types.Part(text=m["content"])]) for m in msgs]
    r = client.models.generate_content(
        model=GEMINI_MODEL, contents=contents,
        config=types.GenerateContentConfig(system_instruction=system,
                                           temperature=0.4, max_output_tokens=1000))
    return r.text


def generate(system, msgs):
    providers = {"groq": _groq, "gemini": _gemini}
    errors = []
    for p in dict.fromkeys([PROVIDER, FALLBACK]):
        if p not in providers:
            continue
        try:
            return providers[p](system, msgs)
        except Exception as e:
            errors.append(f"{p}: {e}")
    raise RuntimeError(" | ".join(errors) or "No valid LLM provider configured")


def answer(question, history):
    """Returns (text, sources, mode) where mode is 'doc' or 'chat'."""
    hist = [{"role": m["role"], "content": m["content"]} for m in history[-8:]]
    ctx = retrieve(question)
    if ctx:
        context = "\n\n".join(f"[{i}] ({c['source']} {c['page']})\n{c['text']}"
                              for i, c in enumerate(ctx, 1))
        msgs = hist + [{"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}]
        srcs = [f"[{i}] {c['source']} {c['page']}".strip() for i, c in enumerate(ctx, 1)]
        return generate(RAG_SYSTEM, msgs), srcs, "doc"
    return generate(CHAT_SYSTEM, hist + [{"role": "user", "content": question}]), [], "chat"


def render_msg(m):
    if m["role"] == "assistant" and m.get("mode"):
        label = "📚 from documents" if m["mode"] == "doc" else "💬 general chat"
        cls = "doc" if m["mode"] == "doc" else "chat"
        st.markdown(f'<span class="badge {cls}">{label}</span>', unsafe_allow_html=True)
    st.markdown(m["content"])
    if m.get("sources"):
        with st.expander("Sources"):
            st.write("\n".join(m["sources"]))


# ---------------- UI ----------------
if check_password():
    chunks, _, _ = build_index()
    n_files = len({c["source"] for c in chunks})

    with st.sidebar:
        st.markdown("### 🚆 Safety Branch AI")
        if chunks:
            st.markdown(f'<span class="pill ok">📚 {n_files} doc(s) · {len(chunks)} passages</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill warn">📭 no documents yet · chat mode</span>',
                        unsafe_allow_html=True)
        st.markdown(f'<span class="pill">⚡ {PROVIDER} → {FALLBACK or "none"}</span>',
                    unsafe_allow_html=True)
        st.write("")
        if st.button("🧹 Clear chat"):
            st.session_state.history = []
            st.rerun()
        st.caption("Add PDFs/DOCX/TXT to the documents/ folder on GitHub to unlock cited answers.")

    st.markdown('<div class="hero"><h1>Safety Branch AI</h1>'
                '<p>ask anything · rules, SOPs, drafts, or just vibes · हिंदी / English</p></div>',
                unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    if not st.session_state.history:
        st.markdown("**Try one:**")
        cols = st.columns(2)
        for i, s in enumerate(SUGGESTIONS):
            if cols[i % 2].button(s, key=f"sug{i}"):
                st.session_state.pending = s.split(" ", 1)[1]
                st.rerun()

    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            render_msg(m)

    q = st.chat_input("say something… / कुछ पूछो…") or st.session_state.pop("pending", None)
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("thinking…"):
                try:
                    a, srcs, mode = answer(q, st.session_state.history)
                except Exception as e:
                    a, srcs, mode = f"⚠️ Error: {e}\n\nIf it names a provider, check that key/model name in Secrets.", [], None
            msg = {"role": "assistant", "content": a, "sources": srcs, "mode": mode}
            render_msg(msg)
        st.session_state.history += [{"role": "user", "content": q}, msg]

    st.markdown('<div class="foot">AI-generated. Verify with the latest official instructions before acting.</div>',
                unsafe_allow_html=True)
