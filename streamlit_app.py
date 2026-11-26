"""
Safety Branch AI Assistant - Streamlit (Groq + Gemini fallback)
- Chats normally (hi/hello/anything) even with NO documents.
- When documents are loaded, answers from them with citations.
"""
import os, re, glob, datetime as dt
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from email.utils import parsedate_to_datetime
import requests
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
GROQ_MODEL = cfg("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = cfg("GEMINI_MODEL", "gemini-3.8-flash")
# backups tried automatically if the main model name is retired
GROQ_BACKUPS = ["openai/gpt-oss-20b"]
GEMINI_BACKUPS = ["gemini-flash-latest"]

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

SUGGESTIONS = ["📰 Latest Solapur division news", "👋 Hi, what can you do?", "🚨 What to do after a derailment?",
               "📝 Help me draft a safety circular"]

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
.newscard {background:#ffffff0a; border:1px solid #ffffff14; border-radius:16px; padding:12px 16px;
  margin-bottom:10px; backdrop-filter: blur(10px); transition: all .2s;}
.newscard:hover {border-color:#00e5ff66; transform: translateY(-2px);}
.newscard a {color:#eceefc; text-decoration:none; font-weight:500; line-height:1.35;}
.newscard a:hover {color:#b6ff3c;}
.newsmeta {color:#9aa0c3; font-size:.75rem; margin-top:5px;}
.spot {border:1px solid #b6ff3c55; background:linear-gradient(135deg,#b6ff3c12,#00e5ff10); border-radius:18px;
  padding:6px 16px 2px; margin-bottom:14px;}
div[role="radiogroup"] {gap: .4rem; flex-wrap: wrap;}
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
    errs = []
    for model in dict.fromkeys([GROQ_MODEL] + GROQ_BACKUPS):
        try:
            kw = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
            r = client.chat.completions.create(
                model=model, temperature=0.4, max_tokens=2000,
                messages=[{"role": "system", "content": system}] + msgs, **kw)
            text = r.choices[0].message.content
            if text:
                return text
            errs.append(f"{model}: empty reply")
        except Exception as e:
            errs.append(f"{model}: {e}")
    raise RuntimeError("; ".join(errs))


def _gemini(system, msgs):
    from google.genai import types
    key = cfg("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing in Secrets")
    client = gemini_client(key)
    contents = [types.Content(role="user" if m["role"] == "user" else "model",
                              parts=[types.Part(text=m["content"])]) for m in msgs]
    errs = []
    for model in dict.fromkeys([GEMINI_MODEL] + GEMINI_BACKUPS):
        try:
            r = client.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(system_instruction=system,
                                                   temperature=0.4, max_output_tokens=2000))
            if r.text:
                return r.text
            errs.append(f"{model}: empty reply")
        except Exception as e:
            errs.append(f"{model}: {e}")
    raise RuntimeError("; ".join(errs))


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
    """Google News RSS (free, no key). Returns list of dicts, newest first."""
    q = query
    if days:
        q += f" when:{days}d"
    if after:
        q += f" after:{after}"
    if before:
        q += f" before:{before}"
    hl, ceid = NEWS_LANGS.get(lang, NEWS_LANGS["English"])
    url = "https://news.google.com/rss/search?" + urlencode({"q": q, "hl": hl, "gl": "IN", "ceid": ceid})
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    items, seen = [], set()
    for it in ET.fromstring(r.content).iter("item"):
        title, link = it.findtext("title", ""), it.findtext("link", "")
        src_el = it.find("source")
        src = src_el.text if src_el is not None and src_el.text else ""
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
        st.markdown(
            f'<div class="newscard"><a href="{n["link"]}" target="_blank">{n["title"]}</a>'
            f'<div class="newsmeta">📰 {n["source"] or "news"} · {fmt_when(n["when"])}</div></div>',
            unsafe_allow_html=True)


NEWS_WORDS = re.compile(r"\b(news|latest|today|recent|recently|update|updates|headline|headlines|happening|"
                        r"past news|old news|previous news)\b|खबर|समाचार|ताज़ा|ताजा|बातम्या|बातमी", re.I)


def news_intent(q):
    return bool(NEWS_WORDS.search(q))


def pick_topic(q):
    ql = q.lower()
    if "solapur" in ql or "सोलापुर" in q or "सोलापूर" in q:
        return "📍 Solapur Division"
    if re.search(r"safety|accident|derail|kavach|collision|spad", ql):
        return "🛡️ Railway Safety"
    if "central railway" in ql or "मध्य रेल" in q or "मध्य रेल्वे" in q:
        return "🚆 Central Railway"
    return "🇮🇳 Indian Railways"


def pick_days(q):
    ql = q.lower()
    if re.search(r"today|आज|आज की|24 hour", ql):
        return 1
    if re.search(r"year|past|old|previous|earlier|history|पिछले|मागील", ql):
        return 365
    if re.search(r"month|महीन|महिन", ql):
        return 30
    return 7


NEWS_SYSTEM = """You are the Safety Branch Assistant for Indian Railways staff. You are given recent NEWS HEADLINES
(numbered, with source and date). Summarize what is happening based ONLY on these headlines.
Rules: cite headlines as [1], [2]; group related items; mention dates; never add details that are not in the
headlines (you only see titles, not full articles) and say so if the user wants more detail - tell them to open the
link. If the headlines don't answer the question, say that. Reply in the user's language, short and clear.
Do not treat news as official instructions."""


def answer(question, history):
    """Returns (text, sources, mode): mode is 'news', 'doc' or 'chat'."""
    hist = [{"role": m["role"], "content": m["content"]} for m in history[-8:]]

    if news_intent(question):
        topic, days = pick_topic(question), pick_days(question)
        try:
            items = fetch_news(NEWS_TOPICS[topic], days=days)[:12]
        except Exception as e:
            return f"⚠️ Couldn't fetch news right now ({e}). Try again in a bit.", [], "chat"
        if not items and days < 365:
            items = fetch_news(NEWS_TOPICS[topic], days=365)[:12]
        if items:
            ctx = "\n".join(f"[{i}] {n['title']} ({n['source']}, {fmt_when(n['when'])})"
                            for i, n in enumerate(items, 1))
            msgs = hist + [{"role": "user", "content": f"NEWS HEADLINES ({topic}):\n{ctx}\n\nQUESTION: {question}"}]
            srcs = [f"[{i}] [{n['title']}]({n['link']}) - {n['source']}, {fmt_when(n['when'])}"
                    for i, n in enumerate(items, 1)]
            return generate(NEWS_SYSTEM, msgs), srcs, "news"

    ctx = retrieve(question)
    if ctx:
        context = "\n\n".join(f"[{i}] ({c['source']} {c['page']})\n{c['text']}" for i, c in enumerate(ctx, 1))
        msgs = hist + [{"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}]
        srcs = [f"[{i}] {c['source']} {c['page']}".strip() for i, c in enumerate(ctx, 1)]
        return generate(RAG_SYSTEM, msgs), srcs, "doc"
    return generate(CHAT_SYSTEM, hist + [{"role": "user", "content": question}]), [], "chat"


def render_msg(m):
    if m["role"] == "assistant" and m.get("mode"):
        label = {"doc": "📚 from documents", "news": "📰 live news"}.get(m["mode"], "💬 general chat")
        cls = "doc" if m["mode"] in ("doc", "news") else "chat"
        st.markdown(f'<span class="badge {cls}">{label}</span>', unsafe_allow_html=True)
    st.markdown(m["content"])
    if m.get("sources"):
        with st.expander("Sources"):
            st.write("\n".join(m["sources"]))


# ---------------- UI ----------------
def news_page():
    st.markdown('<div class="hero"><h1>Rail News</h1><p>live + past headlines · Solapur division first</p></div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    topic = c1.selectbox("Topic", list(NEWS_TOPICS.keys()), index=0)
    lang = c2.selectbox("Language", list(NEWS_LANGS.keys()), index=0)
    rng = st.radio("Time range", ["24 hours", "7 days", "30 days", "1 year", "Custom dates"],
                   index=1, horizontal=True)
    days = after = before = None
    if rng == "Custom dates":
        d1, d2 = st.columns(2)
        a_ = d1.date_input("From", dt.date.today() - dt.timedelta(days=180))
        b_ = d2.date_input("To", dt.date.today())
        after, before = a_.isoformat(), b_.isoformat()
    else:
        days = {"24 hours": 1, "7 days": 7, "30 days": 30, "1 year": 365}[rng]

    # Solapur spotlight (always visible unless Solapur is already the selected topic)
    if topic != "📍 Solapur Division":
        try:
            sp = fetch_news(NEWS_TOPICS["📍 Solapur Division"], days=30, lang=lang)[:4]
        except Exception:
            sp = []
        if sp:
            st.markdown('<div class="spot"><b>📍 Solapur Division spotlight</b> · last 30 days</div>',
                        unsafe_allow_html=True)
            news_cards(sp)

    try:
        with st.spinner("fetching headlines…"):
            items = fetch_news(NEWS_TOPICS[topic], days=days, after=after, before=before, lang=lang)
    except Exception as e:
        st.error(f"Couldn't fetch news: {e}")
        return
    st.markdown(f"**{topic}** · {len(items)} headlines")
    if not items:
        st.info("No headlines found for this range. Try a longer range or another language.")
        return
    if st.button("✨ Summarize these headlines"):
        ctx = "\n".join(f"[{i}] {n['title']} ({n['source']}, {fmt_when(n['when'])})"
                        for i, n in enumerate(items[:15], 1))
        try:
            with st.spinner("summarizing…"):
                st.markdown(generate(NEWS_SYSTEM, [{"role": "user", "content":
                            f"NEWS HEADLINES ({topic}):\n{ctx}\n\nGive a short summary of the key developments."}]))
        except Exception as e:
            st.error(f"Error: {e}")
    news_cards(items)
    st.caption("Headlines via Google News. Open the link for the full story. News is not official instructions.")


def chat_page(chunks):
    st.markdown('<div class="hero"><h1>Safety Branch AI</h1>'
                '<p>ask anything · rules, SOPs, news, drafts, or just vibes · हिंदी / English / मराठी</p></div>',
                unsafe_allow_html=True)
    if "history" not in st.session_state:
        st.session_state.history = []

    if not st.session_state.history:
        st.markdown("**Try one:**")
        cols = st.columns(2)
        for i, sg in enumerate(SUGGESTIONS):
            if cols[i % 2].button(sg, key=f"sug{i}"):
                st.session_state.pending = sg.split(" ", 1)[1]
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


if check_password():
    chunks, _, _ = build_index()
    n_files = len({c["source"] for c in chunks})

    with st.sidebar:
        st.markdown("### 🚆 Safety Branch AI")
        if chunks:
            st.markdown(f'<span class="pill ok">📚 {n_files} doc(s) · {len(chunks)} passages</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill warn">📭 no documents yet · chat + news mode</span>',
                        unsafe_allow_html=True)
        st.markdown(f'<span class="pill">⚡ {PROVIDER} → {FALLBACK or "none"}</span>', unsafe_allow_html=True)
        st.write("")
        if st.button("🧹 Clear chat"):
            st.session_state.history = []
            st.rerun()
        st.caption("Add PDFs/DOCX/TXT to the documents/ folder on GitHub to unlock cited answers.")

    page = st.radio("page", ["💬 Chat", "📰 News"], horizontal=True, label_visibility="collapsed")
    if page == "📰 News":
        news_page()
    else:
        chat_page(chunks)

    st.markdown('<div class="foot">AI-generated. Verify with the latest official instructions before acting.</div>',
                unsafe_allow_html=True)
