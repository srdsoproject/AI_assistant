"""
Safety Branch AI Assistant - Streamlit version (Groq + Gemini fallback).
Answers ONLY from documents in ./documents, with citations.
Run locally: streamlit run streamlit_app.py   |  Deploy: Streamlit Community Cloud
"""
import os, re, glob
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Safety Branch Assistant", page_icon="🚆", layout="centered")

DOC_DIR = os.path.join(os.path.dirname(__file__), "documents")
CHUNK_WORDS, OVERLAP, TOP_K = 220, 40, 6


def cfg(key, default=""):
    """Read from Streamlit secrets first, then environment variables."""
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

SYSTEM = """You are the Safety Branch Assistant for Indian Railways officials and staff.
Rules:
1. Answer ONLY from the numbered CONTEXT passages. Never invent rules, limits, rule numbers or dates.
2. Cite every fact like [1], [2] matching the passage numbers.
3. If the context does not contain the answer, say so clearly and suggest which manual/circular to check or to consult the competent authority.
4. Reply in the same language as the user (Hindi, English or Hinglish).
5. Be concise and practical. Use short bullet points for procedures.
6. For accident/incident questions, remind that the latest Board/Zonal instructions prevail."""


# ---------- optional password gate ----------
def check_password():
    pw = cfg("APP_PASSWORD")
    if not pw or st.session_state.get("auth"):
        return True
    st.title("🚆 Safety Branch Assistant")
    entered = st.text_input("Password", type="password")
    if st.button("Login"):
        if entered == pw:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Wrong password")
    return False


# ---------- documents ----------
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
    scores = cosine_similarity(vec.transform([query]), matrix)[0]
    return [chunks[i] for i in scores.argsort()[::-1][:TOP_K] if scores[i] > 0.02]


# ---------- LLMs ----------
def _groq(system, msgs):
    from groq import Groq
    r = Groq(api_key=cfg("GROQ_API_KEY")).chat.completions.create(
        model=GROQ_MODEL, temperature=0.2, max_tokens=1000,
        messages=[{"role": "system", "content": system}] + msgs)
    return r.choices[0].message.content


def _gemini(system, msgs):
    from google import genai
    from google.genai import types
    contents = [types.Content(role="user" if m["role"] == "user" else "model",
                              parts=[types.Part(text=m["content"])]) for m in msgs]
    r = genai.Client(api_key=cfg("GEMINI_API_KEY")).models.generate_content(
        model=GEMINI_MODEL, contents=contents,
        config=types.GenerateContentConfig(system_instruction=system,
                                           temperature=0.2, max_output_tokens=1000))
    return r.text


def generate(system, msgs):
    providers = {"groq": _groq, "gemini": _gemini}
    last = None
    for p in dict.fromkeys([PROVIDER, FALLBACK]):
        if p not in providers:
            continue
        try:
            return providers[p](system, msgs)
        except Exception as e:
            last = e
    raise last or RuntimeError("No valid LLM provider configured")


def answer(question, history):
    ctx = retrieve(question)
    if not ctx:
        return ("I could not find anything relevant in the loaded documents. "
                "Please add the relevant manual/circular or rephrase."), []
    context = "\n\n".join(f"[{i}] ({c['source']} {c['page']})\n{c['text']}"
                          for i, c in enumerate(ctx, 1))
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    msgs.append({"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"})
    srcs = [f"[{i}] {c['source']} {c['page']}".strip() for i, c in enumerate(ctx, 1)]
    return generate(SYSTEM, msgs), srcs


# ---------- UI ----------
if check_password():
    st.title("🚆 Safety Branch Assistant")
    st.caption("Answers from official manuals & circulars · हिंदी / English")
    chunks, _, _ = build_index()
    if not chunks:
        st.warning("No documents found. Add PDF/DOCX/TXT files to the `documents/` folder and redeploy.")

    if "history" not in st.session_state:
        st.session_state.history = []
    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("Sources"):
                    st.write("\n".join(m["sources"]))

    if q := st.chat_input("Type your question… / अपना प्रश्न लिखें…"):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    a, srcs = answer(q, st.session_state.history)
                except Exception as e:
                    a, srcs = f"Error: {e}", []
            st.markdown(a)
            if srcs:
                with st.expander("Sources"):
                    st.write("\n".join(srcs))
        st.session_state.history += [{"role": "user", "content": q},
                                     {"role": "assistant", "content": a, "sources": srcs}]
    st.caption("AI-generated from loaded documents. Verify with the latest official instructions before acting.")
