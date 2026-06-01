# ============================================================
# chatbot_rag.py — TESA Advisor with RAG + Voice (EBM)
# Brand: Action TESA — White main, dark sidebar, green accents
# EBM: Ears (Whisper) + Brain (GPT-4o-mini RAG) + Mouth (TTS)
# Hybrid: text OR voice input, text + audio output
# Audio stored in session state — survives rerun
# ============================================================

import os
import time
import io
import base64
from audio_recorder_streamlit import audio_recorder
from pinecone import Pinecone
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── PLATFORM LAYER: connect to Pinecone ──
@st.cache_resource
def load_vector_db():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX", "tesa-knowledge"))
    return index

# ── SPELLING CORRECTION ──
def normalize_query(query):
    corrections = {
        "boilio": "BOILO", "boilo": "BOILO", "boylo": "BOILO",
        "bwp": "BOILO BWP HDF Board", "boilo bwp": "BOILO BWP HDF Board",
        "hdmr": "HDHMR", "hdhm": "HDHMR", "hdhmr": "HDHMR",
        "mdf": "MDF board",
        "pb ": "Particle Board ", "particle board": "Particle Board",
        "moistmaster": "Moist Master", "moist master": "Moist Master",
        "abraze": "Abraze Board",
        "ornamat": "Ornamatte",
        "hdhmr door": "HDHMR Doors",
        "moistro": "MOISTRO Doors",
    }
    q = query.lower()
    for wrong, right in corrections.items():
        q = q.replace(wrong, right.lower())
    return q

# ── EMBED ──
def embed_query(query):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding

# ── RETRIEVE ──
def retrieve_chunks(query, index, k=6):
    normalized = normalize_query(query)
    query_embedding = embed_query(normalized)
    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )
    chunks = []
    for match in results["matches"]:
        chunks.append({
            "text": match["metadata"]["text"],
            "source": match["metadata"]["source"],
            "url": match["metadata"]["url"],
            "distance": match["score"]
        })
    return chunks

# ── RAG PROMPT ──
def build_rag_prompt(query, chunks):
    context = "\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in chunks
    ])
    return f"""You are answering a question about Action TESA wood panel products.
Use the context below to answer. Context may come from PDFs with tables — extract any relevant numbers, specs, or facts even if formatting looks broken.

If you find ANY relevant information, use it to answer fully and specifically.
Only say "I don't have that information" if the context truly contains nothing relevant at all.

Note: The user may have misspelled product names — map to the closest Action TESA product.

CONTEXT:
{context}

QUESTION: {query}

Answer directly and specifically. Include any numbers, specs, or measurements you find in the context."""

# ── CC-SC-R SYSTEM PROMPT ──
SYSTEM_PROMPT = """
CONTEXT:
You are TESA Advisor, the official AI product expert for Action TESA (Balaji Action Buildwell Pvt. Ltd.) — India's leading wood panel company. Tagline: Koi Nahi Aisa.

CONSTRAINTS:
- Users may misspell product names. Common variations: BOILO/boilo/boilio, HDHMR/HDMR/hdhmr, MDF/mdf, Particle Board/PB, Abraze/abraze, Moist Master/moistmaster. Always interpret the closest matching product.
- Answer ONLY from the provided context. Never invent specs.
- Do not discuss competitor products.
- For pricing: say "contact Action TESA directly".
- If context is truly insufficient, say so and give contact details.
- Respond in the same language the user writes in — Hindi, English, or Hinglish.
- When responding to voice queries, keep answers under 3 sentences so they sound natural when spoken.

STRUCTURE:
1. Direct answer (1-2 sentences)
2. Why — material science or practical reason
3. Specific Action TESA product in **bold**
4. Flag humidity / load-bearing / outdoor warnings if relevant

CHECKPOINTS:
- Humid/coastal → HDHMR or BOILO
- Load-bearing → minimum 18mm
- Outdoor → BWP/BOILO only, never standard MDF
- Budget → Particle Board

CONTACT (share when relevant):
- Toll Free: 1800-103-454
- Email: info@actiontesa.com
- Website: www.actiontesa.com
"""

# ── EBM: EARS — Whisper STT ──
def transcribe_audio(audio_bytes):
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcript.text
    except Exception as e:
        print(f"STT error: {e}")
        return None

# ── EBM: MOUTH — OpenAI TTS ──
def text_to_speech(text):
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text[:500]
        )
        audio_data = response.content
        print(f"TTS success — {len(audio_data)} bytes generated")
        return audio_data
    except Exception as e:
        print(f"TTS error: {e}")
        return None

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="TESA Advisor | Action TESA",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@600&display=swap');

.stApp {
    background-color: #f7f6f2 !important;
    font-family: 'Inter', sans-serif !important;
}
.main .block-container {
    max-width: 800px !important;
    padding: 1.5rem 2rem 8rem 2rem !important;
}
[data-testid="stSidebar"] {
    background-color: #1a1a18 !important;
    border-right: 3px solid #4a9a3a !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #d8d8d0 !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #4a9a3a !important;
    font-size: 14px !important;
}
[data-testid="stSidebar"] a { color: #4a9a3a !important; }
[data-testid="stSidebar"] hr { border-color: #2a2a28 !important; }
[data-testid="stSidebar"] button {
    background-color: #2a2a28 !important;
    border: 1px solid #3a3a38 !important;
    color: #d8d8d0 !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    text-align: left !important;
    width: 100% !important;
    margin-bottom: 4px !important;
}
[data-testid="stSidebar"] button:hover {
    border-color: #4a9a3a !important;
    color: #4a9a3a !important;
    background-color: #1a2a18 !important;
}
.tesa-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 0 16px 0;
    border-bottom: 2px solid #e8e4dc;
    margin-bottom: 24px;
}
.tesa-title {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    color: #1a1a18;
    margin: 0;
    line-height: 1;
}
.tesa-title span { color: #cc2222; }
.tesa-sub {
    font-size: 12px;
    color: #888;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 4px 0 0 0;
}
[data-testid="stChatMessage"] {
    background-color: #ffffff !important;
    border: 1px solid #e8e4dc !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    color: #1a1a18 !important;
}
[data-testid="stChatMessage"] p {
    color: #1a1a18 !important;
    line-height: 1.7 !important;
    font-size: 14px !important;
}
[data-testid="stChatMessage"] strong { color: #2d7a24 !important; }
[data-testid="stBottom"] {
    background: #f7f6f2 !important;
    border-top: 1px solid #e8e4dc !important;
    padding: 12px 0 !important;
}
[data-testid="stChatInput"] {
    border: 2px solid #4a9a3a !important;
    border-radius: 14px !important;
    background: #ffffff !important;
    box-shadow: 0 2px 8px rgba(74,154,58,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: #1a1a18 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
[data-testid="stChatInputSubmitButton"] svg { fill: #4a9a3a !important; }
.source-tag {
    display: inline-block;
    background: #f0f8ee;
    border: 1px solid #4a9a3a;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    color: #2d7a24;
    margin: 4px 4px 0 0;
    font-weight: 500;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
hr { border-color: #e8e4dc !important; }
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e8e4dc !important;
    border-radius: 10px !important;
    padding: 10px !important;
}
[data-testid="stMetric"] label {
    color: #888 !important;
    font-size: 11px !important;
}
[data-testid="stMetricValue"] {
    color: #1a1a18 !important;
    font-size: 20px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## 🌿 TESA Advisor")
    st.markdown("*Powered by RAG · actiontesa.com*")
    st.divider()
    st.markdown("### 📞 Contact Action TESA")
    st.markdown("""
📞 **Toll Free:** 1800-103-454
📧 **Email:** info@actiontesa.com
🌐 [actiontesa.com](https://www.actiontesa.com)
📸 [@actiontesa_official](https://www.instagram.com/actiontesa_official/)
    """)
    st.divider()
    st.markdown("### ⚡ Quick Questions")
    quick_questions = [
        "Best board for humid climate wardrobe?",
        "HDHMR vs MDF for kitchen shutters?",
        "Which board for a coastal modular kitchen?",
        "What is BOILO and when to use it?",
        "How many hours does BOILO withstand boiling water?",
        "What certifications do TESA boards have?",
        "Best board for a bathroom door?",
        "What flooring options does Action TESA offer?",
        "What thickness options does MDF come in?",
        "What is Abraze board used for?",
    ]
    for q in quick_questions:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["quick_q"] = q
    st.divider()
    temp = st.slider("🌡️ Response creativity", 0.0, 1.0, 0.3, 0.1)
    st.caption("Low = precise specs · High = creative")
    if "total_tokens" in st.session_state:
        st.metric("Tokens used", st.session_state["total_tokens"])

# ── LOAD VECTOR DB ──
try:
    index = load_vector_db()
    db_loaded = True
except Exception as e:
    db_loaded = False
    st.error(f"⚠️ Could not connect to Pinecone: {e}")

# ── SESSION STATE ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

# ── HEADER ──
st.markdown("""
<div class="tesa-header">
    <div>
        <div class="tesa-title">TESA <span>Advisor</span></div>
        <div class="tesa-sub">Action TESA · Wood Panel Expert · Koi Nahi Aisa</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── CHAT HISTORY ──
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            st.markdown(" ".join([
                f'<span class="source-tag">📄 {s}</span>'
                for s in message["sources"]
            ]), unsafe_allow_html=True)
        if "audio" in message and message["audio"]:
            st.audio(message["audio"], format="audio/mp3")

# ── EBM HYBRID INPUT — Voice OR Text ──
col1, col2 = st.columns([1, 11])
with col1:
    audio_bytes = audio_recorder(
        text="",
        recording_color="#4a9a3a",
        neutral_color="#888888",
        icon_size="2x"
    )
with col2:
    text_input = st.chat_input(
        "Type in English or Hindi… or use the mic 🎙️"
    )

# ── DETERMINE PROMPT SOURCE ──
prompt = None

# Voice — EARS
if audio_bytes and audio_bytes != st.session_state.get("last_audio"):
    st.session_state["last_audio"] = audio_bytes
    with st.spinner("🎙️ Transcribing your voice..."):
        transcribed = transcribe_audio(audio_bytes)
    if transcribed:
        st.info(f"🎙️ You said: *{transcribed}*")
        prompt = transcribed

# Text
if text_input:
    prompt = text_input

# Quick question
if "quick_q" in st.session_state and st.session_state["quick_q"]:
    prompt = st.session_state["quick_q"]
    st.session_state["quick_q"] = None

# ── RAG WORKFLOW ──
if prompt and db_loaded:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            start_time = time.time()
            chunks = retrieve_chunks(prompt, index, k=6)
            rag_prompt = build_rag_prompt(prompt, chunks)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *[{"role": m["role"], "content": m["content"]}
                      for m in st.session_state.messages[:-1]],
                    {"role": "user", "content": rag_prompt}
                ],
                temperature=temp
            )

            latency = round(time.time() - start_time, 2)
            reply = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            st.session_state.total_tokens += tokens_used

            print(f"\n[TESA RAG LOG]")
            print(f"  Query      : {prompt}")
            print(f"  Normalized : {normalize_query(prompt)}")
            print(f"  Sources    : {[c['source'] for c in chunks]}")
            print(f"  Latency    : {latency}s | Tokens: {tokens_used}")

        st.markdown(reply)

        # EBM: MOUTH — generate and store audio
        with st.spinner("🔊 Generating audio..."):
            audio_response = text_to_speech(reply)

        if audio_response:
            st.audio(audio_response, format="audio/mp3")

        unique_sources = list(set([c["source"] for c in chunks]))
        st.markdown(" ".join([
            f'<span class="source-tag">📄 {s}</span>'
            for s in unique_sources
        ]), unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "sources": unique_sources,
        "audio": audio_response if audio_response else None
    })
    st.rerun()