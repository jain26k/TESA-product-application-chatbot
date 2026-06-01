import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are TESA Advisor, an expert AI assistant for Action TESA — India's leading wood panel company specialising in Particle Board and MDF.

CONTEXT:
You help architects, interior designers, and contractors make the right product decisions quickly and confidently. You know everything about wood panels — thickness, grades, humidity resistance, load-bearing capacity, surface finishes, and ideal use cases.

CONSTRAINTS:
- Only answer questions related to wood panels, interior design materials, and construction applications
- If asked about pricing, say pricing varies and they should contact Action TESA sales directly
- Never guess technical specs — if you don't know, say so clearly
- Do not discuss competitor products

STRUCTURE your responses:
- Lead with the direct answer
- Follow with the reason (material science or practical explanation)
- End with an Action TESA product recommendation where relevant

CHECKPOINTS — always flag:
- Humid or outdoor environments → recommend moisture-resistant grades
- Load-bearing applications → recommend appropriate thickness
- Budget-sensitive projects → offer a value alternative

REVIEW — a good response is under 150 words, jargon-free, and specific.
"""

st.set_page_config(page_title="TESA Advisor", page_icon="🪵")
st.title("🪵 TESA Advisor")
st.caption("Your expert guide to Action TESA wood panels")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Ask me about wood panels..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.messages,
        temperature=0.3
    )

    reply = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)