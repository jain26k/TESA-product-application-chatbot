import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

st.title('Audio Test')
if st.button('Generate and Play Audio'):
    response = client.audio.speech.create(
        model='tts-1',
        voice='nova',
        input='Hello, I am TESA Advisor. Welcome to Action TESA.'
    )
    st.audio(response.content, format='audio/mp3')
    st.success('Audio generated!')
