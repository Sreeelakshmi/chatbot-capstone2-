import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

# Streamlit configuration
st.set_page_config(
    page_title="TRAVEL ADVISOR CHATBOT",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="auto"
)
st.title("TRAVEL ADVISOR CHATBOT 💬")

# API Key setup
openai.api_key = st.secrets.openai_key

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! 👋 I’m your AI-powered travel advisor. How can I assist you today?"}]

# Cache and load data
@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="data", recursive=True)
    docs = reader.load_data()
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        system_prompt="""Hello! 👋 I’m your AI-powered travel advisor, here to make planning your next trip seamless and exciting!"""
    )
    index = VectorStoreIndex.from_documents(docs)
    return index

index = load_data()

# Initialize chat engine
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=False
    )

# Chat input and processing
if prompt := st.chat_input("Ask me anything about your travel plans!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        response = st.session_state.chat_engine.chat(prompt)
        st.write(response.response)
        st.session_state.messages.append({"role": "assistant", "content": response.response})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
