import os
import streamlit as st
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
os.environ["GRPC_VERBOSITY"] = "NONE"

load_dotenv()

def generate_rag_prompt(query, context="", conversation_history=[]):
    prompt = ""
    if conversation_history:
        for chat in conversation_history:
            prompt += "You: " + chat["query"] + "\n"
            prompt += "BOT: " + chat["answer"] + "\n"
    if context:
        escaped = context.replace("'","").replace('"', "").replace("\n"," ")
        prompt += ("""
        You are a helpful and informative bot that answers questions using text from the reference context included below. \
        Be sure to respond in a complete sentence.\
        You can use bulletpoints. \
        However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
        strike a friendly and converstional tone. \
        If the context is irrelevant to the answer, you may ignore it.
                CONTEXT: '{context}'
        """).format(context=context)
    else:
        prompt += ("""
        You are a helpful and informative bot that answers questions. \
        Be sure to respond in a complete sentence.\
        You can use bulletpoints. \
        However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
        strike a friendly and converstional tone.
        """)
    prompt += ("""
                QUESTION: '{query}'
                  
                ANSWER:
                """).format(query=query)
    return prompt

def get_relevant_context_from_db(query):
    context = ""
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    search_results = vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    return context

def generate_answer(prompt):
    genai.configure(api_key=os.environ["GEMEINI_API_KEY"])
    model = genai.GenerativeModel(model_name='gemini-1.5-flash')
    answer = model.generate_content(prompt)
    return answer.text

def on_click_callback():
    query = st.session_state.query
    use_custom_data = st.session_state.use_custom_data
    if use_custom_data:
        context = get_relevant_context_from_db(query)
    else:
        context = ""
    prompt = generate_rag_prompt(query=query, context=context, conversation_history=st.session_state.chat_history)
    answer = generate_answer(prompt=prompt)
    st.session_state.chat_history.append({"query": query, "answer": answer})

def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

initialize_session_state()

logo_path = r"assets\logo_dark.png"
st.title("")
col1, col2 = st.columns([1, 4])
with col1:
    st.image(logo_path, width=150)
with col2:
    st.header("AI Assistant")
st.markdown("A helpful and informative bot that answers your questions regarding the PIA.")

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-from")

with chat_placeholder:
    for chat in st.session_state.chat_history:
        st.write("You: " + chat["query"])
        st.write("BOT: " + chat["answer"])

with prompt_placeholder:
    st.markdown("**Chat**")
    cols = st.columns((6,1))
    cols[0].text_input(
        "Chat",
        value="",
        label_visibility="collapsed",
        key="query",
    )
    
    cols[1].form_submit_button(
        "Ask",
        type="primary",
        on_click=on_click_callback,
    )
    cols[0].checkbox("Ask about PIA", key="use_custom_data")
