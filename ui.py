import os
import streamlit as st
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
os.environ["GRPC_VERBOSITY"] = "NONE"

load_dotenv()

def generate_rag_prompt(query, context):
    escaped = context.replace("'","").replace('"', "").replace("\n"," ")
    prompt = ("""
    You are a helpful and informative bot that answers questions using text from the reference context included below. \
    Be sure to respond in a complete sentence.\
    You can use bulletpoints. \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and converstional tone. \
    If the context is irrelevant to the answer, you may ignore it.
            QUESTION: '{query}'
            CONTEXT: '{context}'
              
            ANSWER:
            """).format(query=query, context=context)
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

logo_path = r"assets\logo_dark.png"
st.title("")
col1, col2 = st.columns([1, 4])
with col1:
    st.image(logo_path, width=150)
with col2:
    st.header("AI Assistant")
st.markdown("A helpful and informative bot that answers your questions regarding the PIA.")

input_container = st.container()
with input_container:
    query = st.text_input("You:", value="")

output_container = st.container()
with output_container:
    st.markdown("")

if st.button("Ask"):
    context = get_relevant_context_from_db(query)
    prompt = generate_rag_prompt(query=query, context=context)
    answer = generate_answer(prompt=prompt)
    output_container.empty()
    with output_container:
        st.markdown("BOT: " + answer)