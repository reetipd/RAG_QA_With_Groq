import streamlit as st
import openai, os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key, model="Llama3-8b-8192"
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the context that is provided to you only.
    Provide the most accurate response based on the question asked by the user. 

    <context>
    {context}
    <context>

    Question: {input}
    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state: 
        st.session_state.embeddings = OpenAIEmbeddings()

        # Data Ingestion Step:
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")  

        # Load the documents
        st.session_state.docs = st.session_state.loader.load()

        # Text Splitting Step:
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

        #  Vector Store Creation Step:
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("RAG based Research paper QA Using OpenAI and Groq")


user_prompt = st.text_input("Enter your question here from the research papers:")

# Perform embeddings and storage in vector DB
if st.button("Document Emdeddings"):
    create_vector_embeddings()
    st.success("Document embeddings created successfully!")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Before reading the documents, vector data should be rertrieved
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    resp = retrieval_chain.invoke({"input":user_prompt})

    print(f"Response time: {time.process_time() - start} seconds")

    st.write(resp["answer"])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(resp["context"]):
            st.write(doc.page_content)
            st.write("----------------------")


