import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, ConfigurableField
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Model
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo").configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="gpt-3.5-turbo",
    gpt4=ChatOpenAI(temperature=0, model="gpt-4").configurable_fields(
        temperature=ConfigurableField(
            id="temperature",
            name="LLM Temperature",
            description="The temperature of the LLM",
        )
    )).configurable_fields(
    temperature=ConfigurableField(
        id="temperature",
        name="LLM Temperature",
        description="The temperature of the LLM",
    )
)

# Prompt Template

english_template = """
Answer the question based on following context:
{context}

Question: {question}
"""

chinese_template = """
Answer the question in Mandarin based on following context:
{context}

Question: {question}
"""


malay_template = """
Answer the question in Malay based on following context:
{context}

Question: {question}
"""

english_prompt = ChatPromptTemplate.from_template(template=english_template)
chinese_prompt = ChatPromptTemplate.from_template(template=chinese_template)
malay_prompt = ChatPromptTemplate.from_template(template=malay_template)
alternative_prompt = {
    "chinese": chinese_prompt,
    "malay": malay_prompt
}


prompt = english_prompt.configurable_alternatives(
    ConfigurableField(id="lang"),
    default_key="english",
    **alternative_prompt
)

# Load the data
uploaded_file = st.file_uploader(
    "Choose a txt file", type=["txt", "md", "pdf"])


if uploaded_file is not None:
    with open(uploaded_file.name, mode='wb') as w:
        w.write(uploaded_file.getvalue())

    if uploaded_file.type == "text/plain":
        loader = TextLoader(uploaded_file.name)
    if uploaded_file.type == "text/markdown":
        loader = UnstructuredMarkdownLoader(uploaded_file.name)
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(uploaded_file.name)

    documents = loader.load()
    # print(documents)

    # Clean up (delete) the file after processing
    if os.path.exists(uploaded_file.name):
        os.remove(uploaded_file.name)
        print(f"File {uploaded_file.name} has bee deleted")

    # Split the data
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)

    # Vectorstore
    vectorstore = FAISS.from_documents(splitter.split_documents(
        documents), embedding=OpenAIEmbeddings())

    # Retriever
    retriever = vectorstore.as_retriever()

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    # Chain
    chain = (
        setup_and_retrieval
        | prompt
        | model
        | StrOutputParser()
    )

    result = "No Result yet"

    with st.form("my_form"):
        question = st.text_input("Input your question about the documents")
        temperature = st.number_input(
            "Insert a temperature from 0.0 to 1.0", min_value=0.0, max_value=1.0)
        model = st.selectbox("What Model you want to choose",
                             ("gpt-3.5-turbo", "gpt4"))
        lang = st.selectbox(
            "What language of Response do you want", ("english", "chinese", "malay"))
        result = chain.with_config(
            configurable={
                "llm": model,
                "temperature": temperature,
                "lang": lang
            }
        ).invoke(question)
        st.form_submit_button("Submit")

    st.write(result)
