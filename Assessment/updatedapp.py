import os
import streamlit as st

from groq import Groq
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


####################################################
# PAGE CONFIG
####################################################

st.set_page_config(
    page_title="Insurance RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


####################################################
# LOAD EMBEDDINGS
####################################################

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )



####################################################
# LOAD VECTOR DB
####################################################

@st.cache_resource
def load_vectorstore():

    embeddings = get_embeddings()


    vectorstore = Chroma(
        collection_name="insurance_docs",
        embedding_function=embeddings
    )


    file_path = "HDFC-Surgicare-Plan-101N043V01.pdf"


    pdf = PdfReader(file_path)


    texts = []
    metadatas = []


    splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=1000,
        chunk_overlap=200
    )


    for page_number, page in enumerate(pdf.pages):

        content = page.extract_text()


        if content:

            chunks = splitter.split_text(content)


            for i, chunk in enumerate(chunks):

                texts.append(chunk)

                metadatas.append(
                    {
                        "file_name": file_path,
                        "page_number": page_number + 1,
                        "sentence_sequence": i + 1
                    }
                )


    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas
    )


    return vectorstore



vectorstore = load_vectorstore()



####################################################
# ADD UPLOADED DOCUMENT
####################################################

def add_uploaded_document(uploaded_file, vectorstore):


    pdf = PdfReader(uploaded_file)


    splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=1000,
        chunk_overlap=200
    )


    texts = []
    metadatas = []


    for page_number, page in enumerate(pdf.pages):

        content = page.extract_text()


        if content:

            chunks = splitter.split_text(content)


            for i, chunk in enumerate(chunks):

                texts.append(chunk)

                metadatas.append(
                    {
                        "file_name": uploaded_file.name,
                        "page_number": page_number + 1,
                        "sentence_sequence": i + 1
                    }
                )


    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas
    )


    return len(texts)



####################################################
# SIDEBAR
####################################################

with st.sidebar:


    st.title("🤖 Insurance Bot")


    st.divider()


    st.subheader("📂 Upload Document")


    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )


    if uploaded_file:


        if "uploaded_files" not in st.session_state:

            st.session_state.uploaded_files = []


        if uploaded_file.name not in st.session_state.uploaded_files:


            with st.spinner("Processing document..."):


                chunks = add_uploaded_document(
                    uploaded_file,
                    vectorstore
                )


                st.session_state.uploaded_files.append(
                    uploaded_file.name
                )


            st.success(
                f"✅ Added {uploaded_file.name}\n\nChunks: {chunks}"
            )


    st.divider()


    st.info(
        """
Ask questions from:

✔ HDFC Policy  
✔ Uploaded Documents


Powered by:

- LangChain
- Chroma
- HuggingFace
- Groq Llama
"""
    )


    st.divider()


    if st.button("🗑 Clear Chat"):

        st.session_state.messages=[]

        st.rerun()



####################################################
# TITLE
####################################################

st.title("📄 Insurance RAG Chatbot")


st.caption(
    "Ask questions from insurance documents"
)



if "uploaded_files" in st.session_state:

    if st.session_state.uploaded_files:

        st.success(
            "📚 Active Documents: "
            +
            ", ".join(
                st.session_state.uploaded_files
            )
        )



####################################################
# GROQ
####################################################

client = Groq(
    api_key="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"
)



####################################################
# CHAT MEMORY
####################################################

if "messages" not in st.session_state:

    st.session_state.messages=[]



####################################################
# DISPLAY CHAT
####################################################

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )



####################################################
# CHAT INPUT
####################################################

question = st.chat_input(
    "Ask your question..."
)



if question:


    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )


    with st.chat_message("user"):

        st.markdown(question)



    with st.chat_message("assistant"):


        with st.spinner(
            "Searching documents..."
        ):


            docs = vectorstore.as_retriever(
                search_kwargs={
                    "k":4
                }
            ).invoke(question)



            context="\n\n".join(
                [
                    d.page_content
                    for d in docs
                ]
            )



            prompt=f"""
You are an insurance assistant.

Answer only using the context.

If answer is unavailable say:

No context found.


Question:
{question}


Context:
{context}
"""



            response = client.chat.completions.create(

                model="llama-3.3-70b-versatile",

                temperature=0,

                messages=[
                    {
                        "role":"user",
                        "content":prompt
                    }
                ]

            )


            answer=response.choices[0].message.content



            st.markdown(answer)



            with st.expander(
                "📚 Retrieved Context"
            ):


                for i,doc in enumerate(docs):

                    st.markdown(
                        f"### Chunk {i+1}"
                    )

                    st.write(
                        doc.metadata
                    )

                    st.write(
                        doc.page_content
                    )

                    st.divider()



    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )