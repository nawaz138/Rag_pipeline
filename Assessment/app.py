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
# CUSTOM CSS
####################################################

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

.stChatMessage{
    border-radius:15px;
    padding:10px;
}

h1{
    color:#1E88E5;
}

</style>
""", unsafe_allow_html=True)


####################################################
# SIDEBAR
####################################################

with st.sidebar:

    st.title("🤖 Insurance Bot")

    st.markdown("---")

    st.info(
        """
        Ask questions from the uploaded
        HDFC insurance policy.

        Powered by:

        - LangChain
        - Chroma
        - HuggingFace Embeddings
        - Groq Llama-3.3-70B
        """
    )

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()


####################################################
# TITLE
####################################################

st.title("📄 Insurance RAG Chatbot")

st.caption("Ask anything from the policy document")


####################################################
# LOAD VECTOR DB
####################################################

@st.cache_resource
def load_vectorstore():

    file_path = "HDFC-Surgicare-Plan-101N043V01.pdf"

    pdf = PdfReader(file_path)

    documents = []

    for page in pdf.pages:

        documents.append(
            Document(
                page_content=page.extract_text(),
                metadata={
                    "file_name": file_path,
                    "page_number": page.page_number + 1,
                    "total_pages": len(pdf.pages)
                }
            )
        )

    splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = []
    metadatas = []

    for doc in documents:

        chunks = splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):

            meta = dict(doc.metadata)
            meta["sentence_sequence"] = i + 1

            texts.append(chunk)
            metadatas.append(meta)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    return vectorstore


vectorstore = load_vectorstore()

####################################################
# GROQ CLIENT
####################################################

client = Groq(
    api_key="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"
)

####################################################
# SESSION STATE
####################################################

if "messages" not in st.session_state:
    st.session_state.messages = []


####################################################
# DISPLAY CHAT
####################################################

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


####################################################
# USER INPUT
####################################################

question = st.chat_input("Ask your question...")

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

        with st.spinner("Searching document..."):

            docs = vectorstore.as_retriever(
                search_kwargs={"k":4}
            ).invoke(question)

            context = "\n\n".join(
                [d.page_content for d in docs]
            )

            prompt = f"""
You are an expert assistant.

Answer ONLY from the provided context.

If the answer is unavailable in context reply exactly:

No context found.

User Question:
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

            answer = response.choices[0].message.content

            st.markdown(answer)

            with st.expander("📚 Retrieved Context"):

                for i, doc in enumerate(docs):

                    st.markdown(
                        f"### Chunk {i+1}"
                    )

                    st.write(
                        f"**Page:** {doc.metadata.get('page_number')}"
                    )

                    st.write(doc.page_content)

                    st.divider()

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )