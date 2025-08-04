
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ------------------ PDF Processing ------------------

def extract_text_from_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

def split_text_into_chunks(text, chunk_size=2000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# ------------------ Embedding & Vector Store ------------------

def create_vector_store(chunks, path="faiss_index"):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local(path)
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return False

def load_vector_store(path="faiss_index"):
    if not os.path.exists(f"{path}/index.faiss"):
        st.warning("Vector store not found. Please upload and process PDFs first.")
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

# ------------------ QA Chain ------------------

def build_qa_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, say "Answer not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def answer_user_question(question, vector_store):
    if not question:
        st.warning("Please enter a question.")
        return
    try:
        docs = vector_store.similarity_search(question)
        chain = build_qa_chain()
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        st.write("📘 **Reply:**", response["output_text"])
    except Exception as e:
        st.error(f"Error generating response: {e}")

# ------------------ Streamlit UI ------------------

def main():
    st.set_page_config("PDF Chatbot", page_icon="📚")
    st.header("PDF Chatbot 🤖")

    user_question = st.text_input("Ask a question based on the uploaded PDFs:")

    if user_question:
        vector_store = load_vector_store()
        if vector_store:
            answer_user_question(user_question, vector_store)

    with st.sidebar:
        st.image("img/Robot.jpg")
        st.title("📁 Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing PDFs..."):
                raw_text = extract_text_from_pdfs(pdf_docs)
                if raw_text:
                    chunks = split_text_into_chunks(raw_text)
                    if create_vector_store(chunks):
                        st.success("Vector store created successfully.")
                else:
                    st.warning("No text extracted from PDFs.")

        #st.image("img/gkj.jpg")
        st.caption("AI App created by @ Srikanth")

    st.markdown("""
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #27F5D6; padding: 15px; text-align: center;">
            © https://github.com/srikanth449 | Made with ❤️
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
