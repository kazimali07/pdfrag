import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from uuid import uuid4
import os

# Load environment variables
load_dotenv()

# Configuration
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

# Page config
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Header
st.title("📚 Chat with Your PDFs")

# Initialize the ChatOpenAI model
@st.cache_resource
def initialize_llm():
    return ChatOpenAI(
        temperature=0.5,
        model="gpt-4",
        streaming=True
    )

# Sidebar for PDF processing
with st.sidebar:
    st.header("📁 PDF Processing")
    if st.button("Process PDFs in data folder"):
        try:
            # Create data directory if it doesn't exist
            if not os.path.exists(DATA_PATH):
                os.makedirs(DATA_PATH)
                st.info(f"Created directory: {DATA_PATH}")
            
            # Initialize embeddings model
            embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
            
            # Load PDFs
            loader = PyPDFDirectoryLoader(DATA_PATH)
            documents = loader.load()
            
            if not documents:
                st.error("No PDFs found in the data directory!")
                st.stop()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_documents(documents)
            
            # Generate unique IDs
            uuids = [str(uuid4()) for _ in range(len(chunks))]
            
            # Create vector store
            vector_store = Chroma(
                collection_name="example_collection",
                embedding_function=embeddings_model,
                persist_directory=CHROMA_PATH,
            )
            
            # Add documents to vector store
            vector_store.add_documents(documents=chunks, ids=uuids)
            st.session_state.vector_store = vector_store
            
            st.success(f"Processed {len(chunks)} chunks from {len(documents)} PDFs")
        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")

# Main chat interface
if st.session_state.vector_store is None and os.path.exists(CHROMA_PATH):
    # Try to load existing vector store
    try:
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
        st.session_state.vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings_model,
            persist_directory=CHROMA_PATH
        )
    except Exception as e:
        st.error(f"Error loading existing vector store: {str(e)}")

if st.session_state.vector_store is None:
    st.info("Please process some PDFs using the sidebar first!")
    st.stop()

# Initialize LLM
llm = initialize_llm()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your PDFs..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get relevant documents using the retriever as a runnable
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(prompt)  # Using the newer invoke method
    knowledge = "\n\n".join([doc.page_content for doc in docs])

    # Create the RAG prompt
    rag_prompt = f"""You are an assistant that answers questions based on the provided knowledge.
    Only use the information from the provided knowledge to answer questions.
    If you're unsure or the answer isn't in the provided knowledge, say so.
    
    Question: {prompt}
    
    Knowledge: {knowledge}
    """

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        for response in llm.stream(rag_prompt):
            full_response += response.content
            message_placeholder.write(full_response + "▌")
        message_placeholder.write(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Show debug info in expander
    with st.expander("Debug Information", expanded=False):
        st.write("🔍 Retrieved Chunks:")
        for i, doc in enumerate(docs, 1):
            st.text_area(f"Chunk {i}", doc.page_content, height=100)
