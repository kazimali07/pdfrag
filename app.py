import streamlit as st
# Page config must be the first Streamlit command
st.set_page_config(
    page_title="CWEPS PDF Assistant",
    page_icon="📚",
    layout="wide"
)

import os
import ssl
import certifi

# Configure SSL certificate
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from uuid import uuid4
import requests
import sys

# Load environment variables
load_dotenv()

# Debug network connectivity
st.sidebar.write("System Info:")
st.sidebar.write(f"Python version: {sys.version}")
try:
    # Test basic internet connectivity
    response = requests.get("https://api.openai.com", timeout=5)
    st.sidebar.success("✅ Can reach OpenAI's domain")
except Exception as e:
    st.sidebar.error(f"❌ Cannot reach OpenAI's domain: {str(e)}")

# Add API connection test with more detailed error handling
try:
    # First verify the API key is loaded
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("API key not found in environment variables")
    
    st.sidebar.write("API Key length:", len(api_key) if api_key else "No key found")
    
    # Initialize with explicit API key
    test_llm = ChatOpenAI(
        openai_api_key=api_key,  # Explicitly pass the API key
        timeout=10,
        model="gpt-4o",  # Use a specific model
        temperature=0.2  # Match the temperature used in initialize_llm
    )
    test_response = test_llm.invoke("Say 'API connection successful'")
    st.sidebar.success("✅ OpenAI API connection successful!")
    st.sidebar.write(test_response.content)
except ValueError as ve:
    st.sidebar.error(f"❌ Configuration Error: {str(ve)}")
except requests.exceptions.ProxyError as pe:
    st.sidebar.error(f"❌ Proxy Error: {str(pe)}")
except requests.exceptions.SSLError as se:
    st.sidebar.error(f"❌ SSL Error: {str(se)}")
except requests.exceptions.ConnectionError as ce:
    st.sidebar.error(f"❌ Connection Error: {str(ce)}")
except Exception as e:
    st.sidebar.error(f"❌ API Connection Error: {str(e)}")
    st.sidebar.error(f"Error Type: {type(e).__name__}")
    import traceback
    st.sidebar.error(f"Full error: {traceback.format_exc()}")

# Debug information for API key
st.sidebar.write("API Key exists:", bool(os.getenv('OPENAI_API_KEY')))
if api_key:
    st.sidebar.write("API Key format valid:", api_key.startswith('sk-') and len(api_key) > 40)

# Configuration
DATA_PATH = "data"

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
        temperature=0.7,
        model="gpt-4o"
    )

# Sidebar for PDF processing
with st.sidebar:
    st.header("📁 PDF Processing")
    
    # Add number of results control
    k_results = st.slider("Number of Results", min_value=1, max_value=10, value=3, step=1,
                         help="Number of PDF documents to retrieve for each query")
    
    if st.button("Process PDFs in data folder"):
        try:
            # Create data directory if it doesn't exist
            if not os.path.exists(DATA_PATH):
                os.makedirs(DATA_PATH)
                st.info(f"Created data directory at {DATA_PATH}")
                st.info("Please add PDF files to the data directory and try again.")
                st.stop()
            
            # Initialize embeddings model
            embeddings = OpenAIEmbeddings()
            
            # Load PDFs
            if not os.listdir(DATA_PATH):
                st.error("The data directory is empty! Please add PDF files.")
                st.stop()
                
            loader = PyPDFDirectoryLoader(DATA_PATH)
            try:
                documents = loader.load()
                st.info(f"Loaded {len(documents)} PDF pages")
            except Exception as e:
                st.error(f"Error loading PDFs: {str(e)}")
                st.stop()
                
            # Process each PDF as a separate chunk
            st.info("Processing each PDF as a separate chunk")
            
            # Group documents by source file
            pdf_groups = {}
            for doc in documents:
                source = doc.metadata.get('source', 'unknown')
                if source not in pdf_groups:
                    pdf_groups[source] = []
                pdf_groups[source].append(doc)
            
            # Combine pages from each PDF into a single document
            chunks = []
            for source, docs in pdf_groups.items():
                # Sort pages by page number if available
                docs.sort(key=lambda x: x.metadata.get('page', 0))
                
                # Combine all pages from this PDF
                combined_text = "\n\n".join([doc.page_content for doc in docs])
                
                # Create a new document with the combined text
                try:
                    # Try the new method first (for newer versions of LangChain)
                    combined_doc = docs[0].model_copy()
                except AttributeError:
                    # Fall back to the old method for backward compatibility
                    combined_doc = docs[0].copy()
                
                combined_doc.page_content = combined_text
                combined_doc.metadata['page_count'] = len(docs)
                combined_doc.metadata['is_combined_pdf'] = True
                combined_doc.metadata['filename'] = source.split('/')[-1]  # Extract just the filename
                
                chunks.append(combined_doc)
            
            st.info(f"Created {len(chunks)} chunks from {len(pdf_groups)} PDFs")
            
            # Create vector store in memory (not persisted to disk)
            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=f"pdf_collection_{uuid4()}"  # Unique collection name
            )
            
            # Store the k_results value in session state
            st.session_state.k_results = k_results
            
            st.success(f"✅ Successfully processed {len(pdf_groups)} PDFs ({len(documents)} total pages)")
            
        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")

# Main chat interface
if st.session_state.vector_store is None:
    st.info("Please process some PDFs using the sidebar first!")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your PDFs"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Get k_results from session state (default to 5 if not set)
            k = getattr(st.session_state, 'k_results', 5)
            
            # Search for relevant documents
            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            docs = retriever.invoke(prompt)
            
            # Initialize LLM
            llm = initialize_llm()
            
            # Prepare context with better metadata information
            context_parts = []
            for i, doc in enumerate(docs):
                # Extract useful metadata
                filename = doc.metadata.get('filename', doc.metadata.get('source', 'Unknown').split('/')[-1])
                page_count = doc.metadata.get('page_count', 'Unknown')
                
                # Format header with PDF information
                header = f"Document {i+1} (PDF: {filename}, Pages: {page_count}):"
                
                context_parts.append(f"{header}\n{doc.page_content}")
            
            # Join all context parts
            context = "\n\n" + "\n\n".join(context_parts)
            
            # Generate response
            response = llm.invoke(
                f"""You are a helpful assistant that answers questions based on the provided PDF documents.
                Each document in the context represents a complete PDF file.
                
                Context from PDFs:
                {context}
                
                Question: {prompt}
                
                Answer the question thoroughly based on the provided context. When referencing information, mention which document it came from (e.g., "According to Document 1..."). 
                
                If the information in the context is insufficient to answer the question completely, say what you can based on the available information and note what's missing. Use specific details from the documents when possible.
                """
            )
            
            full_response = response.content
            message_placeholder.write(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Show debug info in expander
        with st.expander("View Source Documents", expanded=False):
            st.write("🔍 Retrieved Document Chunks:")
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Document Chunk {i}**")
                st.text_area(f"Content", doc.page_content, height=200)
                if hasattr(doc, 'metadata') and doc.metadata:
                    st.write("Metadata:", doc.metadata)
                st.markdown("---")
