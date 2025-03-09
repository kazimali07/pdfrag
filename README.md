# Ask Mr. Data Modeler 🤖

An intelligent AI-powered assistant that helps you understand and analyze data models through natural conversation. Built with Streamlit and powered by OpenAI's language models.

## 🌟 Features

- **PDF Document Processing**: Automatically extracts and processes text from uploaded PDF files
- **Conversational Interface**: Natural dialogue with context-aware responses
- **Vector Database**: Uses ChromaDB with OpenAI embeddings for efficient storage/retrieval
- **Streaming Responses**: Real-time answer generation with typing indicator
- **Session Persistence**: Maintains conversation history and document context
- **Debug Tools**: Inspect retrieved document chunks during conversations

## 🚀 Quick Start

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/pdfrag.git
   cd pdfrag
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   echo 'OPENAI_API_KEY="your-api-key-here"' > .env
   ```

4. **Organize PDFs**
   ```bash
   mkdir -p data  # Place PDF files here
   ```

5. **Launch Application**
   ```bash
   streamlit run app.py
   ```

## 📋 Requirements

- Python 3.10+
- OpenAI API key
- PDF documents in `data/` directory

## 🔧 Technical Stack

- **Framework**: Streamlit
- **Language Model**: GPT-4 via OpenAI API
- **Embeddings**: text-embedding-3-large
- **Vector Database**: ChromaDB
- **PDF Processing**: PyPDFDirectoryLoader

## 🛠️ Implementation Details

The application follows these key steps:

1. **Document Ingestion**:
   - Load PDFs from `data/` directory
   - Split documents into 2000-character chunks
   - Generate vector embeddings for each chunk

2. **Vector Storage**:
   - Store embeddings in ChromaDB with persistent storage
   - Enable efficient similarity search

3. **Conversation Flow**:
   - Maintain chat history in session state
   - Combine retrieved documents with LLM context
   - Stream responses character-by-character

4. **Error Handling**:
   - Validate PDF directory existence
   - Catch and display processing errors
   - Graceful handling of missing credentials

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📝 License

This project is licensed under the terms included in the LICENSE file.

## 🆘 Troubleshooting

- Ensure all dependencies are correctly installed
- Check that your OpenAI API key is valid and properly set in the `.env` file
- For PDF processing issues, ensure your PDFs are text-based and not scanned images
- If the vector store isn't loading, check file permissions and storage space
