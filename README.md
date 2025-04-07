# 📄 AI Document Assistant

A powerful AI-powered document analysis tool that helps you extract, analyze, and interact with information from PDF documents using various AI models.

## ✨ Features

- 🤖 Multiple AI Provider Support
  - OpenAI (GPT-4)
  - DeepSeek
  - Claude
  - Groq
  - Mistral

- 📑 Document Processing
  - PDF text extraction
  - Smart text chunking
  - Efficient vector storage
  - Semantic search capabilities

- 💬 Interactive Chat Interface
  - Real-time Q&A about documents
  - Clean and modern UI
  - Persistent chat history
  - Source citations
  - Beautiful message formatting

- 🔒 Privacy & Security
  - API keys are never stored
  - Documents processed locally
  - Session-based storage
  - Secure data handling

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MayaDispeler/DOC-AI-APP.git
cd DOC-AI-APP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python -m streamlit run app.py
```

4. Open your browser and navigate to:
```
http://localhost:8501
```

## 📋 Usage

1. Select your preferred AI provider from the sidebar
2. Enter your API key for the selected provider
3. Upload a PDF document
4. Start asking questions about your document
5. View responses in a clean, formatted interface

## 🔧 Configuration

The application supports multiple AI providers:

- OpenAI (GPT-4)
- DeepSeek
- Claude
- Groq
- Mistral

Each provider requires its own API key, which can be obtained from their respective platforms.

## 🛠️ Technical Details

- Built with Streamlit for the web interface
- Uses LangChain for document processing
- FAISS for vector storage and similarity search
- PyPDF2 for PDF text extraction
- LiteLLM for unified API access to multiple AI providers

## 📝 Dependencies

- streamlit
- langchain
- pypdf2
- python-dotenv
- openai
- anthropic
- litellm
- faiss-cpu

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [LangChain](https://langchain.org/)
- Vector search by [FAISS](https://github.com/facebookresearch/faiss) 