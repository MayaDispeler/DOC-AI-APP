import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import litellm
from streamlit_extras.row import row
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import os
from typing import List, Dict, Any, Optional
import tempfile
import time

# ======================
# APP CONFIGURATION
# ======================
st.set_page_config(
    page_title="📄 AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        position: relative;
    }
    .chat-message.user {
        background-color: #f0f2f6;
        margin-left: 50px;
        margin-right: 100px;
    }
    .chat-message.assistant {
        background-color: #ffffff;
        border: 1px solid #e0e3e9;
        margin-right: 50px;
        margin-left: 100px;
    }
    .chat-message .avatar {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
    }
    .chat-message.user .avatar {
        left: -45px;
        background-color: #4a90e2;
        color: white;
    }
    .chat-message.assistant .avatar {
        right: -45px;
        background-color: #f0f2f6;
        color: #4a90e2;
    }
    .chat-message .message {
        color: #444654;
        line-height: 1.6;
        font-size: 1rem;
        margin: 0;
    }
    .chat-message .source {
        color: #666;
        font-size: 0.875rem;
        padding-top: 0.5rem;
        border-top: 1px solid #e0e3e9;
        margin-top: 0.5rem;
    }
    .chat-message .source::before {
        content: "💡";
        margin-right: 5px;
    }
    .sidebar .stButton > button {
        width: 100%;
    }
    /* Improve overall readability */
    .stMarkdown {
        color: #444654;
    }
    /* Table styling */
    .chat-message table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        background-color: white;
    }
    .chat-message table th,
    .chat-message table td {
        padding: 0.75rem;
        text-align: left;
        border: 1px solid #e0e3e9;
    }
    .chat-message table th {
        background-color: #f8f9fa;
        font-weight: 600;
        color: #2c3e50;
    }
    .chat-message table tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    .chat-message table tr:hover {
        background-color: #f0f2f6;
    }
    .chat-message.assistant .message {
        overflow-x: auto;
        max-width: 100%;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px;
        color: #444654;
        font-size: 16px;
        font-weight: 500;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    
    /* Badges for free/paid */
    .free-badge {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        margin-left: 5px;
    }
    .paid-badge {
        display: inline-block;
        background-color: #9C27B0;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        margin-left: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_doc' not in st.session_state:
    st.session_state.current_doc = None
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "paid"
    
# ======================
# HELPER FUNCTIONS
# ======================

def process_document_free(pdf, hf_token=None):
    """Process document using free models and return vector store"""
    with st.spinner("📑 Reading document..."):
        try:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            
            if not text.strip():
                st.error("⚠️ The PDF appears to be empty or contains no extractable text.")
                return None
            
            # For large documents, use optimized processing
            chunk_size = 1000
            chunk_overlap = 200
            
            if len(text) > 500000:  # ~500k characters
                st.warning("Large document detected - using optimized processing")
                chunk_size = 2000  # Bigger chunks
                chunk_overlap = 400
            
            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            # Create embeddings using HuggingFace
            if hf_token:
                embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-en",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                    huggingfacehub_api_token=hf_token
                )
            else:
                # Use default embedding if no token provided
                embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-en",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
            
            # Use FAISS for vector storage
            vector_store = FAISS.from_texts(chunks, embeddings)
            
            # Clean up memory
            st.cache_data.clear()
            
            return vector_store
            
        except Exception as e:
            st.error(f"⚠️ Error processing document: {str(e)}")
            return None

def process_document_paid(pdf, api_key):
    """Process document using paid models and return vector store"""
    with st.spinner("📑 Reading document..."):
        try:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            
            if not text.strip():
                st.error("⚠️ The PDF appears to be empty or contains no extractable text.")
                return None
            
            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            # Create embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vector_store = FAISS.from_texts(chunks, embeddings)
            
            return vector_store
            
        except Exception as e:
            st.error(f"⚠️ Error processing document: {str(e)}")
            return None

# ======================
# SIDEBAR CONFIGURATION
# ======================
with st.sidebar:
    colored_header(
        label="Document Assistant",
        description="Powered by AI",
        color_name="violet-70"
    )
    
    add_vertical_space(2)
    
    # Provider Selection based on the current mode
    if st.session_state.current_mode == "paid":
        provider = st.selectbox(
            "Select AI Provider",
            ("OpenAI", "DeepSeek", "Claude", "Groq", "Mistral"),
            index=0,
            help="Choose which AI service to power your document analysis"
        )
        
        # API Key Input
        api_key = st.text_input(
            f"Enter your {provider} API Key",
            type="password",
            help="Key is used only for this session and never stored"
        )
        
        # Provider URL Helper
        def get_provider_url(provider):
            urls = {
                "OpenAI": "https://platform.openai.com/api-keys",
                "DeepSeek": "https://platform.deepseek.com",
                "Claude": "https://console.anthropic.com/settings/keys",
                "Groq": "https://console.groq.com/keys",
                "Mistral": "https://console.mistral.ai/api-keys"
            }
            return urls.get(provider, "")
        
        if st.button("❓ Get API Key"):
            st.markdown(f"[Click here to get your {provider} API key]({get_provider_url(provider)})")
    
    else:  # Free mode
        provider = st.selectbox(
            "Select Free Model",
            ("Groq (Llama 3 8B)", "Hugging Face", "Ollama (Local)"),
            index=0,
            help="Choose which free AI service to power your document analysis"
        )
        
        # Different inputs based on provider
        if provider == "Groq (Llama 3 8B)":
            api_key = st.text_input(
                "Enter Groq API Key (Free Tier)",
                type="password",
                help="Optional - Get 50 req/min for free"
            )
            st.markdown("[Get a free Groq API key](https://console.groq.com/keys)")
        
        elif provider == "Hugging Face":
            api_key = st.text_input(
                "Enter Hugging Face Token",
                type="password",
                help="Optional - Some models work without a token"
            )
            st.markdown("[Get a free Hugging Face token](https://huggingface.co/settings/tokens)")
        
        else:  # Ollama
            st.info("No API key needed for Ollama. Make sure you've installed Ollama locally")
            api_key = "not_needed"
            st.markdown("""
            To install Ollama locally:
            ```bash
            curl -fsSL https://ollama.com/install.sh | sh
            ollama pull mistral
            ```
            """)
    
    add_vertical_space(2)
    
    # Document Upload
    st.markdown("### 📄 Upload Document")
    pdf = st.file_uploader("Upload a PDF file", type="pdf")
    
    if pdf:
        if st.button("🔄 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    add_vertical_space(2)
    
    # Instructions
    with st.expander("ℹ️ How to use"):
        if st.session_state.current_mode == "paid":
            st.markdown("""
            1. **Select AI Provider** from the dropdown
            2. **Enter API Key** (click ❓ for help)
            3. **Upload PDF** document
            4. **Ask Questions** in the chat
            
            🔒 **Privacy**: Your API key and documents are processed only in your browser session.
            """)
        else:
            st.markdown("""
            1. **Select Free Model** from the dropdown
            2. **Enter API Key/Token** if needed
            3. **Upload PDF** document
            4. **Ask Questions** in the chat
            
            ⚠️ **Note**: Free models may be slower or less accurate than paid versions.
            """)

# ======================
# MAIN CONTENT AREA
# ======================

# Create tabs for Free and Paid versions
tab1, tab2 = st.tabs([
    "🆓 Free Version", 
    "💎 Premium Version"
])

# Handle tab selection
with tab1:
    # Set the current mode when this tab is viewed
    if st.session_state.current_mode != "free":
        st.session_state.current_mode = "free"
        st.session_state.vector_store = None
        st.session_state.document_processed = False
        st.session_state.chat_history = []
        st.session_state.current_doc = None
        st.experimental_rerun()
        
    # Free Version UI
    if not pdf:
        # Welcome screen for free version
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #4CAF50; font-size: 2.5rem; margin-bottom: 1rem;">📄 AI Document Assistant <span class="free-badge">FREE</span></h1>
            <p style="font-size: 1.2rem; color: #444654; margin-bottom: 2rem;">Process and analyze documents using open-source models</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights for free version
        col1, col2 = st.columns([3, 2], gap="large")
        
        with col1:
            st.image("image.png", use_container_width=True)
        
        with col2:
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #4CAF50; height: 100%;">
                <h3 style="color: #4CAF50; margin-bottom: 1.5rem;">🚀 Free Version Features</h3>
                <ul style="margin-left: 1rem; font-size: 1rem; color: #444654; line-height: 1.8;">
                    <li><strong>Embeddings:</strong> BAAI/bge-small-en (Hugging Face)</li>
                    <li><strong>LLM:</strong> Llama 3 8B (Groq) or Mistral 7B</li>
                    <li><strong>Vector DB:</strong> FAISS (local processing)</li>
                    <li><strong>Large Documents:</strong> Optimized for 1000+ pages</li>
                    <li><strong>No Credit Card Required</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Add feature section with better spacing
        st.markdown("<hr style='margin: 2.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style="color: #4CAF50; margin-bottom: 1.5rem; text-align: center;">⚡ Performance Tips</h3>
        """, unsafe_allow_html=True)
        
        # Create three columns with proper spacing
        feature_cols = st.columns(3, gap="medium")
        
        # Define feature card style for consistency
        card_style = """
        background-color: white; 
        padding: 1.5rem; 
        border-radius: 12px; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.08); 
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        """
        
        # Feature Cards with consistent styling
        features = [
            {
                "icon": "📊",
                "title": "Memory Optimization",
                "description": "Automatically adapts chunk size for large documents to balance performance and accuracy"
            },
            {
                "icon": "⚡",
                "title": "Processing Speed",
                "description": "Groq's free tier offers fast processing, while local Ollama works offline but slower"
            },
            {
                "icon": "🔍",
                "title": "Document Size",
                "description": "Handles large documents (1000+ pages) with optimized memory management"
            }
        ]
        
        # Render feature cards with consistent structure
        for i, feature in enumerate(features):
            with feature_cols[i]:
                st.markdown(f"""
                <div style="{card_style}">
                    <div style="font-size: 2.2rem; margin-bottom: 1rem;">{feature["icon"]}</div>
                    <h4 style="margin-bottom: 1rem; color: #2c3e50;">{feature["title"]}</h4>
                    <p style="color: #666; font-size: 0.95rem; line-height: 1.5;">{feature["description"]}</p>
                </div>
                """, unsafe_allow_html=True)
    
    elif pdf and (api_key or provider == "Ollama (Local)" or provider == "Hugging Face"):
        # Reset vector store if new document is uploaded
        if st.session_state.current_doc != pdf.name:
            st.session_state.vector_store = None
            st.session_state.document_processed = False
            st.session_state.current_doc = pdf.name
            st.session_state.chat_history = []

        # Document Processing for free version
        if not st.session_state.vector_store:
            # Process with free models
            st.session_state.vector_store = process_document_free(pdf, api_key if provider == "Hugging Face" else None)
            if st.session_state.vector_store:
                st.session_state.document_processed = True
            else:
                st.stop()
        
        # Configure LLM based on free provider
        litellm.drop_params = True
        
        # Display Chat History
        st.markdown("### 💬 Chat History")
        for message in st.session_state.chat_history:
            with st.container():
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user">
                        <div class="message">{message["content"]}</div>
                        <div class="avatar">👤</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Clean and format the message content
                    content = message["content"]
                    # Remove any HTML artifacts
                    content = content.replace("</div>", "").replace("<div>", "").replace("</div", "").strip()
                    if content.endswith(">"):
                        content = content.rsplit(">", 1)[0].strip()
                    
                    # Format as table if it contains DIN information
                    if "following" in content.lower() and any(x in content.lower() for x in ["din", "director"]):
                        lines = content.split('\n')
                        formatted_content = []
                        for line in lines:
                            line = line.strip()
                            if '-' in line and not line.startswith('|'):
                                # Format as table row
                                name, din = line.split('-', 1)
                                formatted_content.append(f"| {name.strip()} | {din.strip()} |")
                            elif line and not line.startswith('|'):
                                formatted_content.append(line)
                        
                        if formatted_content:
                            content = "| Name | DIN |\n|------|-----|\n" + "\n".join(formatted_content)

                    st.markdown(f"""
                    <div class="chat-message assistant">
                        <div class="message">{content}</div>
                        <div class="avatar">🤖</div>
                        {f'<div class="source">{message.get("source", "No source context available")}</div>' if message.get("source") else ""}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat Input
        query = st.chat_input("Ask a question about the document...")
        
        if query:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            with st.spinner("🤔 Thinking..."):
                try:
                    # Configure LLM based on free provider
                    if provider == "Groq (Llama 3 8B)":
                        if api_key:
                            llm = litellm.completion(
                                model="groq/llama3-8b-8192",
                                messages=[{"role": "user", "content": query}],
                                api_key=api_key
                            )
                        else:
                            st.error("⚠️ Groq API key is required")
                            st.stop()
                    
                    elif provider == "Hugging Face":
                        # Use Hugging Face for completions
                        from langchain_community.llms import HuggingFaceEndpoint
                        
                        if api_key:
                            llm = HuggingFaceEndpoint(
                                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                                huggingfacehub_api_token=api_key,
                            )
                        else:
                            # Fallback to a model that doesn't require token
                            llm = HuggingFaceEndpoint(
                                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                            )
                    
                    else:  # Ollama
                        # Use local Ollama installation
                        from langchain_community.chat_models import ChatOllama
                        
                        llm = ChatOllama(model="mistral")
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vector_store.as_retriever()
                    )
                    
                    result = qa_chain.invoke({"query": query})
                    
                    # Clean and format the response
                    response_text = result["result"]
                    # Remove any HTML tags and clean up the text
                    response_text = response_text.replace("</div>", "").replace("<div>", "").replace("</div", "").strip()
                    
                    # Add AI response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "source": result.get("source_text", "")
                    })
                    
                    # Rerun to update the chat display
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}. Please check your inputs and try again.")
    else:
        st.warning("👆 Please provide a PDF document to start chatting. For some providers, an API key is required.")

with tab2:
    # Set the current mode when this tab is viewed
    if st.session_state.current_mode != "paid":
        st.session_state.current_mode = "paid"
        st.session_state.vector_store = None
        st.session_state.document_processed = False
        st.session_state.chat_history = []
        st.session_state.current_doc = None
        st.experimental_rerun()
        
    # Paid Version UI - original code
    if not pdf:
        # Create a clean, modern welcome page with local image
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #9C27B0; font-size: 2.5rem; margin-bottom: 1rem;">📄 AI Document Assistant <span class="paid-badge">PREMIUM</span></h1>
            <p style="font-size: 1.2rem; color: #444654; margin-bottom: 2rem;">Your intelligent companion for document analysis and exploration</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Use columns for better layout
        col1, col2 = st.columns([3, 2], gap="large")
        
        with col1:
            # Fix deprecated parameter
            st.image("image.png", use_container_width=True)
        
        with col2:
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #9C27B0; height: 100%;">
                <h3 style="color: #9C27B0; margin-bottom: 1.5rem;">🚀 Premium Features</h3>
                <ul style="margin-left: 1rem; font-size: 1rem; color: #444654; line-height: 1.8;">
                    <li><strong>Advanced AI Models:</strong> GPT-4, Claude, more</li>
                    <li><strong>Superior Accuracy:</strong> Better reasoning</li>
                    <li><strong>State-of-the-art:</strong> Latest AI capabilities</li>
                    <li><strong>Faster Processing:</strong> Optimized performance</li>
                    <li><strong>Enhanced Analysis:</strong> More detailed insights</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Add feature section with better spacing
        st.markdown("<hr style='margin: 2.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style="color: #9C27B0; margin-bottom: 1.5rem; text-align: center;">✨ Premium AI Providers</h3>
        """, unsafe_allow_html=True)
        
        # Create three columns with proper spacing
        feature_cols = st.columns(3, gap="medium")
        
        # Define feature card style for consistency
        card_style = """
        background-color: white; 
        padding: 1.5rem; 
        border-radius: 12px; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.08); 
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        """
        
        # Feature Cards with consistent styling
        features = [
            {
                "icon": "🌟",
                "title": "OpenAI GPT-4",
                "description": "State-of-the-art reasoning and comprehension for detailed document analysis"
            },
            {
                "icon": "🤖",
                "title": "Anthropic Claude",
                "description": "Excellent for long document context and nuanced understanding"
            },
            {
                "icon": "⚡",
                "title": "Groq & Mistral",
                "description": "Lightning-fast responses and cutting-edge AI capabilities"
            }
        ]
        
        # Render feature cards with consistent structure
        for i, feature in enumerate(features):
            with feature_cols[i]:
                st.markdown(f"""
                <div style="{card_style}">
                    <div style="font-size: 2.2rem; margin-bottom: 1rem;">{feature["icon"]}</div>
                    <h4 style="margin-bottom: 1rem; color: #2c3e50;">{feature["title"]}</h4>
                    <p style="color: #666; font-size: 0.95rem; line-height: 1.5;">{feature["description"]}</p>
                </div>
                """, unsafe_allow_html=True)

    elif pdf and api_key:
        # Reset vector store if new document is uploaded
        if st.session_state.current_doc != pdf.name:
            st.session_state.vector_store = None
            st.session_state.document_processed = False
            st.session_state.current_doc = pdf.name
            st.session_state.chat_history = []

        # Document Processing for paid version
        if not st.session_state.vector_store:
            st.session_state.vector_store = process_document_paid(pdf, api_key)
            if st.session_state.vector_store:
                st.session_state.document_processed = True
            else:
                st.stop()
        
        # Configure AI Models
        litellm.drop_params = True
        model_map = {
            "OpenAI": "gpt-4-turbo-preview",  # Using the latest GPT-4 model
            "DeepSeek": "deepseek-chat",
            "Claude": "claude-3-haiku-20240307",
            "Groq": "groq/llama2-70b-4096",
            "Mistral": "mistral/mistral-large-latest"
        }
        
        # Display Chat History
        st.markdown("### 💬 Chat History")
        for message in st.session_state.chat_history:
            with st.container():
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user">
                        <div class="message">{message["content"]}</div>
                        <div class="avatar">👤</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Clean and format the message content
                    content = message["content"]
                    # Remove any HTML artifacts
                    content = content.replace("</div>", "").replace("<div>", "").replace("</div", "").strip()
                    if content.endswith(">"):
                        content = content.rsplit(">", 1)[0].strip()
                    
                    # Format as table if it contains DIN information
                    if "following" in content.lower() and any(x in content.lower() for x in ["din", "director"]):
                        lines = content.split('\n')
                        formatted_content = []
                        for line in lines:
                            line = line.strip()
                            if '-' in line and not line.startswith('|'):
                                # Format as table row
                                name, din = line.split('-', 1)
                                formatted_content.append(f"| {name.strip()} | {din.strip()} |")
                            elif line and not line.startswith('|'):
                                formatted_content.append(line)
                        
                        if formatted_content:
                            content = "| Name | DIN |\n|------|-----|\n" + "\n".join(formatted_content)

                    st.markdown(f"""
                    <div class="chat-message assistant">
                        <div class="message">{content}</div>
                        <div class="avatar">🤖</div>
                        {f'<div class="source">{message.get("source", "No source context available")}</div>' if message.get("source") else ""}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat Input
        query = st.chat_input("Ask a question about the document...")
        
        if query:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            with st.spinner("🤔 Thinking..."):
                try:
                    # Configure LLM based on provider
                    if provider == "OpenAI":
                        llm = ChatOpenAI(
                            model_name=model_map[provider],
                            openai_api_key=api_key,
                            temperature=0.7
                        )
                    else:
                        llm = litellm.completion(
                            model=f"{provider.lower()}/{model_map[provider]}",
                            messages=[{"role": "user", "content": query}],
                            api_key=api_key
                        )
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vector_store.as_retriever()
                    )
                    
                    result = qa_chain.invoke({"query": query})
                    
                    # Clean and format the response
                    response_text = result["result"]
                    # Remove any HTML tags and clean up the text
                    response_text = response_text.replace("</div>", "").replace("<div>", "").replace("</div", "").strip()
                    
                    # Add AI response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "source": result.get("source_text", "")
                    })
                    
                    # Rerun to update the chat display
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}. Please check your API key and try again.")
    else:
        st.warning("👆 Please provide both a PDF document and an API key to start chatting.")
