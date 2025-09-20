# Lore Weaver Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) chatbot specializing in geomythological storytelling, built with LangChain and integrated with Cultural Knowledge Graph (CKG) and PostGIS databases.

## 🌟 Features

- **RAG-powered storytelling** using LangChain for contextual narrative generation
- **Multi-database integration** with ArangoDB (CKG) and PostgreSQL/PostGIS
- **Explainable AI** with LangSmith integration for tracing and debugging
- **Reinforcement learning feedback** loops for continuous improvement
- **Creative prompt engineering** for evocative mythological narratives
- **RESTful API backend** built with FastAPI
- **Modern web interface** with real-time chat functionality
- **Vector embeddings** using Sentence Transformers and ChromaDB/FAISS

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   FastAPI       │    │   LangChain     │
│   (HTML/CSS/JS) │◄──►│   Backend       │◄──►│   RAG Pipeline  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   ArangoDB      │    │   PostgreSQL    │
                       │   (CKG)         │    │   (PostGIS)     │
                       └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+ (for React interface, optional)
- ArangoDB instance
- PostgreSQL with PostGIS extension
- OpenAI API key
- LangSmith account (optional, for explainable AI)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to the terra-constellata project
cd terra-constellata

# Install Python dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-openai-api-key"
export USE_LANGSMITH="true"  # Optional
export LANGCHAIN_PROJECT="lore-weaver-chatbot"  # Optional
```

### 2. Database Setup

Ensure your databases are running and configured:

- **ArangoDB**: Default connection `http://localhost:8529`
- **PostgreSQL/PostGIS**: Default connection `localhost:5432`

Update connection settings in:
- `data/ckg/connection.py`
- `data/postgis/connection.py`

### 3. Initialize Databases

```bash
# Initialize CKG schema
python -c "from data.ckg.schema import create_collections; create_collections()"

# Initialize PostGIS schema
python -c "from data.postgis.schema import initialize_database; db = __import__('data.postgis.connection').PostGISConnection(); initialize_database(db)"
```

### 4. Load Data

```python
from chatbot.rag.rag_pipeline import LoreWeaverRAG

# Initialize RAG system
rag = LoreWeaverRAG()

# Load data from databases into vector store
rag.load_data()
```

### 5. Start the Backend

```bash
# Start FastAPI server
python -m uvicorn chatbot.backend:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Launch the Interface

Open `interfaces/web/index.html` in your browser, or serve it with a web server:

```bash
# Using Python's built-in server
cd interfaces/web
python -m http.server 3000
```

Then visit `http://localhost:3000`

## 🎯 Usage

### API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /chat` - Send chat query
- `POST /feedback` - Submit user feedback
- `GET /stats` - Get system statistics
- `POST /reload-data` - Reload data from databases

### Example API Usage

```python
import requests

# Send a query
response = requests.post("http://localhost:8000/chat",
    json={"question": "Tell me about mythological mountains", "max_results": 5}
)
result = response.json()
print(result["answer"])

# Submit feedback
requests.post("http://localhost:8000/feedback",
    json={
        "query": "Tell me about mythological mountains",
        "response": result["answer"],
        "rating": 5,
        "feedback": "Great storytelling!"
    }
)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_lore_weaver.py
```

This will test:
- Vector store initialization
- RAG pipeline setup
- Sample queries
- Feedback system
- Performance benchmarks

## 🎨 Customization

### Prompt Engineering

Modify storytelling prompts in `rag/rag_pipeline.py`:

```python
self.storytelling_prompt = PromptTemplate(
    template="Your custom prompt template here...",
    input_variables=["context", "question"]
)
```

### Model Configuration

Adjust model settings in the RAG initialization:

```python
rag = LoreWeaverRAG(
    model_name="gpt-4",  # Use GPT-4 for better quality
    temperature=0.8,     # Increase for more creativity
    use_langsmith=True   # Enable tracing
)
```

### Vector Store Options

Choose between ChromaDB (persistent) or FAISS (in-memory):

```python
# ChromaDB (recommended for production)
vector_store = LoreWeaverVectorStore(use_chroma=True)

# FAISS (faster for development)
vector_store = LoreWeaverVectorStore(use_chroma=False)
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | GPT model to use | `gpt-3.5-turbo` |
| `MODEL_TEMPERATURE` | Model creativity | `0.7` |
| `USE_LANGSMITH` | Enable LangSmith tracing | `true` |
| `LANGCHAIN_PROJECT` | LangSmith project name | `lore-weaver-chatbot` |
| `HOST` | API server host | `0.0.0.0` |
| `PORT` | API server port | `8000` |

### Database Configuration

Update connection settings in the respective connection files:

```python
# CKG Connection
get_db_connection(
    host='http://localhost:8529',
    username='root',
    password='',
    database='ckg_db'
)

# PostGIS Connection
PostGISConnection(
    host='localhost',
    port=5432,
    database='terra_constellata',
    user='postgres',
    password=''
)
```

## 📊 Monitoring & Analytics

### LangSmith Integration

When enabled, all interactions are traced in LangSmith for:
- Query analysis
- Response quality assessment
- Performance monitoring
- Debugging failed requests

### Feedback System

User feedback is collected and can be used for:
- Model fine-tuning
- Prompt optimization
- Quality improvement
- User preference analysis

## 🚀 Deployment

### Production Setup

1. **Database Setup**: Configure production ArangoDB and PostGIS instances
2. **Environment Variables**: Set all required environment variables
3. **SSL/TLS**: Enable HTTPS in production
4. **Load Balancing**: Deploy multiple instances behind a load balancer
5. **Monitoring**: Set up logging and monitoring solutions

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "chatbot.backend:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is part of the Terra Constellata project. See the main project license for details.

## 🆘 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Ensure databases are running
   - Check connection credentials
   - Verify network connectivity

2. **OpenAI API Errors**
   - Verify API key is valid
   - Check API quota/limits
   - Ensure proper environment variable setup

3. **Vector Store Errors**
   - Check disk space for ChromaDB
   - Verify embedding model availability
   - Ensure proper permissions for data directories

4. **LangSmith Issues**
   - Verify LangSmith credentials
   - Check network connectivity to LangSmith
   - Ensure project name is correct

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Further Reading

- [LangChain Documentation](https://python.langchain.com/)
- [ArangoDB Documentation](https://www.arangodb.com/docs/)
- [PostGIS Documentation](https://postgis.net/documentation/)
- [LangSmith Guide](https://docs.smith.langchain.com/)

---

*Built with ❤️ for the Terra Constellata project*