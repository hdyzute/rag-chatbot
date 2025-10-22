# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that combines document retrieval with large language models to provide accurate, context-aware responses.

## Features

- 📄 Document ingestion and processing
- 🔍 Semantic search using vector embeddings
- 💬 Conversational AI interface
- 🐳 Docker containerization for easy deployment
- 🎨 Web-based user interface

## Tech Stack

- **Backend**: Python, Flask/FastAPI
- **Vector Database**: ChromaDB
- **LLM Integration**: OpenAI/Anthropic/Local models
- **Frontend**: HTML, CSS, JavaScript
- **Containerization**: Docker, Docker Compose

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (optional)
- API keys for LLM services (if using cloud providers)

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Run the application:
```bash
python app.py
```

6. Open your browser and navigate to `http://localhost:5000`

### Docker Setup

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. Access the application at `http://localhost:5000`

## Usage

1. **Upload Documents**: Add your documents through the web interface
2. **Ask Questions**: Type your questions in the chat interface
3. **Get Answers**: The chatbot will retrieve relevant information and generate responses

## Project Structure

```
rag-chatbot/
├── app.py                 # Main application file
├── index.html            # Frontend interface
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── compose.yaml         # Docker Compose setup
├── .dockerignore        # Docker ignore rules
├── .gitignore          # Git ignore rules
└── chroma_dbs/         # Vector database storage (gitignored)
```

## Configuration

Edit `.env` file to configure:
- API keys for LLM services
- Database settings
- Server port and host
- Model parameters

## Development

### Debug Mode

Run with debug mode enabled:
```bash
docker-compose -f compose.debug.yaml up
```

### Adding New Features

1. Create a new branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## Troubleshooting

**Issue**: ChromaDB connection errors
- **Solution**: Ensure the database directory exists and has proper permissions

**Issue**: API key errors
- **Solution**: Verify your API keys in the `.env` file

**Issue**: Docker build fails
- **Solution**: Check Docker daemon is running and you have sufficient disk space

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ChromaDB for vector storage
- OpenAI/Anthropic for LLM capabilities
- The open-source community
