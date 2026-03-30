# RAG Project

RAG with conversation memory built with LangChain, ChromaDB, and Groq.

## Setup

1. Clone the repository
2. Create a virtual environment: `python3 -m venv venv`
3. Activate it: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create `.env` file and add your Groq API key:
```
   GROQ_API_KEY=your_key_here
```
6. Add your PDF to the `data/` folder
7. Run: `python3 rag.py`