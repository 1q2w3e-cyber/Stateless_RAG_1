# RAG project with Conversational Memory
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# With this:
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector DB
if os.path.exists("chroma_db"):
    print("Loading existing vector store...")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )
else:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF found in {data_dir}")

    pdf_path = os.path.join(data_dir, pdf_files[0])
    print(f"Using PDF: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )
    splits = text_splitter.split_documents(documents)

    print("Creating new vector store...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")

# Step 1 — Rephrase follow-up questions as standalone
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given chat history and latest question, reformulate a standalone question. Return it as is if already standalone."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Step 2 — History aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

# Step 3 — Answer prompt
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the context below. If unsure, say you don't know.\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Step 4 — Full RAG chain
combine_docs_chain = create_stuff_documents_chain(llm, answer_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

# Step 5 — Session memory store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Step 6 — Wrap chain with memory
conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Step 7 — Query loop
print("\n Conversational RAG ready!")
print("Type 'exit' to quit\n")

while True:
    question = input("You: ")
    if question.lower() == "exit":
        break
    response = conversational_rag.invoke(
        {"input": question},
        config={"configurable": {"session_id": "user_1"}}
    )
    print(f"\nAssistant: {response['answer']}\n")