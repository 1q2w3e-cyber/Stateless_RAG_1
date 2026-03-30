# RAG project main script
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. 
Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the provided document."

Context: {context}
Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\nRAG system ready! Ask questions about your document.")
print("Type 'exit' to quit\n")

while True:
    question = input("Your question: ")
    
    if question.lower() == "exit":
        break
    
    response = qa_chain.invoke(question)
    print(f"\nAnswer: {response}\n")