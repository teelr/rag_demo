from langchain.document_loaders import TextLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def run_rag(question: str = "What is LangChain?"):
    loader = TextLoader("src/data/sample.txt")
    documents = loader.load()

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory="rag_demo_store")
    vectordb.persist()

    llm = Ollama(model="llama3")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    result = qa.run(question)
    print(f"\nQ: {question}\nA: {result}")
