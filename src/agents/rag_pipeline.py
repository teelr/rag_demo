from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM


def run_rag(question: str):
    # Establish base path
    base_dir = Path(__file__).resolve().parent.parent  # -> /home/rich/projects/rag_demo
    data_path = base_dir / "data" / "context.txt"
    persist_dir = base_dir / "rag_demo_store"

    # Load and split documents
    loader = TextLoader(file_path=str(data_path))
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Create vector DB with embeddings from neurX
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://100.82.174.94:11434"
    )
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=str(persist_dir))  # noqa: E501

    # Set up LLM and retriever
    llm = OllamaLLM(
        model="llama3",
        base_url="http://100.82.174.94:11434"
    )
    retriever = vectordb.as_retriever()

    # Optional custom prompt
    prompt_template = PromptTemplate.from_template(
        "Use the context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}"
    )

    # QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True,
    )

    # Run query
    result = qa_chain.invoke({"query": question})
    answer = result.get("result", "[No answer returned]")

    print("=" * 50)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("=" * 50)
