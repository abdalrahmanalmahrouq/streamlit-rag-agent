import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load the environment variables from the .env file automatically!
load_dotenv() 

# Notice: pdf_path is now optional (=None), and api_key is gone!
def get_rag_chain(pdf_path=None):
    
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db_folder = "./chroma_db_storage"

    # If the UI passes a PDF, build a new database
    if pdf_path:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        chunks = text_splitter.split_documents(pages)
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=db_folder 
        )
    # If the UI doesn't pass a PDF, just load the existing database from the hard drive
    else:
        vectorstore = Chroma(
            persist_directory=db_folder, 
            embedding_function=embeddings_model
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    rag_template = """Use the following context to answer the question.
    Context: {context}
    Question: {question}
    Answer:"""
    rag_prompt = PromptTemplate.from_template(rag_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return chain