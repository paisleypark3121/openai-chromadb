import os
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

def vectordb_exists(persist_directory):
    return os.path.exists(persist_directory)

def create_vectordb(filename,persist_directory,embedding):
    if vectordb_exists(persist_directory)==False:
        print("creating vectordb")
        loader = TextLoader(filename,encoding="utf-8")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # persist to vectordb: in a notebook, we should call persist() to ensure the embeddings are written to disk
        # This isn't necessary in a script: the database will be automatically persisted when the client object is destroyed
        return Chroma.from_documents(
            documents=texts, 
            embedding=embedding, 
            persist_directory=persist_directory
        )
    else:
        return load_vectordb(persist_directory=persist_directory,embedding=embedding)

def load_vectordb(persist_directory,embedding):
    if vectordb_exists(persist_directory):
        print("vectordb already exists")
        return Chroma(persist_directory=persist_directory, embedding_function=embedding)
    else:
        raise ValueError(f"VectorDB does not exist in {persist_directory}")

def do_query(vectordb,llm,query):
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectordb.as_retriever()
    )    
    return qa.run(query)

load_dotenv()

filename="state_of_the_union.txt"
persist_directory = 'chroma'
embedding = OpenAIEmbeddings()

vectordb=create_vectordb(
    filename=filename,
    persist_directory=persist_directory,
    embedding=embedding)

llm=OpenAI()
query = "What did the president say about Ketanji Brown Jackson"

response=do_query(
    vectordb=vectordb,
    llm=llm,
    query=query)
print(response)