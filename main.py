import os
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from chromadb_manager import *


def simple_query_from_file():
    '''
    Performs the query using the basic method
    '''
    filename="./files/state_of_the_union.txt"
    persist_directory = 'chroma'
    embedding = OpenAIEmbeddings()
    overwrite=True

    vectordb=create_vectordb_from_file(
        filename=filename,
        persist_directory=persist_directory,
        embedding=embedding,
        overwrite=overwrite)

    llm=OpenAI()
    query = "What did the president say about Ketanji Brown Jackson"

    response=do_query(
        vectordb=vectordb,
        llm=llm,
        query=query)
    print(response)

def chain_query_from_file():
    '''
    Performs the query using the chain method
    '''
    filename="./files/state_of_the_union.txt"
    persist_directory = 'chroma'
    embedding = OpenAIEmbeddings()
    overwrite=True
    
    vectordb=create_vectordb_from_file(
        filename=filename,
        persist_directory=persist_directory,
        embedding=embedding,
        overwrite=overwrite)

    llm=OpenAI()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": vectordb.as_retriever(), "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    query = "What did the president say about Ketanji Brown Jackson"
    response=chain.invoke(query)
    print(response)

def simple_query_from_youtube():
    url="https://www.youtube.com/watch?v=KjOUQBzl2Ug&list=PLc53cT1vE-OLHeGIFHwLvMqNqRm4BH6Jw"
    language_code="it"
    local_file=save_remote_file(
        url=url,
        language_code=language_code
    )
    
    persist_directory = 'chroma'
    embedding = OpenAIEmbeddings()
    overwrite=True
    vectordb=create_vectordb_from_file(
        filename=local_file,
        persist_directory=persist_directory,
        embedding=embedding,
        overwrite=True
    )

    llm=OpenAI()
    query = "What is an Access List?"

    response=do_query(
        vectordb=vectordb,
        llm=llm,
        query=query)
    print(response)

def simple_query_from_texts():
    persist_directory = 'chroma'
    embedding = OpenAIEmbeddings()
    overwrite=True

    texts=[
        "Prince is a great musician", 
        "He is a multi-instrumentalist", 
        "We produced, arranged, composed and performed most of his albums", 
        "My favorite album is The Gold Experience", 
        "He was born in 1958"]

    vectordb=create_vectordb_from_texts(
        texts=texts,
        persist_directory=persist_directory,
        embedding=embedding,
        overwrite=overwrite)

    llm=OpenAI()
    query = "When did Prince was born?"

    response=do_query(
        vectordb=vectordb,
        llm=llm,
        query=query)
    print(response)

def chain_query_from_texts():
    persist_directory = 'chroma'
    embedding = OpenAIEmbeddings()
    overwrite=True

    texts=[
        "Prince is a great musician", 
        "He is a multi-instrumentalist", 
        "We produced, arranged, composed and performed most of his albums", 
        "My favorite album is The Gold Experience", 
        "He was born in 1958"]

    vectordb=create_vectordb_from_texts(
        texts=texts,
        persist_directory=persist_directory,
        embedding=embedding,
        overwrite=overwrite)

    llm=OpenAI()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": vectordb.as_retriever(), "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    query = "When did Prince was born?"
    response=chain.invoke(query)
    print(response)

def multiple_files():
    urls = [
        "https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/PT719-Transcript.pdf", 
        "https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/PT717-Transcript.pdf",
        "https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/PT715-Transcript.pdf",
        "https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/PT713-Transcript.pdf"
    ]
    downloaded_files = []
    for url in urls:
        local_file = save_remote_file(url)
        downloaded_files.append(local_file)
    
    persist_directory = 'chroma'
    embedding = OpenAIEmbeddings()
    overwrite=True
    
    vectordb=create_vectordb_from_files(
        files=downloaded_files, 
        persist_directory=persist_directory, 
        embedding=embedding,
        overwrite=overwrite
    )

    llm=OpenAI()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": vectordb.as_retriever(), "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    query = "What is ToolFormer?"
    response=chain.invoke(query)
    print(response)


load_dotenv()

#chain_query_from_text_file()
#simple_query_from_texts()
#chain_query_from_texts()
#local_file=save_remote_file("https://www.gutenberg.org/cache/epub/1934/pg1934.txt")
#simple_query_from_youtube()
# multiple_files()