import os
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from chromadb_manager import *

from langchain.chains.conversation.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory
)

from langchain.agents import (
    AgentExecutor,
    AgentType,
    initialize_agent,
    Tool
)

from langchain.agents.agent_toolkits import (
    create_retriever_tool,
    create_conversational_retrieval_agent
)

from langchain.chains import ConversationalRetrievalChain

from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)



def simple_query_from_file():
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

    #query = "What did the president say about Ketanji Brown Jackson"

    try:
        print("\n***WELCOME***\n")
        while True:
            query = input("\nUser: ")
            response=do_query(
                vectordb=vectordb,
                llm=llm,
                query=query)
            print(f"Assistant: {response}")
    except KeyboardInterrupt:
        print("BYE BYE!!!")

def conversation_from_file():
    print("TODO")

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
    local_file=save_file(
        location=url,
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

def multiple_files(language_code='en'):
    locations = [
        "https://www.youtube.com/watch?v=8o9y8HGgqjw&list=PLc53cT1vE-OLOLNrOQiFbbSwHybkQov3L&index=1", 
        "https://www.youtube.com/watch?v=5olNe6ZlKB8&list=PLc53cT1vE-OLOLNrOQiFbbSwHybkQov3L&index=2",
        "https://www.youtube.com/watch?v=Kf6uz9UO-vA&list=PLc53cT1vE-OLOLNrOQiFbbSwHybkQov3L&index=3",
        "./files/DHCP online - v1.0.pdf",
        "./files/DHCP v2.0.pdf",
        "./files/dhcp.txt"
    ]
    downloaded_files = []
    for location in locations:
        local_file = save_file(
            location,
            language_code=language_code)
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

    try:
        print("\n***WELCOME***\n")
        while True:
            query = input("\nUser: ")
            response=chain.invoke(query)
            print(f"Assistant: {response}")
    except KeyboardInterrupt:
        print("BYE BYE!!!")

def conversation_multiple_files(language_code='en'):
    locations = [
        "https://www.youtube.com/watch?v=8o9y8HGgqjw&list=PLc53cT1vE-OLOLNrOQiFbbSwHybkQov3L&index=1", 
        "https://www.youtube.com/watch?v=5olNe6ZlKB8&list=PLc53cT1vE-OLOLNrOQiFbbSwHybkQov3L&index=2",
        "https://www.youtube.com/watch?v=Kf6uz9UO-vA&list=PLc53cT1vE-OLOLNrOQiFbbSwHybkQov3L&index=3",
        "./files/DHCP online - v1.0.pdf",
        "./files/DHCP v2.0.pdf",
        "./files/dhcp.txt"
    ]
    downloaded_files = []
    for location in locations:
        local_file = save_file(
            location,
            language_code=language_code)
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

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True)

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

    try:
        print("\n***WELCOME***\n")
        while True:
            query = input("\nUser: ")
            response=chain.invoke(query)
            print(f"Assistant: {response}")
    except KeyboardInterrupt:
        print("BYE BYE!!!")

def conversation_agent():

    save_directory='./files'
    file_source = "./files/jokerbirot_space_musician.txt"
    persist_directory = './chroma/jokerbirot_space_musician'
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 200

    model_name="gpt-3.5-turbo"
    temperature=0
    overwrite=True

    embedding = OpenAIEmbeddings()

    vectordb=create_vectordb_from_file(
        filename=file_source,
        persist_directory=persist_directory,
        embedding=embedding,
        overwrite=overwrite
    )

    retriever = vectordb.as_retriever()

    conversational_memory = ConversationBufferWindowMemory(
        memory_key="chat_history", #needs to be present
        ai_prefix="AI Assistant",
        k=3, #how many interactions can be remembered
        return_messages=True
    )

    llm=ChatOpenAI(
        model_name=model_name,
        temperature=temperature
    )

    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever
    )

    # STANDARD WAY
    # query="Who is Jokerbirot?"
    # response=qa.run(query)
    # print(response)

    tools = [
        Tool(
            name="jokerbirot story",
            func=qa.run,
            description="use this tool when answering questions about the topic" #the description is important because the agent, based on this will decide which tool to use
        )
    ]

    agent=initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, #https://api.python.langchain.com/en/latest/agents/langchain.agents.agent_types.AgentType.html#langchain.agents.agent_types.AgentType
        llm=llm,
        tools=tools,
        memory=conversational_memory,
        verbose=True,
        max_iterations=3, #caps the agent at taking a certain number of steps
        early_stopping_method="generate", #By default, the early stopping uses the force method which just returns that constant string. Alternatively, you could specify the generate method which then does one FINAL pass through the LLM to generate an output
    )

    # NEW WAY
    query="Who is Jokerbirot?"
    response=agent(query)
    # Entering new AgentExecutor chain...
    # {
    #     "action": "jokerbirot story",
    #     "action_input": "Jokerbirot"
    # }
    # Observation: Jokerbirot è un musicista di fama intergalattica proveniente dalla galassia di Andromedar. Ha atterrato sulla Terra nell'anno 2077 con la sua astronave dalle linee fluide e lucenti. La sua musica è composta da melodie celesti e armonie cosmiche, che hanno il potere di plasmare la realtà stessa. Le sue esibizioni, in cui suona il suo strumento chiamato "Armonar", sono molto ammirate e riempiono gli stadi di un pubblico estasiato. Jokerbirot sente però la nostalgia di casa e inizia a pensare di tornare a Andromedar.
    # Thought:{
    #     "action": "Final Answer",
    #     "action_input": "Jokerbirot is a renowned intergalactic musician from the Andromeda galaxy. He landed on Earth in the year 2077 with his sleek and shiny spacecraft. His music consists of celestial melodies and cosmic harmonies, which have the power to shape reality itself. His performances, where he plays his instrument called the 'Armonar', are highly admired and fill stadiums with an ecstatic audience. Jokerbirot, however, feels homesick and begins to contemplate returning to Andromeda."
    # }
    # > Finished chain.
    print(response)
    # {'input': 'Who is Jokerbirot?', 'chat_history': [], 'output': "Jokerbirot is a renowned intergalactic musician from the Andromeda galaxy. He landed on Earth in the year 2077 with his sleek and shiny spacecraft. His music consists of celestial melodies and cosmic harmonies, which have the power to shape reality itself. His performances, where he plays his instrument called the 'Armonar', are highly admired and fill stadiums with an ecstatic audience. Jokerbirot, however, feels homesick and begins to contemplate returning to Andromeda."}

    query="Which instrument does he play?"
    response=agent(query)
    # > Entering new AgentExecutor chain...
    # {
    #     "action": "jokerbirot story",
    #     "action_input": "instrument"
    # }
    # Observation: Jokerbirot suonava uno strumento chiamato "Armonar".
    # Thought:{
    #     "action": "Final Answer",
    #     "action_input": "Jokerbirot plays an instrument called the 'Armonar'."
    # }
    # > Finished chain.
    print(response)
    # {'input': 'Which instrument does he play?', 'chat_history': [HumanMessage(content='Who is Jokerbirot?'), AIMessage(content="Jokerbirot is a renowned intergalactic musician from the Andromeda galaxy. He landed on Earth in the year 2077 with his sleek and shiny spacecraft. His music consists of celestial melodies and cosmic harmonies, which have the power to shape reality itself. His performances, where he plays his instrument called the 'Armonar', are highly admired and fill stadiums with an ecstatic audience. Jokerbirot, however, feels homesick and begins to contemplate returning to Andromeda.")], 'output': "Jokerbirot plays an instrument called the 'Armonar'."}


load_dotenv()

#chain_query_from_text_file()
#simple_query_from_texts()
#chain_query_from_texts()
#local_file=save_remote_file("https://www.gutenberg.org/cache/epub/1934/pg1934.txt")
#simple_query_from_youtube()
#multiple_files(language_code='it')

#simple_query_from_file()
#conversation_from_file()
conversation_agent()