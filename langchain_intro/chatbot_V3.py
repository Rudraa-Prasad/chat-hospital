'''chatbot using Agents instead of chain.'''
''' to activate this chatbot run query_V3.py'''

import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
##new imports
from langchain.agents import (create_openai_functions_agent, Tools, AgentExecutor)
from langchain import hub
from langchain_intro.tools import get_current_wait_time


dotenv.load_dotenv()

######## defining prompt template

review_template_str = """Your job is to use patient
reviews to answer questions about their experience at
a hospital. Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer, say
you don't know.

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)


review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)

messages = [review_system_prompt, review_human_prompt]


review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

#######   retriving similar data from DB
REVIEWS_CHROMA_PATH = "chroma_data/"

reviews_vector_db = Chroma(persist_directory = REVIEWS_CHROMA_PATH, embedding_function = OpenAIEmbeddings())

'''added reviews_retriever to review_chain so that relevant reviews are passed to the prompt as context not all context. 
specified k=10, the retriever will fetch the ten reviews most similar to the user’s question.'''


reviews_retriever = reviews_vector_db.as_retriever(k=10)


######## defining model

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
output_parser = StrOutputParser()

review_chain =({"context": reviews_retriever, "question" : RunnablePassthrough()} | 
review_prompt_template | chat_model | output_parser)

######### using Agent to decide sequence of operations based on user input

tools = [
    Tool(
        name="Reviews",
        func=review_chain.invoke,
        description="""Useful when you need to answer questions
        about patient reviews or experiences at the hospital.
        Not useful for answering questions about specific visit
        details such as payer, billing, treatment, diagnosis,
        chief complaint, hospital, or physician information.
        Pass the entire question as input to the tool. For instance,
        if the question is "What do patients think about the triage system?",
        the input should be "What do patients think about the triage system?"
        """,
    ),
    Tool(
        name="Waits",
        func=get_current_wait_time,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. This tool returns wait times in
        minutes. Do not pass the word "hospital" as input,
        only the hospital name itself. For instance, if the question is
        "What is the wait time at hospital A?", the input should be "A".
        """,
    ),
]

hospital_agent_prompt = hub.pull("hwchase17/openai-functions-agent")

agent_chat_model =  ChatOpenAI(model="gpt-3.5-turbo-1106",temperature=0)

hospital_agent = create_openai_functions_agent(llm = agent_chat_model, prompt = hospital_agent_prompt, tools = tools)

hospital_agent_executor = AgentExecutor(agent = hospital_agent, tools = tools, return_interediate_steps = True, verbose = True)


'''Tool is an interface that an agent uses to interact with a function.
 For instance, the first tool is named Reviews and it calls review_chain.invoke() 
 if the question meets the criteria of description.'''