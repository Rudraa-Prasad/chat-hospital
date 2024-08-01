'''chatbot - passing prompt {context} from  ChromaDB'''


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

''' Instead of passing context in manually as in version V0, review_chain will pass your question
 to the retriever to pull relevant reviews. Assigning question to a RunnablePassthrough object ensures
  the question gets passed unchanged to the next step in the chain.'''

'''Here the input to prompt is expected to be a map with keys "context" and "question".
 The user input is just the question. So we need to get the context using our retriever and passthrough
  the user input under the "question" key. In this case, the RunnablePassthrough allows us to pass on the 
  user's question to the prompt and model.'''

'''Placeholder in Pipelines: When constructing a data processing pipeline, you might have steps that are conditionally applied. 
RunnablePassthrough can act as a placeholder in these cases, allowing the pipeline to function without errors even 
when certain steps are skipped. so that here we can pass question later.'''