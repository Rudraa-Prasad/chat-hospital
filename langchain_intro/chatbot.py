'''chatbot passing prompt by defining here only , noy using ChromaBD'''


import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()

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


chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

output_parser = StrOutputParser()

#  You can chain together complex pipelines to create chatbot, and you end up with 
# an object that executes pipeline in a single method call. 
review_chain = review_prompt_template | chat_model | output_parser