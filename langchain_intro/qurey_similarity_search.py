'''performing similarity search in chromaDB to get most relevant content '''


import dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

REVIEWS_CHROMA_PATH = "/workspaces/chat-hospital/chroma_data"
dotenv.load_dotenv()

reviews_vector_db = Chroma(persist_directory=REVIEWS_CHROMA_PATH, embedding_function = OpenAIEmbeddings())

question = """has anyone mentioned bad behavior of staff ?"""

relevant_docs = reviews_vector_db.similarity_search(question, k=3)

print(f'page-01: {relevant_docs[0].page_content} \n')
print(f'page-02: {relevant_docs[1].page_content} \n')
print(f'page-03: {relevant_docs[2].page_content}')