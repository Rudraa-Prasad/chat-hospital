import dotenv
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
#OPENAI_API_KEY = "sk-None-r04axS8ytRqPbihtpxjDT3BlbkFJesRkGv8Js8MOf5548lHa"
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)