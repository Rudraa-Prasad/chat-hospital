'''to activate and invoke chatbot_V1. passing prompt {context} from  ChromaDB'''



from langchain_intro.chatbot_V1 import review_chain

question = """Has anyone complained about
           communication with the hospital staff?"""

review_chain.invoke(question)


# AI_Response
'''Yes, several patients have complained about communication
with the hospital staff. Terri Smith mentioned that the
communication between the medical staff and her was unclear,
leading to misunderstandings about her treatment plan.
Kurt Gordon also mentioned that the lack of communication
between the staff and him left him feeling frustrated and
confused about his treatment plan. Ryan Jacobs also experienced
frustration due to the lack of communication from the staff.
Shannon Williams also mentioned that the lack of communication
between the staff and her made her stay at the hospital less enjoyable.'''