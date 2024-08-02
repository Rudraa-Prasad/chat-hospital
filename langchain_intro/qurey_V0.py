'''to activate and invoke chatbot_V0 , both context and question passed'''



from langchain_intro.chatbot_V0 import review_chain

context = "I had a great stay!"
question = "Did anyone have a positive experience?"

review_chain.invoke({"context": context, "question": question})

# AI_Response
'''Yes, the patient had a great stay and had a
positive experience at the hospital.'''   #short response, context is short here.