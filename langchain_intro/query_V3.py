'''to activate and invoke chatbot_V3.'''



from langchain_intro.chatbot_V3 import hospital_agent_executor

hospital_agent_executor.invoke(
    {"input": "What is the current wait time at hospital C?"}
)

##AI_Response and output
'''> Entering new AgentExecutor chain...

Invoking: `Waits` with `C`

1374The current wait time at Hospital C is 1374 minutes.

> Finished chain.
{'input': 'What is the current wait time at hospital C?',
'output': 'The current wait time at Hospital C is 1374 minutes.',
'intermediate_steps': [(AgentActionMessageLog(tool='Waits',
tool_input='C', log='\nInvoking: `Waits` with `C`\n\n\n',
message_log=[AIMessage(content='', additional_kwargs={'function_call':
{'arguments': '{"__arg1":"C"}', 'name': 'Waits'}})]), 1374)]}'''

#### new query

hospital_agent_executor.invoke(
    {"input": "What have patients said about their comfort at the hospital?"}
)


#AI_Response and optput
'''> Entering new AgentExecutor chain...

Invoking: `Reviews` with `What have patients said about their comfort at the
hospital?`

Patients have mentioned both positive and negative aspects of their comfort at
the hospital. One patient mentioned that the hospital's dedication to patient
comfort was evident in the well-designed private rooms and comfortable furnishings,
which made their recovery more bearable and contributed to an overall positive
experience. However, other patients mentioned that the uncomfortable beds made
it difficult for them to get a good night's sleep during their stay, affecting
their overall comfort. Another patient mentioned that the outdated and
uncomfortable beds affected their overall comfort, despite the doctors being
knowledgeable and the hospital having a clean environment. Patients have shared
mixed feedback about their comfort at the hospital. Some have praised the well-designed
private rooms and comfortable furnishings, which contributed to a positive experience.
However, others have mentioned discomfort due to the outdated and uncomfortable beds,
affecting their overall comfort despite the hospital's clean environment and knowledgeable
doctors.

> Finished chain.
{'input': 'What have patients said about their comfort at the hospital?', 'output':
"Patients have shared mixed feedback about their comfort at the hospital. Some have
praised the well-designed private rooms and comfortable furnishings, which contributed
to a positive experience. However, others have mentioned discomfort due to the outdated
and uncomfortable beds, affecting their overall comfort despite the hospital's clean
environment and knowledgeable doctors.", 'intermediate_steps':
[(AgentActionMessageLog(tool='Reviews', tool_input='What have patients said about their
comfort at the hospital?', log='\nInvoking: `Reviews` with `What have patients said about
their comfort at the hospital?`\n\n\n', message_log=[AIMessage(content='',
additional_kwargs={'function_call': {'arguments': '{"__arg1":"What have patients said about
their comfort at the hospital?"}', 'name': 'Reviews'}})]), "Patients have mentioned both
positive and negative aspects of their comfort at the hospital. One patient mentioned that
the hospital's dedication to patient comfort was evident in the well-designed private rooms
and comfortable furnishings, which made their recovery more bearable and contributed to an
overall positive experience. However, other patients mentioned that the uncomfortable beds
made it difficult for them to get a good night's sleep during their stay, affecting their
overall comfort. Another patient mentioned that the outdated and uncomfortable beds affected
their overall comfort, despite the doctors being knowledgeable and the hospital having a clean
environment.")]}'''