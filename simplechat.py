# simplechat.py (patched)
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
# removed MemorySaver import - we won't checkpoint in memory
from dotenv import load_dotenv

load_dotenv()

# initialize the LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], ...]


def chat_node(state: ChatState):
    messages = state["messages"]
    # delegate streaming to the underlying llm - yield chunks as the graph expects
    for chunk in llm.stream(messages):
        yield {"messages": [chunk]}


# Build the state graph and compile WITHOUT an in-memory checkpointer
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# compile the graph *without* a checkpointer so no MemorySaver is used
chatbot = graph.compile()
# 
'''
for message_chunk, metadata in chatbot.stream(
    {'messages':[HumanMessage(content='what is recipe to make pasta')]},
    config={'configurable':{'thread_id':'thread-1'}},
    stream_mode='messages'
):
    if message_chunk.content:
        print(message_chunk.content,end="",flush=True)

'''
'''
initial_state={
    'messages':[HumanMessage(content='what is capital of india')]

}


result=workflow.invoke(initial_state)['messages'].content
print(result)'''
'''
thread_id = '1'
while True:
    user_message = input("enter your message -: ")
    print("user message:- ", user_message)

    if user_message.strip().lower() in ['exit', 'close', 'bye', 'by']:
        break

    config = {'configurable': {'thread_id': thread_id}}

    response = chatbot.invoke(
        {'messages': [HumanMessage(content=user_message)]},
        config=config
    )

    print("AI", response['messages'][-1].content)



'''
