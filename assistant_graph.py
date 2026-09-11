from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# Initialize local Ollama model
llm = Ollama(model="qwen")

# Define state graph workflow
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    messages = state["messages"]
    prompt_history = "\n".join([
        f"User: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}"
        for m in messages[:-1]
    ])
    current_input = messages[-1].content
    
    full_prompt = f"{prompt_history}\nUser: {current_input}\nAssistant:" if prompt_history else f"User: {current_input}\nAssistant:"
    response_text = llm.invoke(full_prompt)
    return {"messages": [AIMessage(content=response_text.strip())]}

workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)

# Compile graph with memory checkpointer so Studio can track state threads
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)