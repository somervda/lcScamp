from langchain_ollama import ChatOllama
from langchain.agents import create_agent


llm = ChatOllama(
    model="granite4:3b",
    base_url="http://jetai:11434",
    temperature=0,
)

agent = create_agent(
    model=llm,
    system_prompt="You are a helpful assistant."
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What time is it?"}
    ]
})

print(result)