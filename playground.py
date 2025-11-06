from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="granite4:3b",
    base_url="http://jetai:11434",
    temperature=0,
)

response = llm.invoke("what state parks are within 20 miles of me?")
print(response)