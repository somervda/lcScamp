from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient 
from langsmith import traceable
import asyncio 
import time
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

load_dotenv(dotenv_path=".env", override=True)

app = FastAPI()

# Set up the model, using the chatOllama provider package
llm = ChatOllama(
    model="granite4:3b",
    base_url="http://192.168.1.117:11434",
    num_ctx=20480,
    max_tokens=4096,
    temperature=0.1,
)

# I use the MultiServerMCPClient object to expose the mcpScamp service to the langchain agent
client = MultiServerMCPClient(  
    {
        "scamp": {
            "transport": "streamable_http",  # HTTP-based remote server
            "url": "http://piai:8100/mcp",
        }
    }
)

# See https://docs.langchain.com/oss/python/langchain/mcp
@traceable
async def aichat(prompt:str):
    tools =  await client.get_tools()  

    # Create a new agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant. Use tools, where available to get the most accurite answers. If a wikihow or wikipedia link is available include it in the user response"
    )

    scamp_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )

    # Return user response
    return(scamp_response["messages"][-1].content)

# Web service : e.g. http://localhost:8123/chat?prompt=How%20do%20I%20make%20scones 
@app.get("/chat")
async def chat(prompt: str):
    response=await aichat(prompt)
    return(response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8123)