from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient 
from langsmith import traceable

from dotenv import load_dotenv
import asyncio 

# Add in uvicorn web server and FastAPI web service module.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

load_dotenv(dotenv_path=".env", override=True)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def lcchat(prompt:str):
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
    response=await lcchat(prompt)
    return(response)

# Note: Make sure this line is at the end of the file so fastAPI falls through the other
# routes before serving up static files 
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8123)

"""
Angular client Note:
Use the following chat prompt to make an angular component to use this service
Create an angular component that accepts a prompt string and sends it to a web service
 in the format of http://piai:8123/chat?prompt={prompt}, this will return a
  markdown formated string response that needs to be displayed to the user.
"""