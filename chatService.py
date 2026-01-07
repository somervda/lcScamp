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
async def lcchat(prompt:str,llmService:int):
    # Set up the model, using the chatOllama provider package
    match llmService:
        case 1:
            llm = ChatOllama(
                model="granite4:3b",
                base_url="http://jetai:11434",
                num_ctx=16384,
                temperature=0.1,
            )
        case 2:
            llm = ChatOllama(
                model="granite4:3b",
                base_url="http://powerspec-g703:11434",
                num_ctx=32768,
                temperature=0.1,
            )        
        case 3:
            llm = ChatOllama(
                model="gpt-oss:20b",
                base_url="http://mac:11434",
                num_ctx=32768,
                temperature=0.1,
            )        
    tools =  await client.get_tools()  

    # Create a new agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant. Use tools, where available to get the most accurate answers. If a wikihow or wikipedia link is available include it in the user response"
    )

    scamp_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )

    # Return user response
    return(scamp_response["messages"][-1].content)

# Web service : e.g. http://localhost:8123/chat?prompt=How%20do%20I%20make%20scones 
#  and http://piai:8123/chat?prompt=I%20was%20born%2019th%20of%20April%201959.%20How%20old%20am%20I%20today,%20based%20on%20the%20local%20date%20and%20time&llmService=2
# http://localhost:8123/chat?prompt=How%20do%20I%20make%20scones 
@app.get("/chat")
async def chat(prompt: str,llmService:int=1):
    response=await lcchat(prompt,llmService)
    return(response)

# Note: Make sure this line is at the end of the file so fastAPI falls through the other
# routes before serving up static files 
app.mount("/", StaticFiles(directory="static", html=True), name="static")
# ngScamp is at http://piai:8123 


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8123)

