from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient 
import asyncio 

# See https://docs.langchain.com/oss/python/langchain/mcp
async def main():

    llm = ChatOllama(
        model="granite4:3b",
        base_url="http://jetai:11434",
        temperature=0,
    )

    client = MultiServerMCPClient(  
        {
            "scamp": {
                "transport": "streamable_http",  # HTTP-based remote server
                "url": "http://piai:8100/mcp",
            }
        }
    )

    tools = await client.get_tools()  

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant. Use tools, where available, to get the most accurite awnsers."
    )

    scamp_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What are all the State Parks are within 20 miles of me?"}]}
    )

    print(scamp_response)

if __name__ == "__main__":
    asyncio.run(main())