from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient 
from langsmith import traceable
import asyncio 
import time
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

# See https://docs.langchain.com/oss/python/langchain/mcp
@traceable
async def main():
    start_time = time.time()

    # Set up the model, using the chatOllama class, to be used with this agent
    # I've found that the granate4:3b model was the model that works best 
    # based on balancing memory availability on a Jetson Nano (About 7Gb)  and speed.
    # Note: I set the context window size (num_ctx) to be the maximum that reliable runs on the jetson nano
    # Note: I set the context window size (num_ctx) to be the maximum that reliable runs on the jetson nano
    # and set temprature to 0.1 to ensure model accuracy high, and variablity low. 

    llm = ChatOllama(
        model="granite4:3b",
        base_url="http://jetai:11434",
        num_ctx=20480,
        max_tokens=20480,
        temperature=0.1,
    )
    # llm = ChatOllama(
    #     model="gpt-oss:20b",
    #     base_url="http://mac:11434",
    #     num_ctx=131072,
    #     max_tokens=64636,
    #     temperature=0.1,
    # )

    # I use the MultiServerMCPClient object to expose the mcpScamp service to the langchain agent

    client = MultiServerMCPClient(  
        {
            "scamp": {
                "transport": "streamable_http",  # HTTP-based remote server
                "url": "http://piai:8100/mcp",
            }
        }
    )

    tools =  await client.get_tools()  

    # Finally I create the agent. In langchain 1.0 this is now a class in the Langchain module (Even though it uses langgraph under the covers)
    #  This is where the model and tools get assigned to the agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant. Use tools, where available to get the most accurite answers."
    )

    scamp_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What state parks are within 20 miles of Lewisburg, PA"}]}
    )
    # scamp_response = await agent.ainvoke(
    #     {"messages": [{"role": "user", "content": "Compare the RV park ratings within 15 miles of Pittsburgh, PA and Scranton, PA. Which location has the highest average ratings?"}]}
    # )

    # Show results
    print(scamp_response)
    elapsed_time = time.time() - start_time
    print(f"\nElapsed time: {elapsed_time:.1f} seconds")
    final_response = scamp_response["messages"][-1].content
    print(final_response)


if __name__ == "__main__":
    asyncio.run(main())