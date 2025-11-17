from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient 
import asyncio 
import time

# See https://docs.langchain.com/oss/python/langchain/mcp
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

    tools = await client.get_tools()  

    # Finally I create the agent. In langchain 1.0 this is now a class in the Langchain module (Even though it uses langgraph under the covers)
    #  This is where the model and tools get assigned to the agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant. Use tools, where available to get the most accurite answers."
    )

    # scamp_response = await agent.ainvoke(
    #     {"messages": [{"role": "user", "content": "What does wikihow say about how to boil an egg"}]}
    # )

    # scamp_response = await agent.ainvoke(
    #     {"messages": [{"role": "user", "content": "What are the top 3 State Parks would you recomend me staying at that are within 40 miles of Lewisburg, PA. I will need RV camping at the park and prefer larger parks with good walking trails?"}]}
    # )

    scamp_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Whay state parks are within 15 miles of me"}]}
    )

    # print(scamp_response)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.1f} seconds")
    final_response = scamp_response["messages"][-1].content
    print(final_response)


if __name__ == "__main__":
    asyncio.run(main())