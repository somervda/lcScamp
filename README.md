# lcScamp

The langchain based client for my [ScampAI](https://youtube.com/playlist?list=PLmfc7XNirQox21HoV-banMpcQ_cHs_5uc&si=DowJU4ugVvkk38A7) project. I am using Langchain 1.0 to manage working with my Ollama based LLMs and working with MCP tools [mcpScamp](https://github.com/somervda/mcpScamp).

See [Langchain](https://reference.langchain.com/python/langchain/langchain/). The new Langchain 1.0 is rearchitected to basically run on top of Langgraph. This abstracts away some of the difficulty with working directly with Langgraph and makes capabilities like [Agents](https://docs.langchain.com/oss/python/langchain/agents) and [Agent Video](https://youtu.be/08qXj9w-CG4?si=hvrMkSrHRUc3DiKf) much easier to implement. I will be attempting to use Langchain 1.0 features, where appropriate, to make the coding simplier. see [Langchain 1.0 Video](https://youtu.be/r5Z_gYZb4Ns?si=muByUZT2Xo7yquSy)


## Installation

Requires uv package manager. Follow the following steps to setup uv in the project, create a python virtual environement 

- uv init
- uv venv
- source .venv/bin/activate

install langchain packages

- uv add langchain
- uv add langchain-ollama
- uv add langchain-mcp-adapters