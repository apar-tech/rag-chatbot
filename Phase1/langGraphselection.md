## Discuss langGraph and n8n

After completing a basic RAG pipeline (chunking → embeddings → vector database → retrieval → LLM), the next learning path is often LangGraph or n8n.
1. LangGraph
   
LangGraph is a framework for building AI agents with workflows, memory, and decision-making.

Uses:

Multi-step AI agents

Chatbots with memory

Tool calling (search, database, APIs)

Complex workflows with branching logic

Example: User Question → Retrieve Documents → LLM → Check Answer Quality → If poor, Retrieve Again → Final Answer

LangGraph is a framework for creating stateful AI agent workflows using nodes and edges.

2. n8n

n8n is a no-code/low-code automation platform.

Uses:

Connect AI models with apps

Automate emails, WhatsApp, Google Sheets, databases, etc.

Build AI workflows without much coding

Example: Form Submission → Gemini/OpenAI → Generate Reply → Send Email Automatically

n8n is an open-source workflow automation tool that integrates AI and external services.

For this project, I have chosen LangGraph as the development framework. Since the project is being built using Python along with LangChain, ChromaDB, Gemini Embeddings, and Groq LLM, LangGraph integrates well with the existing technology stack. It also provides greater flexibility and control over the RAG workflow, making it suitable for developing a customized chatbot system.
