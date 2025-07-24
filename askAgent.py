import os
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool


# Start FastAPI app
app = FastAPI()
# Load environment variables
env_path = load_dotenv()

# Initialize Chroma client and collection

chroma_client = chromadb.HttpClient(host='10.112.154.220', port=8001)
default_ef = DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(name="my_collection", embedding_function =default_ef)

# Define a tool that queries Chroma
class ChromaQueryResponse(BaseModel):
    query: str
    documents: list[str]


def query_chroma(query: str, n_results: int = 1) -> str:
    """
    Retrieve top-n related documents from the Chroma collection for the given query.
    Returns a plain-text concatenation of the documents.
    """
    results = collection.query(
        query_embeddings=default_ef(query),
        n_results=n_results
    )

    print("////////////////////////////////////////////////////////////")
    print(f"Query: {query}, Results: {results}")
    print("////////////////////////////////////////////////////////////")

    return results if results else "No relevant documents found."

# Wrap as a LangChain Tool
tool_query = Tool(
    name="chroma_query",
    func=query_chroma,
    description=(
        "Retrieves context documents from the Chroma vector store. "
        "Input: a natural-language query. Returns the concatenated top documents."
    )
)

# Define the Pydantic model for structured output
class ChromResponse(BaseModel):
    query: str
    answer: str
    used_documents: list[str]

# Instantiate the LLM and output parser
#llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", max_tokens=1024 , temperature=0.3)
# open AI model
llm = ChatOpenAI(model="gpt-4o", max_tokens=1024, temperature=0.3)
parser = PydanticOutputParser(pydantic_object=ChromResponse)




# Build the chat prompt template
template = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent assistant that answers questions using retrieved context."
                " Use the `chroma_query` tool to fetch relevant docs when needed and cite them."
                "The response should be nom more than 100 words."
                " Wrap your response following the structured format instructions: {format_instructions}"),
    ("human", "Query: {query}"),
    ("placeholder", "{agent_scratchpad}")
]).partial(format_instructions=parser.get_format_instructions())

# Create the tool-calling agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=template,
    tools=[tool_query]
)
executor = AgentExecutor(agent=agent, tools=[tool_query], verbose=True)



@app.post("/chroma-query", response_model=ChromResponse)
def run_chroma_agent(query: str):
    """Endpoint to query the Chroma agent and receive structured response."""
    raw = executor.invoke({"query": query})
    parsed = parser.parse(raw.get("output")[0]["text"])
    return parsed

# Utility: optional save endpoint
@app.post("/save-output")
def save_output(data: str, filename: str = "chroma_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"---{timestamp}---\n{data}\n\n")
    return {"status": "saved", "file": filename}