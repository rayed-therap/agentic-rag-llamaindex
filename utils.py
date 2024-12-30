from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.settings import Settings
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector


def get_router_query_engine(file_path: str, llm=None, embed_model=None):
    """Get router query engine."""

    # Set LLM and embedding model
    Settings.llm = llm or Gemini(model="models/gemini-2.0-flash-exp")
    Settings.embed_model = embed_model or GeminiEmbedding(
        model="models/text-embedding-004"
    )

    # Load data
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    nodes = SentenceSplitter(chunk_size=1024).get_nodes_from_documents(documents)

    # Create summary and vector indices
    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes)

    # Create summary and vector query engines
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize", use_async=True
    )
    vector_query_engine = vector_index.as_query_engine()

    # Create summary and vector tools
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=("Useful for summarization questions related to MetaGPT"),
    )
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=("Useful for retrieving specific context from the MetaGPT paper"),
    )

    # Create the router query engine
    router_query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[summary_tool, vector_tool],
        verbose=True,
    )

    # Return the router query engine
    return router_query_engine
