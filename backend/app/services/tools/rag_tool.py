"""
RAG Search Tool - Your existing RAG as a tool for agents.

This wraps your VectorStore.search() so agents can use it.
The agent decides WHEN to search, this tool does the actual searching.
"""

from app.services.tools.base import BaseTool, ToolResult
from app.services.vector_store import VectorStore
from app.core.logging import get_logger

logger = get_logger(__name__)


class RAGSearchTool(BaseTool):
    """
    Search the knowledge base for information.

    This tool wraps your existing vector store search.
    When an agent needs factual information, it calls this tool.

    The 'description' is crucial - the LLM reads it to decide
    when to use this tool vs other tools or just answering directly.
    """

    name = "rag_search"
    description = (
        "Search the knowledge base for information. "
        "Use this tool when you need to find facts, documentation, "
        "or specific information from the indexed documents. "
        "Always use this before answering factual questions."
    )

    def __init__(self, vector_store: VectorStore, collection_id: int | None = None):
        """
        Initialize with a VectorStore instance.

        Args:
            vector_store: Your existing VectorStore for searching
            collection_id: Optional collection to search within
        """
        self.vector_store = vector_store
        self.collection_id = collection_id

    def get_parameters_schema(self) -> dict:
        """
        Define what parameters this tool accepts.

        The LLM will see this schema and provide appropriate arguments.
        """
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant information"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    async def run(self, query: str, limit: int = 5) -> ToolResult:
        """
        Execute the search.

        This is called by the agent executor when the LLM
        decides to use this tool.

        Args:
            query: What to search for
            limit: Max results (default 5)

        Returns:
            ToolResult with search results or error
        """
        logger.info("rag_tool_search", query=query, limit=limit)

        try:
            # Use your existing vector store search
            results = await self.vector_store.search(
                query=query,
                collection_id=self.collection_id,
                limit=limit,
                score_threshold=0.3  # Lower threshold to get more results
            )

            if not results:
                return ToolResult(
                    success=True,
                    output="No relevant information found in the knowledge base."
                )

            # Format results for the LLM to read
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"[Source {i}: {result['document_title']}]\n"
                    f"{result['content']}\n"
                    f"(Relevance: {result['score']:.2f})"
                )

            output = "\n\n---\n\n".join(formatted_results)

            logger.info(
                "rag_tool_results",
                query=query,
                num_results=len(results),
                top_score=results[0]["score"] if results else 0
            )

            return ToolResult(
                success=True,
                output=output
            )

        except Exception as e:
            logger.error("rag_tool_error", query=query, error=str(e))
            return ToolResult(
                success=False,
                output="",
                error=f"Search failed: {str(e)}"
            )
