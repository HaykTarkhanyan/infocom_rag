"""RAG: retrieve from Weaviate and generate answers with Gemini."""

import weaviate
from google import genai

from config import (
    COLLECTION_NAME,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_TOP_P,
    GEMINI_TOP_K_SAMPLING,
    SYSTEM_PROMPT,
    TOP_K,
    MAX_DISTANCE,
    HNSW_EF,
)
from embeddings import EmbeddingModel


class RAG:
    def __init__(
        self,
        top_k: int = TOP_K,
        max_distance: float = MAX_DISTANCE,
        temperature: float = GEMINI_TEMPERATURE,
        max_output_tokens: int = GEMINI_MAX_OUTPUT_TOKENS,
        model: str = GEMINI_MODEL,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.embedder = EmbeddingModel()
        self.weaviate_client = weaviate.connect_to_local()
        self.collection = self.weaviate_client.collections.get(COLLECTION_NAME)
        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)

        # Tunable at runtime
        self.top_k = top_k
        self.max_distance = max_distance
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.model = model
        self.system_prompt = system_prompt

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve the most relevant chunks for a query."""
        query_vec = self.embedder.embed_query(query)

        results = self.collection.query.near_vector(
            near_vector=query_vec,
            limit=self.top_k,
            return_metadata=["distance"],
        )

        retrieved = []
        for obj in results.objects:
            dist = obj.metadata.distance if obj.metadata else None
            if dist is not None and dist > self.max_distance:
                continue
            props = obj.properties
            retrieved.append({
                "text": props.get("text", ""),
                "sender": props.get("sender", ""),
                "date": props.get("date", ""),
                "chunk_type": props.get("chunk_type", ""),
                "distance": dist,
            })

        return retrieved

    def format_context(self, results: list[dict]) -> str:
        """Format retrieved chunks into a context string."""
        lines = []
        for r in results:
            # For single messages the text already has [date] sender: format from chunking
            # For thread/window chunks the text is pre-formatted
            if r["chunk_type"] in ("single_message", ""):
                lines.append(r["text"])
            else:
                lines.append(r["text"])
        return "\n".join(lines)

    def answer(self, query: str) -> tuple[str, list[dict]]:
        """Full RAG pipeline: retrieve context, then generate answer.

        Returns (answer_text, retrieved_chunks) so the UI can show sources.
        """
        results = self.retrieve(query)

        if not results:
            return "I couldn't find any relevant messages in the chat history.", []

        context = self.format_context(results)

        prompt = f"""Context from chat history:
---
{context}
---

User question: {query}"""

        response = self.gemini_client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                top_p=GEMINI_TOP_P,
                top_k=GEMINI_TOP_K_SAMPLING,
            ),
        )

        return response.text, results

    def close(self):
        self.weaviate_client.close()
