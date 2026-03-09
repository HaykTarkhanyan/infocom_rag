"""RAG: retrieve from Weaviate and generate answers with Gemini."""

import weaviate
from google import genai

from config import WEAVIATE_URL, COLLECTION_NAME, GEMINI_API_KEY, TOP_K
from embeddings import EmbeddingModel


SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based on Telegram chat history.
You will be given relevant messages from the chat as context.
Answer the user's question based on the provided context.
If the context doesn't contain enough information, say so honestly.
You can respond in Armenian, English, or Russian — match the language of the user's question.
Keep answers concise and relevant."""


class RAG:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.weaviate_client = weaviate.connect_to_local()
        self.collection = self.weaviate_client.collections.get(COLLECTION_NAME)
        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Retrieve the most relevant messages for a query."""
        query_vec = self.embedder.embed_query(query)

        results = self.collection.query.near_vector(
            near_vector=query_vec,
            limit=top_k,
            return_metadata=["distance"],
        )

        retrieved = []
        for obj in results.objects:
            props = obj.properties
            retrieved.append({
                "text": props.get("text", ""),
                "sender": props.get("sender", ""),
                "date": props.get("date", ""),
                "distance": obj.metadata.distance if obj.metadata else None,
            })

        return retrieved

    def format_context(self, results: list[dict]) -> str:
        """Format retrieved messages into a context string."""
        lines = []
        for r in results:
            lines.append(f"[{r['date']}] {r['sender']}: {r['text']}")
        return "\n".join(lines)

    def answer(self, query: str) -> str:
        """Full RAG pipeline: retrieve context, then generate answer."""
        results = self.retrieve(query)

        if not results:
            return "I couldn't find any relevant messages in the chat history."

        context = self.format_context(results)

        prompt = f"""Context from chat history:
---
{context}
---

User question: {query}"""

        response = self.gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )

        return response.text

    def close(self):
        self.weaviate_client.close()
