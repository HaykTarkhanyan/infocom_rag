import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch import Tensor

from config import EMBEDDING_MODEL_NAME, MAX_TOKENS


def _average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def embed(self, texts: list[str], prefix: str = "passage") -> list[list[float]]:
        """Embed a list of texts. Use prefix='query' for queries, 'passage' for documents."""
        prefixed = [f"{prefix}: {t}" for t in texts]
        batch_dict = self.tokenizer(
            prefixed,
            max_length=MAX_TOKENS,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self.model(**batch_dict)
        embeddings = _average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], prefix="query")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts, prefix="passage")
