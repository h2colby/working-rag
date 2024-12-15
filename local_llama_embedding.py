from typing import Any, Optional
from pydantic import Field, ConfigDict
from llama_index.core.base.embeddings.base import BaseEmbedding
import logging
import torch

logger = logging.getLogger(__name__)

class LocalLlamaEmbedding(BaseEmbedding):
    # Pydantic model configuration to allow arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Declare fields with types
    tokenizer: Any = Field(...)
    model: Any = Field(...)
    device: Any = Field(...)
    max_tokens: int = Field(default=4096)

    def __init__(self, tokenizer: Any, model: Any, device: Any, max_tokens: int = 4096):
        # Pass fields to parent constructor so that Pydantic can validate and set them
        super().__init__(tokenizer=tokenizer, model=model, device=device, max_tokens=max_tokens)
        logger.info(f"Initialized LocalLlamaEmbedding (max_tokens={max_tokens})")

    def _get_text_embedding(self, text: str) -> list[float]:
        inputs = self.tokenizer(
            text, return_tensors='pt', truncation=True, max_length=self.max_tokens
        ).to(self.device)

        attention_mask = inputs['attention_mask']
        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        embeddings = (
            last_hidden_state * attention_mask.unsqueeze(-1)
        ).sum(1) / attention_mask.sum(-1, keepdim=True)

        return embeddings.cpu().numpy()[0].tolist()

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

