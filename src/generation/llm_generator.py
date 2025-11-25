import os
import logging

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from .prompt_templates import SYSTEM_PROMPT, QA_PROMPT_TEMPLATE
from typing import List, Dict, Tuple, Generator, Iterable, cast

logger = logging.getLogger(__name__)


class LLMGenerator:
    def __init__(
            self,
            max_tokens: int = 3000
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.max_tokens = max_tokens

        logger.info(f"LLM Generator initialized")

    def _prepare_messages(
            self,
            query: str,
            retrieved_chunks: List[Tuple[Dict, float]],
            max_context_chunks: int = 5
    ) -> Iterable[ChatCompletionMessageParam]:
        """Build messages array with context"""

        context = self._build_context(retrieved_chunks[:max_context_chunks])
        user_prompt = QA_PROMPT_TEMPLATE.format(context=context, question=query)

        return cast(Iterable[ChatCompletionMessageParam], [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ])

    def _get_completion_params(self, model: str, temperature: float, stream: bool = False) -> Dict:
        # Use max_completion_tokens for newer models
        params = {
            "model": model,
            "temperature": temperature,
            "stream": stream
        }

        # Support different models
        if model.startswith("gpt-4") or model.startswith("gpt-5") or "turbo" in model:
            params["max_completion_tokens"] = self.max_tokens
        else:
            params["max_tokens"] = self.max_tokens

        return params

    def _build_context(self, chunks: List[Tuple[Dict, float]]) -> str:
        context_parts = []

        for i, (chunk, score) in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}] (Score: {score:.3f})\n"
                f"From: {chunk['source']}, Page {chunk['page_num']}\n"
                f"{chunk['text']}\n"
            )

        return "\n---\n".join(context_parts)

    def _extract_sources(self, chunks: List[Tuple[Dict, float]]) -> List[Dict]:
        return [
            {
                "source": chunk['source'],
                "page_num": chunk['page_num'],
                "chunk_id": chunk['chunk_id'],
                "score": float(score),
                "text_preview": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            }
            for chunk, score in chunks
        ]

    def _error_response(self, model: str, error_msg: str) -> Dict:
        return {
            "answer": f"Error generating answer: {error_msg}",
            "sources": [],
            "model": model,
            "tokens_used": 0
        }

    def generate_answer(
            self,
            query: str,
            retrieved_chunks: List[Tuple[Dict, float]],
            model: str = "gpt-5-nano",
            temperature: float = 1,
            max_context_chunks: int = 5
    ) -> Dict:

        messages = self._prepare_messages(query, retrieved_chunks, max_context_chunks)

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                **self._get_completion_params(model, temperature)
            )

            return {
                "answer": response.choices[0].message.content,
                "sources": self._extract_sources(retrieved_chunks[:max_context_chunks]),
                "model": model,
                "tokens_used": response.usage.total_tokens
            }

        except Exception as e:
            print(f"[ERROR] LLM generation failed: {e}")
            return self._error_response(model, str(e))

    def generate_streaming(
            self,
            query: str,
            retrieved_chunks: List[Tuple[Dict, float]],
            model: str = "gpt-5-nano",
            temperature: float = 1,
            max_context_chunks: int = 5
    ) -> Generator[str, None, None]:

        # Generate answer with real-time responses
        messages = self._prepare_messages(query, retrieved_chunks, max_context_chunks)

        try:
            stream = self.client.chat.completions.create(
                messages=messages,
                **self._get_completion_params(model, temperature, stream=True)
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"Error: {str(e)}"
