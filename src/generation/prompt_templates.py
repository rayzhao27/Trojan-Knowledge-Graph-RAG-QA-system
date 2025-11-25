SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.

Guidelines:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say "I don't have enough information to answer that"
- Be concise and accurate
- Cite specific parts of the context when relevant
- If asked about something not in the context, acknowledge the limitation
- Limits the final answer under 2500 tokens"""

QA_PROMPT_TEMPLATE = """Context information from relevant documents:

{context}

---

Question: {question}

Answer the question based on the context above. If the context doesn't contain the answer, say so."""

CHAT_PROMPT_TEMPLATE = """Based on the following context, please answer the user's question:

{context}

Question: {question}

Answer:"""
