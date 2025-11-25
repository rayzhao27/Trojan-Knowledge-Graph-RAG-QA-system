import os
import re
import logging
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class Chunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Split by Classes -> Functions -> Paragraphs -> Lines
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=[
                "\nclass ",  # Python Classes
                "\ndef ",  # Python Functions
                "\n\n",  # Paragraphs
                "\n",  # Lines
                " ",  # Words
                ""
            ]
        )
        logger.info("Using LangChain RecursiveCharacterTextSplitter")

    def chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []

        # Clean text but preserve code symbols
        text = self._clean_text(text)
        text_chunks = self.splitter.split_text(text)

        chunks = []
        source_name = self._get_source_name(source)

        for i, chunk_text in enumerate(text_chunks, 1):
            # Skip chunks that are too small
            if len(chunk_text) < 50:
                continue

            chunks.append({
                'chunk_id': f"{source_name}_c{i}",
                'text': chunk_text,
                'source': source,
                'page_num': self._estimate_page(i)
            })

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean text and preserve technical symbols"""

        # Replace multi-spaces with single space
        lines = text.split('\n')
        cleaned_lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
        text = '\n'.join(cleaned_lines)

        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _get_source_name(self, source: str) -> str:
        basename = os.path.splitext(os.path.basename(source))[0]

        return re.sub(r'[^\w]', '_', basename)

    def _estimate_page(self, chunk_id: int) -> int:
        return (chunk_id - 1) // 3 + 1
