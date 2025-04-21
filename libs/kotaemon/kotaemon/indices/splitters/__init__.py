from ..base import DocTransformer, LlamaIndexDocTransformerMixin
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser


class BaseSplitter(DocTransformer):
    """Represent base splitter class"""

    ...


class TokenSplitter(LlamaIndexDocTransformerMixin, BaseSplitter):
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        separator: str = " ",
        **params,
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            **params,
        )

    def _get_li_class(self):
        from llama_index.core.text_splitter import TokenTextSplitter

        return TokenTextSplitter


class RecursiveNodeParser(LangchainNodeParser):
    def __init__(
        self,
        chunk_size=1024,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""],
        **params,
    ):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators
        )
        super().__init__(text_splitter)


class RecursiveSplitter(LlamaIndexDocTransformerMixin, BaseSplitter):
    def __init__(
        self,
        chunk_size=1024,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""],
        **params,
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            **params,
        )

    def _get_li_class(self):
        return RecursiveNodeParser


class SentenceWindowSplitter(LlamaIndexDocTransformerMixin, BaseSplitter):
    def __init__(
        self,
        window_size: int = 3,
        window_metadata_key: str = "window",
        original_text_metadata_key: str = "original_text",
        **params,
    ):
        super().__init__(
            window_size=window_size,
            window_metadata_key=window_metadata_key,
            original_text_metadata_key=original_text_metadata_key,
            **params,
        )

    def _get_li_class(self):
        from llama_index.core.node_parser import SentenceWindowNodeParser

        return SentenceWindowNodeParser
