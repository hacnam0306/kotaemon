from ..base import DocTransformer, LlamaIndexDocTransformerMixin


class BaseSplitter(DocTransformer):
    """Represent base splitter class"""

    ...


class TokenSplitter(LlamaIndexDocTransformerMixin, BaseSplitter):
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        separator: str = " ",
        backup_separators: list[str] = None,
        **params,
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            backup_separators=backup_separators,
            **params,
        )

    def _get_li_class(self):
        from llama_index.core.text_splitter import TokenTextSplitter

        return TokenTextSplitter


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


class AcademicDocStructureSplitter:
    """A document structure-based splitter designed specifically for academic papers.
    
    This splitter identifies common sections in academic papers and uses them as natural
    chunk boundaries, while also respecting maximum chunk sizes.
    """
    
    def __init__(self, chunk_size=1024, chunk_overlap=256):
        """Initialize the academic paper structure splitter.
        
        Args:
            chunk_size: Maximum size of each chunk in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Common section headers in academic papers (lowercase for case-insensitive matching)
        self.section_patterns = [
            r"(?i)^\s*abstract\s*$",
            r"(?i)^\s*introduction\s*$",
            r"(?i)^\s*literature review\s*$",
            r"(?i)^\s*related work\s*$",
            r"(?i)^\s*methodology\s*$",
            r"(?i)^\s*methods\s*$",
            r"(?i)^\s*materials and methods\s*$",
            r"(?i)^\s*experimental setup\s*$",
            r"(?i)^\s*results\s*$",
            r"(?i)^\s*findings\s*$",
            r"(?i)^\s*discussion\s*$",
            r"(?i)^\s*analysis\s*$",
            r"(?i)^\s*conclusion\s*$",
            r"(?i)^\s*future work\s*$",
            r"(?i)^\s*acknowledgements?\s*$",
            r"(?i)^\s*references\s*$",
            r"(?i)^\s*bibliography\s*$",
            r"(?i)^\s*appendix\s*$"
        ]
        # Additional numeric section pattern (e.g., "1. Introduction", "2.3 Methods")
        self.numeric_section_pattern = r"(?i)^\s*\d+(\.\d+)*\s+[a-z][a-z0-9\s]+\s*$"
        
        # Import necessary libraries
        import re
        self.re = re
        
        try:
            from langchain_text_splitters import TokenTextSplitter
            self.token_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        except ImportError:
            # Fallback to a simple character count estimator if TokenTextSplitter is not available
            self.token_splitter = None
    
    def _estimate_tokens(self, text):
        """Estimate the number of tokens in the text.
        
        If TokenTextSplitter is available, use it. Otherwise, use a simple character-based estimation.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            int: Estimated number of tokens
        """
        if self.token_splitter:
            # Using langchain's token counter implementation
            return len(self.token_splitter._tokenizer.encode(text))
        else:
            # Simple character-based estimate (roughly 4 chars per token)
            return len(text) // 4
    
    def _is_section_header(self, line):
        """Check if a line is a section header.
        
        Args:
            line: The line to check
            
        Returns:
            bool: True if the line is a section header, False otherwise
        """
        # Check against our predefined section patterns
        for pattern in self.section_patterns:
            if self.re.match(pattern, line):
                return True
        
        # Check if it's a numeric section header
        if self.re.match(self.numeric_section_pattern, line):
            return True
            
        return False
    
    def _find_section_boundaries(self, text):
        """Find the boundaries of sections in the document.
        
        Args:
            text: The document text
            
        Returns:
            list: List of tuples (start_index, end_index, section_title)
        """
        lines = text.split('\n')
        section_boundaries = []
        current_section_start = 0
        current_section_title = "Header"  # Default section title for content before first section
        
        for i, line in enumerate(lines):
            if self._is_section_header(line):
                # End the previous section
                if i > 0:
                    section_end = i - 1
                    section_boundaries.append((
                        current_section_start, 
                        section_end, 
                        current_section_title
                    ))
                
                # Start a new section
                current_section_start = i
                current_section_title = line.strip()
        
        # Add the final section
        section_boundaries.append((
            current_section_start, 
            len(lines) - 1, 
            current_section_title
        ))
        
        return section_boundaries, lines
    
    def _create_chunks_from_section(self, section_lines, section_title):
        """Create appropriate chunks from a section.
        
        Args:
            section_lines: Lines of text in the section
            section_title: Title of the section
            
        Returns:
            list: List of text chunks from this section
        """
        section_text = '\n'.join(section_lines)
        
        # If the section is small enough, return it as a single chunk
        if self._estimate_tokens(section_text) <= self.chunk_size:
            return [f"{section_title}\n\n{section_text}"]
        
        # Otherwise, we need to split it further
        chunks = []
        
        # Use paragraph boundaries for further splitting
        paragraphs = section_text.split('\n\n')
        current_chunk = [section_title]  # Start with the section title
        current_chunk_size = self._estimate_tokens(section_title)
        
        for paragraph in paragraphs:
            paragraph_size = self._estimate_tokens(paragraph)
            
            # If adding this paragraph would exceed the chunk size, finalize the current chunk
            if current_chunk_size + paragraph_size + 2 > self.chunk_size:  # +2 for the newlines
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                
                # Start a new chunk with the section title and this paragraph
                current_chunk = [f"{section_title} (continued)", paragraph]
                current_chunk_size = self._estimate_tokens(current_chunk[0]) + paragraph_size + 2
            else:
                # Add this paragraph to the current chunk
                current_chunk.append(paragraph)
                current_chunk_size += paragraph_size + 2
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def split_text(self, text):
        """Split the document into chunks based on its structure.
        
        Args:
            text: The document text
            
        Returns:
            list: List of text chunks
        """
        # Find section boundaries
        section_boundaries, lines = self._find_section_boundaries(text)
        
        # Create chunks from sections
        all_chunks = []
        for start, end, title in section_boundaries:
            section_lines = lines[start:end+1]
            chunks = self._create_chunks_from_section(section_lines, title)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def __call__(self, docs):
        """Make the splitter callable directly with documents.
        
        Args:
            docs: List of Document objects with text content
            
        Returns:
            list: List of Document objects representing chunks
        """
        result = []
        for doc in docs:
            # Get the document text
            text = doc.page_content if hasattr(doc, 'page_content') else doc.text
            
            # Split the text into chunks
            text_chunks = self.split_text(text)
            
            # Create new documents for each chunk, preserving metadata
            for i, chunk_text in enumerate(text_chunks):
                # Make a copy of the metadata
                chunk_metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
                
                # Add chunk information to metadata
                chunk_metadata["chunk"] = i
                chunk_metadata["chunk_total"] = len(text_chunks)
                
                # Create a new document with the chunk text and updated metadata
                if hasattr(doc, 'page_content'):
                    # LangChain style document
                    chunk_doc = type(doc)(page_content=chunk_text, metadata=chunk_metadata)
                else:
                    # Assume generic document class with text attribute
                    chunk_doc = type(doc)(text=chunk_text, metadata=chunk_metadata)
                    if hasattr(doc, 'doc_id'):
                        chunk_doc.doc_id = f"{doc.doc_id}_chunk_{i}"
                
                result.append(chunk_doc)
        
        return result

def get_doc_structure_chunker(chunk_size=1024, chunk_overlap=256):
    """Create a document structure-based chunker specifically for academic papers.
    
    Args:
        chunk_size: Maximum size of each chunk in tokens
        chunk_overlap: Overlap between chunks in tokens
        
    Returns:
        AcademicDocStructureSplitter: A structure-aware splitter for academic papers
    """
    return AcademicDocStructureSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)