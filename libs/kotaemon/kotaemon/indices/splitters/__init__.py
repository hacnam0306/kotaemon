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


class SemanticChunker:
    """A semantic-based document chunker that creates chunks based on embedding similarity.
    
    This chunker uses embeddings to group semantically similar text segments together,
    resulting in more coherent chunks compared to pure token-based splitting.
    """
    
    def __init__(self, chunk_size=1024, chunk_overlap=256, embedding_model=None):
        """Initialize the semantic chunker.
        
        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Overlap between chunks in tokens
            embedding_model: Model to use for creating embeddings. If None, attempts to use 
                             a default model.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
        # Import necessary libraries
        import numpy as np
        self.np = np
        import re
        self.re = re
        
        # Try to load a text splitter for initial segmentation
        try:
            from langchain_text_splitters import TokenTextSplitter
            self.token_splitter = TokenTextSplitter(
                chunk_size=min(512, chunk_size), 
                chunk_overlap=min(128, chunk_overlap)
            )
        except ImportError:
            # Fallback to a simpler splitter if LangChain is not available
            self.token_splitter = None
        
        # Set up the embedding model if none was provided
        if self.embedding_model is None:
            try:
                # Try to use SentenceTransformers if available
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Using SentenceTransformer for embeddings")
            except ImportError:
                try:
                    # Try to use the OpenAI embedding API if available
                    from openai import OpenAI
                    client = OpenAI()
                    
                    def get_embedding(text):
                        response = client.embeddings.create(
                            input=text,
                            model="text-embedding-ada-002"
                        )
                        return response.data[0].embedding
                    
                    self.embedding_model = get_embedding
                    print("Using OpenAI embedding API")
                except ImportError:
                    # No suitable embedding model available
                    raise ImportError(
                        "No embedding model available. Please install sentence-transformers "
                        "or provide your own embedding model."
                    )
    
    def _get_embedding(self, text):
        """Get embedding for a text segment.
        
        Args:
            text: Text to embed
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        if hasattr(self.embedding_model, 'encode'):
            # SentenceTransformer style
            return self.embedding_model.encode(text)
        else:
            # Function style
            return self.np.array(self.embedding_model(text))
    
    def _segment_text(self, text):
        """Create initial fine-grained text segments.
        
        Args:
            text: Document text
            
        Returns:
            list: List of small text segments
        """
        if self.token_splitter:
            # Use LangChain's TokenTextSplitter
            return self.token_splitter.split_text(text)
        else:
            # Simple paragraph-based splitting as fallback
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            segments = []
            
            for p in paragraphs:
                # If paragraph is very long, split it further
                if len(p) > 1000:
                    sentences = [s for s in self.re.split(r'(?<=[.!?])\s+', p) if s.strip()]
                    segments.extend(sentences)
                else:
                    segments.append(p)
            
            return segments
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            float: Cosine similarity value (-1 to 1)
        """
        dot_product = self.np.dot(vec1, vec2)
        norm1 = self.np.linalg.norm(vec1)
        norm2 = self.np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)
    
    def _cluster_segments(self, segments, embeddings):
        """Group segments into semantically coherent chunks.
        
        Args:
            segments: List of text segments
            embeddings: List of embedding vectors for each segment
            
        Returns:
            list: List of chunks, where each chunk is a list of segments
        """
        # Estimate tokens per segment
        avg_tokens_per_char = 0.25  # Rough estimate: 4 chars per token
        segment_token_estimates = [int(len(s) * avg_tokens_per_char) for s in segments]
        
        # Initialize chunks
        chunks = []
        current_chunk = []
        current_chunk_tokens = 0
        current_chunk_embedding = None
        
        # Process each segment
        for i, segment in enumerate(segments):
            segment_tokens = segment_token_estimates[i]
            segment_embedding = embeddings[i]
            
            # If this is the first segment or the chunk is empty
            if not current_chunk:
                current_chunk.append(segment)
                current_chunk_tokens = segment_tokens
                current_chunk_embedding = segment_embedding
                continue
            
            # Check if adding this segment would exceed the chunk size
            if current_chunk_tokens + segment_tokens > self.chunk_size:
                # Finalize the current chunk and start a new one
                chunks.append(current_chunk)
                current_chunk = [segment]
                current_chunk_tokens = segment_tokens
                current_chunk_embedding = segment_embedding
                continue
            
            # Check semantic similarity with the current chunk
            similarity = self._cosine_similarity(current_chunk_embedding, segment_embedding)
            
            # If semantically similar enough or the chunk is small, add to current chunk
            if similarity > 0.7 or current_chunk_tokens < self.chunk_size / 2:
                current_chunk.append(segment)
                current_chunk_tokens += segment_tokens
                # Update the chunk embedding as a weighted average of existing and new
                weight = current_chunk_tokens / (current_chunk_tokens + segment_tokens)
                current_chunk_embedding = (
                    weight * current_chunk_embedding + 
                    (1 - weight) * segment_embedding
                )
            else:
                # Start a new chunk due to semantic shift
                chunks.append(current_chunk)
                current_chunk = [segment]
                current_chunk_tokens = segment_tokens
                current_chunk_embedding = segment_embedding
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def split_text(self, text):
        """Split document into semantically coherent chunks.
        
        Args:
            text: Document text
            
        Returns:
            list: List of text chunks
        """
        # Step 1: Create initial fine-grained segments
        segments = self._segment_text(text)
        if not segments:
            return []
            
        # Step 2: Generate embeddings for each segment
        embeddings = [self._get_embedding(segment) for segment in segments]
        
        # Step 3: Cluster segments into coherent chunks
        chunk_groups = self._cluster_segments(segments, embeddings)
        
        # Step 4: Join segments in each chunk
        text_chunks = ['\n\n'.join(group) for group in chunk_groups]
        
        return text_chunks
        
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
                chunk_metadata["chunking_strategy"] = "semantic"
                
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

def get_semantic_chunker(chunk_size=1024, chunk_overlap=256, embedding_model=None):
    """Create a semantic chunker.
    
    Args:
        chunk_size: Target size of each chunk in tokens
        chunk_overlap: Overlap between chunks in tokens
        embedding_model: Model to use for creating embeddings
        
    Returns:
        SemanticChunker: A semantic-aware document chunker
    """
    return SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model
    )






