import base64
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import pdfplumber
import pytesseract
from PIL import Image
from langdetect import detect

from kotaemon.base import Document, Param
from .base import BaseReader
from .utils.adobe import make_markdown_table


class EnhancedDocumentParser(BaseReader):
    """Extract content from native PDFs using pdfplumber, Tesseract OCR, and langdetect"""

    figure_friendly_filetypes: list[str] = Param(
        [".pdf", ".jpeg", ".jpg", ".png", ".bmp", ".tiff", ".heif", ".tif"],
        help="File types we can reliably open and extract figures from.",
    )

    def run(
        self, file_path: str | Path, extra_info: Optional[dict] = None, **kwargs
    ) -> List[Document]:
        return self.load_data(file_path, extra_info, **kwargs)

    def load_data(
        self, file_path: str | Path, extra_info: Optional[dict] = None, **kwargs
    ) -> List[Document]:
        metadata = extra_info or {}
        file_path = Path(file_path)
        file_name = file_path.name
        documents = []

        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract document structure
                font_hierarchy = self.analyze_fonts(page)
                layout_elements = self.extract_layout_elements(page)
                
                # Process tables separately
                tables = self.extract_tables(page)
                
                # Process main text with structure awareness
                structured_blocks = self.detect_document_structure(layout_elements)
                
                # Classify content by type and position
                for block in structured_blocks:
                    content_type = self.classify_content(block["text"], block.get("position"))
                    if content_type == "body":
                        lang, processed_text = self.detect_and_handle_language(block["text"])
                        
                        documents.append(Document(
                            text=processed_text,
                            metadata={
                                "type": "text",
                                "structure_type": block["type"],
                                "heading_level": block.get("level"),
                                "page_label": i + 1,
                                "language": lang,
                                "position": block.get("position"),
                                "file_name": file_name,
                                **metadata,
                            }
                        ))
                
                # Add table documents
                for table in tables:
                    markdown = make_markdown_table(table["table"])
                    documents.append(Document(
                        text=markdown,
                        metadata={
                            "type": "table",
                            "has_header": table["has_header"],
                            "page_label": i + 1,
                            "file_name": file_name,
                            **metadata,
                        }
                    ))
                    
                # Process images and captions
                self.process_images_and_captions(page, documents)
                
                # Extract page thumbnail
                image = page.to_image(resolution=100).original
                if image:
                    img_bytes = BytesIO()
                    image.save(img_bytes, format="PNG")
                    img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
                    
                    documents.append(Document(
                        text="Page thumbnail",
                        metadata={
                            "type": "thumbnail",
                            "image_origin": f"data:image/png;base64,{img_base64}",
                            "page_label": i + 1,
                            "file_name": file_name,
                            **metadata,
                        }
                    ))

        return documents

    def extract_layout_elements(self, page):
        """Extract document elements with layout awareness."""
        # Get page dimensions for relative positioning
        width, height = page.width, page.height
        
        # Extract text with position information
        text_elements = []
        for obj in page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False):
            text_elements.append({
                "type": "text",
                "content": obj["text"],
                "bbox": (obj["x0"]/width, obj["top"]/height, obj["x1"]/width, obj["bottom"]/height),
                "confidence": 1.0
            })
        
        # Group text by lines and paragraphs
        lines = self._group_into_lines(text_elements)
        paragraphs = self._group_into_paragraphs(lines)
        
        return paragraphs

    def extract_tables(self, page):
        """Improved table extraction with border and borderless detection."""
        # Try standard extraction first (bordered tables)
        tables = page.extract_tables()
        
        # If no tables detected, try borderless table detection
        if not tables:
            # Look for text arranged in grid-like patterns
            tables = self._detect_borderless_tables(page)
        
        # Process detected tables
        processed_tables = []
        for table in tables:
            # Clean empty cells and normalize content
            cleaned_table = [[cell.strip() if cell else "" for cell in row] for row in table]
            
            # Add header detection
            has_header = self._detect_header_row(cleaned_table)
            
            processed_tables.append({
                "table": cleaned_table,
                "has_header": has_header
            })
        
        return processed_tables

    def detect_document_structure(self, text_blocks):
        """Detect document structure like headers, lists, etc."""
        structured_blocks = []
        
        for block in text_blocks:
            text = block["text"]
            
            # Detect headers using patterns and font properties
            if self._is_header(text, block.get("font_size", 0)):
                level = self._determine_heading_level(text, block.get("font_size", 0))
                structured_blocks.append({
                    "type": "heading",
                    "level": level,
                    "text": text
                })
                
            # Detect lists
            elif self._is_list_item(text):
                list_type = "numbered" if re.match(r"^\d+[.)]", text) else "bulleted"
                structured_blocks.append({
                    "type": "list_item",
                    "list_type": list_type,
                    "text": text
                })
                
            # Regular paragraph
            else:
                structured_blocks.append({
                    "type": "paragraph",
                    "text": text
                })
                
        return structured_blocks

    def process_images_and_captions(self, page, documents):
        """Extract images and associate them with captions."""
        # Extract images
        images = page.images
        
        for img in images:
            # Convert image data to PIL Image
            image_obj = self._extract_image(img)
            if not image_obj:
                continue
                
            # Get image position
            bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
            
            # Try to find caption (text below the image)
            caption = self._find_caption_for_image(page, bbox)
            
            # Create image document with caption if found
            img_bytes = BytesIO()
            image_obj.save(img_bytes, format="PNG")
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
            
            documents.append(Document(
                text=caption or "Image",
                metadata={
                    "type": "figure",
                    "image_origin": f"data:image/png;base64,{img_base64}",
                    "has_caption": bool(caption),
                    "bbox": bbox,
                    # other metadata...
                }
            ))

    def classify_content(self, text, position=None):
        """Classify content type based on text patterns and position."""
        if not text.strip():
            return "empty"
            
        # Check for page numbers
        if re.match(r"^\s*\d+\s*$", text) and position and (position[1] > 0.9):  # Near bottom
            return "page_number"
            
        # Check for headers/footers
        if position and (position[1] < 0.1 or position[1] > 0.9):
            if len(text.split()) <= 10:  # Short text in header/footer position
                return "header_footer"
        
        # Check for references/citations
        if re.match(r"^\[\d+\]", text) or re.match(r"^[0-9]+\.\s+", text):
            return "reference"
            
        # Default is body text
        return "body"

    def detect_and_handle_language(self, text):
        """Better language detection and handling."""
        if not text.strip():
            return "en", text
            
        try:
            # Use longer text samples for better detection
            sample = text[:min(len(text), 1000)]
            lang = detect(sample)
            
            # Special handling for specific languages
            if lang == "zh-cn" or lang == "zh-tw":
                # Apply Chinese-specific text processing
                processed_text = self._process_chinese_text(text)
                return lang, processed_text
                
            # Handle RTL languages
            if lang in ["ar", "he", "fa"]:
                # Mark as RTL for proper display
                return lang, text
                
            return lang, text
        except:
            return "en", text  # Default to English on error

    def analyze_fonts(self, page):
        """Analyze fonts to detect structure from styling."""
        fonts = defaultdict(list)
        
        # Extract character data with font info
        for char in page.chars:
            fonts[char["fontname"]].append({
                "size": char["size"],
                "text": char["text"],
                "color": char.get("non_stroking_color", [0, 0, 0])
            })
        
        # Determine document hierarchy from font usage
        font_hierarchy = {}
        avg_sizes = {}
        
        for font, chars in fonts.items():
            sizes = [c["size"] for c in chars]
            avg_sizes[font] = sum(sizes) / len(sizes)
        
        # Sort fonts by average size (descending)
        sorted_fonts = sorted(avg_sizes.items(), key=lambda x: x[1], reverse=True)
        
        # Assign hierarchy levels
        for i, (font, _) in enumerate(sorted_fonts):
            font_hierarchy[font] = i
            
        return font_hierarchy
