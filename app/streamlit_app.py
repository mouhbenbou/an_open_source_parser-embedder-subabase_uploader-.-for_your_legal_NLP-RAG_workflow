"""Legal Document Chunk Uploader for Streamlit
A tool for parsing legal documents and uploading them to Supabase for RAG applications.
"""

import streamlit as st
import os
import re
import datetime
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Generator, Set, Tuple, Any, Union
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib

import numpy as np
from supabase import create_client, Client
from transformers import AutoTokenizer, AutoModel
import torch
from tenacity import retry, stop_after_attempt, wait_exponential
import pgvector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_EMBEDDING_LENGTH = 256
BATCH_SIZE = 50
MAX_WORKERS = 4
DEFAULT_TABLE_NAME = "legal_chunks"
EMBEDDING_DIMENSION = 768  # LegalBERT embedding dimension

# Compiled regex patterns for better performance
SECTION_PATTERN =re.compile(r"^(section\s+\d+\s*:\s*[^\n]*)", re.IGNORECASE | re.MULTILINE)
ARTICLE_PATTERN = re.compile(r"^(Article\s+\d+(?:\s+bis)?(?:\s+\d+)?)\s*:\s*", re.IGNORECASE | re.MULTILINE)


class DocumentType(Enum):
    """Enumeration of supported document types."""
    CIVIL_CODE = "civil code"
    CRIMINAL_CODE = "criminal code"
    COMMERCIAL_CODE = "commercial code"
    CONSTITUTION = "constitution"
    REGULATION = "regulation"
    STATUTE = "statute"
    CASE_LAW = "case law"
    TREATY = "treaty"
    DIRECTIVE = "directive"


@dataclass
class LegalChunk:
    """Data class representing a legal document chunk."""
    document_type: str
    jurisdiction: str
    language: str
    section_title: str
    article_number: str
    title: str
    body: str
    tags: List[str] = field(default_factory=list)
    source_url: str = ""
    date_published: Union[datetime.date, str] = ""
    section_title_embedding: Optional[List[float]] = None
    article_embedding: Optional[List[float]] = None
    is_active: bool = True
    chunk_hash: Optional[str] = None
    id: Optional[str] = None  # UUID from database

    def __post_init__(self):
        """Generate a unique hash for the chunk and validate data."""
        # Generate hash
        content = f"{self.document_type}_{self.jurisdiction}_{self.section_title}_{self.article_number}_{self.title}_{self.body}"
        self.chunk_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Ensure date is in correct format
        if isinstance(self.date_published, str) and self.date_published:
            try:
                self.date_published = datetime.datetime.strptime(self.date_published, "%Y-%m-%d").date()
            except ValueError:
                logger.warning(f"Invalid date format: {self.date_published}, using today's date")
                self.date_published = datetime.date.today()
        elif not self.date_published:
            self.date_published = datetime.date.today()

    def validate(self) -> List[str]:
        """Validate the chunk data and return list of errors."""
        errors = []
        
        # Check required fields
        if not self.document_type:
            errors.append("document_type is required")
        if not self.jurisdiction:
            errors.append("jurisdiction is required")
        if not self.language:
            errors.append("language is required")
        if not self.section_title:
            errors.append("section_title is required")
        if not self.article_number:
            errors.append("article_number is required")
        if not self.body:
            errors.append("body is required")
        if not self.source_url:
            errors.append("source_url is required")
            
        # Validate embeddings
        if self.section_title_embedding and len(self.section_title_embedding) != EMBEDDING_DIMENSION:
            errors.append(f"section_title_embedding must have {EMBEDDING_DIMENSION} dimensions")
        if self.article_embedding and len(self.article_embedding) != EMBEDDING_DIMENSION:
            errors.append(f"article_embedding must have {EMBEDDING_DIMENSION} dimensions")
            
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert the chunk to a dictionary for database insertion."""
        # Format embeddings as strings for pgvector
        section_embedding_str = None
        article_embedding_str = None
        
        if self.section_title_embedding:
            section_embedding_str = f"[{','.join(map(str, self.section_title_embedding))}]"
        
        if self.article_embedding:
            article_embedding_str = f"[{','.join(map(str, self.article_embedding))}]"
        
        return {
            "document_type": self.document_type,
            "jurisdiction": self.jurisdiction.lower(),
            "language": self.language.lower(),
            "section_title": self.section_title,
            "article_number": self.article_number,
            "title": self.title,
            "body": self.body,
            "tags": self.tags if self.tags else [],
            "source_url": self.source_url,
            "date_published": self.date_published.isoformat() if isinstance(self.date_published, datetime.date) else self.date_published,
            "section_title_embedding": section_embedding_str,
            "article_embedding": article_embedding_str,
            "is_active": self.is_active
        }


class EmbeddingGenerator:
    """Handles the generation of embeddings using LegalBERT."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None
        logger.info(f"EmbeddingGenerator initialized for device: {self.device}")
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer, self._model = self._load_model()
        return self._tokenizer
    
    @property
    def model(self):
        if self._model is None:
            self._tokenizer, self._model = self._load_model()
        return self._model
    
    @st.cache_resource
    def _load_model(_self) -> Tuple[AutoTokenizer, AutoModel]:
        """Load the LegalBERT model with caching."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(_self.model_name)
            model = AutoModel.from_pretrained(_self.model_name).to(_self.device)
            model.eval()  # Set to evaluation mode
            logger.info(f"Successfully loaded model: {_self.model_name}")
            return tokenizer, model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load model {_self.model_name}: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of float values representing the embedding
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * EMBEDDING_DIMENSION
        
        try:
            with torch.no_grad():
                # Tokenize with proper truncation
                encoded = self.tokenizer(
                    text, 
                    truncation=True, 
                    max_length=MAX_EMBEDDING_LENGTH, 
                    padding='max_length',
                    return_tensors="pt"
                )
                
                # Move tensors to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Generate embeddings
                output = self.model(**encoded)
                
                # Use CLS token embedding (first token)
                embedding = output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                
                # Normalize the embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                return embedding.astype(np.float32).tolist()
                
        except Exception as e:
            logger.error(f"Error generating embedding for text: {text[:50]}... Error: {e}")
            raise


class LegalDocumentParser:
    """Parses legal documents into structured chunks."""
    
    def __init__(self):
        # More comprehensive regex patterns
        self.section_pattern = re.compile(r"^(section\s+\d+\s*:\s*[^\n]*)", re.IGNORECASE | re.MULTILINE)
        self.article_pattern = re.compile(r"^(Article\s+\d+(?:\s+bis)?(?:\s+\d+)?)\s*:\s*", re.IGNORECASE | re.MULTILINE)
  
    def validate_metadata(self, **kwargs) -> Dict[str, Any]:
        """Validate and sanitize metadata inputs."""
        validated = {}
        
        # Validate jurisdiction code
        jurisdiction = kwargs.get('jurisdiction', '').lower().strip()
        if not re.match(r'^[a-z]{2,3}$', jurisdiction):
            raise ValueError(f"Invalid jurisdiction code: {jurisdiction}. Must be 2-3 lowercase letters.")
        validated['jurisdiction'] = jurisdiction
        
        # Validate language code
        language = kwargs.get('language', '').lower().strip()
        if not re.match(r'^[a-z]{2}$', language):
            raise ValueError(f"Invalid language code: {language}. Must be 2 lowercase letters.")
        validated['language'] = language
        
        # Validate document type
        document_type = kwargs.get('document_type', '').strip()
        if not document_type:
            raise ValueError("Document type cannot be empty.")
        validated['document_type'] = document_type
        
        # Validate URL
        source_url = kwargs.get('source_url', '').strip()
        if not source_url:
            raise ValueError("Source URL is required.")
        if not source_url.startswith(('http://', 'https://')):
            raise ValueError("Source URL must start with http:// or https://")
        validated['source_url'] = source_url
        
        # Validate date
        date_published = kwargs.get('date_published', datetime.date.today())
        if isinstance(date_published, str):
            try:
                date_published = datetime.datetime.strptime(date_published, "%Y-%m-%d").date()
            except ValueError:
                raise ValueError(f"Invalid date format: {date_published}. Use YYYY-MM-DD format.")
        validated['date_published'] = date_published
        
        return validated
    
    def parse_legal_file(self, text: str, **metadata) -> Generator[LegalChunk, None, None]:
        """
        Parses a legal text file into structured LegalChunk objects.
        
        Fixed version that correctly handles the document structure and prevents data loss.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for parsing")
            return

        try:
            validated_metadata = self.validate_metadata(**metadata)
        except ValueError as e:
            logger.error(f"Metadata validation failed: {e}")
            raise

        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Find all section matches with their positions
        section_matches = list(self.section_pattern.finditer(text))
        
        if not section_matches:
            logger.warning("No section markers found. Treating entire document as a single section.")
            # If no sections found, treat whole document as one section and parse articles
            yield from self._parse_section_content("General Provisions", text, validated_metadata)
            return
        
        # Process each section
        for i, section_match in enumerate(section_matches):
            section_title = section_match.group(1).strip()
            
            # Determine section content boundaries
            section_start = section_match.end()
            if i + 1 < len(section_matches):
                section_end = section_matches[i + 1].start()
            else:
                section_end = len(text)
            
            section_content = text[section_start:section_end].strip()
            
            if section_content:
                yield from self._parse_section_content(section_title, section_content, validated_metadata)
            else:
                logger.warning(f"Empty section found: {section_title}")
    
    def _parse_section_content(self, section_title: str, content: str, metadata: Dict[str, Any]) -> Generator[LegalChunk, None, None]:
        """Parse the content within a section to extract articles."""
        
        # Find all article matches with their positions
        article_matches = list(self.article_pattern.finditer(content))
        
        if not article_matches:
            # No articles found, treat entire content as one chunk
            if content.strip():
                chunk = LegalChunk(
                    section_title=section_title,
                    article_number="General Content",
                    title="",
                    body=content.strip(),
                    tags=self._extract_tags(content),
                    **metadata
                )
                errors = chunk.validate()
                if not errors:  # Only yield if no validation errors
                    yield chunk
                else:
                    logger.warning(f"Chunk validation failed: {errors}")
            return
        
        # Handle content before the first article (if any)
        first_article_start = article_matches[0].start()
        if first_article_start > 0:
            intro_content = content[:first_article_start].strip()
            if intro_content:
                chunk = LegalChunk(
                    section_title=section_title,
                    article_number="Section Introduction",
                    title="",
                    body=intro_content,
                    tags=self._extract_tags(intro_content),
                    **metadata
                )
                errors = chunk.validate()
                if not errors:
                    yield chunk
                else:
                    logger.warning(f"Intro chunk validation failed: {errors}")
        
        # Process each article
        for i, article_match in enumerate(article_matches):
            article_number = article_match.group(1).strip()
            
            # Determine article content boundaries
            article_content_start = article_match.end()
            if i + 1 < len(article_matches):
                article_content_end = article_matches[i + 1].start()
            else:
                article_content_end = len(content)
            
            article_body = content[article_content_start:article_content_end].strip()
            
            if article_body:
                chunk = LegalChunk(
                    section_title=section_title,
                    article_number=article_number,
                    title="",  # Title is intentionally left blank
                    body=article_body,
                    tags=self._extract_tags(article_body),
                    **metadata
                )
                errors = chunk.validate()
                if not errors:
                    yield chunk
                else:
                    logger.warning(f"Article chunk validation failed for {article_number}: {errors}")
            else:
                logger.warning(f"Empty article content found: {article_number}")
    

    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from the text."""
        # Predefined legal tags
        legal_tags = {
            # Core legal concepts
            "contract", "agreement", "obligation", "liability", "damages", "laws",
            "compensation", "dispute", "arbitration", "compliance", "regulation",
            "rights", "duties", "property", "ownership", "penalty", "punishment",
            "jurisdiction", "enforcement", "breach", "termination", "warranty",
            "indemnity", "negligence", "tort", "remedy", "injunction", "crime",

            # Family and civil law
            "family", "marriage", "divorce", "custody", "adoption", "human rights",
            "inheritance", "succession", "guardianship", "minor", "parental rights",

            # Commercial and corporate law
            "sale", "purchase", "partnership", "company", "corporation", "shareholder",
            "director", "merger", "acquisition", "bankruptcy", "insolvency",

            # Procedural and evidentiary terms
            "evidence", "proof", "witness", "testimony", "court", "judge", "lawsuit",
            "prescription", "limitation", "appeal", "verdict", "hearing", "trial",

            # Financial and contractual mechanisms
            "debtor", "creditor", "payment", "performance", "rescission", "annulment",
            "fraud", "duress", "capacity", "representation", "agency", "power",

            # Personal status and identity
            "domicile", "residence", "nationality", "citizenship", "legal person",
            "natural person", "good faith", "public order", "morals",

            # International and digital law
            "treaty", "protocol", "jurisprudence", "extradition", "cybercrime",
            "data protection", "privacy", "intellectual property", "copyright",
            "trademark", "patent", "licensing", "terms of service", "digital rights",

            # Administrative and constitutional law
            "statute", "ordinance", "decree", "constitution", "amendment",
            "legislation", "executive", "judicial", "legislative", "sovereignty",

            # Labor and employment law
            "employment", "labor", "contractor", "employee", "employer", "wages",
            "benefits", "termination", "discrimination", "harassment", "union",

            # Environmental and public law
            "environment", "pollution", "sustainability", "public health",
            "regulatory compliance", "licensing", "zoning", "land use"

        }

        tags = []
        text_lower = text.lower()
        
        # Add matching predefined tags
        for tag in legal_tags:
            if tag in text_lower:
                tags.append(tag)

        # Extract legal references (e.g., "Article 123", "Section 45")
        legal_refs = re.findall(r'\b(?:Article|Section|Clause|Paragraph)\s+\d+(?:\s+bis)?(?:\s+\d+)?\b', text, re.IGNORECASE)
        tags.extend([ref.lower() for ref in legal_refs[:3]])  # Limit to 3 references
        
        # Extract important legal terms that are capitalized
        caps_terms = re.findall(r'\b[A-Z][A-Z]+(?:\s+[A-Z][A-Z]+)*\b', text)
        for term in caps_terms[:3]:  # Limit to 3 terms
            if 2 < len(term) < 20:  # Reasonable length for a tag
                tags.append(term.lower())
        
        # Remove duplicates and limit total tags
        unique_tags = list(dict.fromkeys(tags))  # Preserves order
        return unique_tags[:10]  # Limit to 10 tags total

class SupabaseUploader:
    """Handles uploading chunks to Supabase with retry logic and batch processing."""
    
    def __init__(self, client: Client, table_name: str = DEFAULT_TABLE_NAME):
        self.client = client
        self.table_name = table_name
    
    def check_existing_chunks(self, chunk_hashes: List[str]) -> Set[str]:
        """Check which chunks already exist in the database."""
        try:
            # Query in batches to avoid too large queries
            existing_hashes = set()
            batch_size = 100
            
            for i in range(0, len(chunk_hashes), batch_size):
                batch = chunk_hashes[i:i + batch_size]
                response = self.client.table(self.table_name)\
                    .select("chunk_hash")\
                    .in_("chunk_hash", batch)\
                    .execute()
                
                if response.data:
                    existing_hashes.update(item['chunk_hash'] for item in response.data)
            
            return existing_hashes
        except Exception as e:
            logger.error(f"Error checking existing chunks: {e}")
            return set()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _upload_batch(self, batch: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """
        Upload a batch of chunks with retry logic.
        
        Returns:
            Tuple of (success_count, error_messages)
        """
        try:
            # Remove any existing IDs to let database generate them
            for item in batch:
                item.pop('id', None)
                item.pop('created_at', None)
                item.pop('updated_at', None)
            
            response = self.client.table(self.table_name).insert(batch).execute()
            
            if response.data:
                return len(response.data), []
            else:
                return 0, ["No data returned from insert operation"]
                
        except Exception as e:
            logger.error(f"Batch upload failed: {e}")
            error_msg = str(e)
            
            # Parse specific database errors
            if "duplicate key" in error_msg.lower():
                return 0, ["Duplicate chunks detected - some chunks already exist in database"]
            elif "violates not-null constraint" in error_msg.lower():
                return 0, ["Missing required fields in chunks"]
            elif "invalid input syntax for type vector" in error_msg.lower():
                return 0, ["Invalid embedding format"]
            else:
                return 0, [error_msg]
    
    def upload_chunks_batch(
        self, 
        chunks: List[LegalChunk], 
        embedding_generator: EmbeddingGenerator,
        progress_callback=None,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Upload chunks in batches with progress tracking.
        
        Args:
            chunks: List of LegalChunk objects to upload
            embedding_generator: EmbeddingGenerator instance
            progress_callback: Optional callback for progress updates
            skip_existing: Whether to skip chunks that already exist
            
        Returns:
            Dictionary with upload statistics
        """
        total_chunks = len(chunks)
        uploaded = 0
        failed = 0
        skipped = 0
        errors = []
        
        # Check for existing chunks if skip_existing is True
        existing_hashes = set()
        if skip_existing:
            chunk_hashes = [chunk.chunk_hash for chunk in chunks]
            existing_hashes = self.check_existing_chunks(chunk_hashes)
            skipped = len(existing_hashes)
            
            if skipped > 0:
                logger.info(f"Found {skipped} existing chunks that will be skipped")
        
        # Filter out existing chunks
        chunks_to_process = [
            chunk for chunk in chunks 
            if chunk.chunk_hash not in existing_hashes
        ] if skip_existing else chunks
        
        if not chunks_to_process:
            return {
                "total": total_chunks,
                "uploaded": 0,
                "failed": 0,
                "skipped": skipped,
                "errors": ["All chunks already exist in database"]
            }
        
        # Generate embeddings concurrently
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit embedding tasks
            future_to_chunk = {}
            
            for chunk in chunks_to_process:
                future_section = executor.submit(
                    embedding_generator.get_embedding, 
                    chunk.section_title
                )
                future_article = executor.submit(
                    embedding_generator.get_embedding, 
                    f"{chunk.title}\n{chunk.body}"
                )
                future_to_chunk[(future_section, future_article)] = chunk
            
            # Process completed embeddings
            processed_chunks = []
            
            for idx, ((future_section, future_article), chunk) in enumerate(future_to_chunk.items()):
                try:
                    chunk.section_title_embedding = future_section.result()
                    chunk.article_embedding = future_article.result()
                    
                    # Validate embeddings
                    if len(chunk.section_title_embedding) != EMBEDDING_DIMENSION:
                        raise ValueError(f"Section embedding has wrong dimension: {len(chunk.section_title_embedding)}")
                    if len(chunk.article_embedding) != EMBEDDING_DIMENSION:
                        raise ValueError(f"Article embedding has wrong dimension: {len(chunk.article_embedding)}")
                    
                    processed_chunks.append(chunk.to_dict())
                    
                    if progress_callback and idx % 10 == 0:
                        progress = (idx / len(chunks_to_process)) * 0.5  # First 50% for embedding
                        progress_callback(progress, f"Generating embeddings... {idx}/{len(chunks_to_process)}")
                        
                except Exception as e:
                    logger.error(f"Failed to generate embeddings for chunk {chunk.article_number}: {e}")
                    errors.append(f"Embedding error for {chunk.article_number}: {str(e)}")
                    failed += 1
        
        # Upload in batches
        for i in range(0, len(processed_chunks), BATCH_SIZE):
            batch = processed_chunks[i:i + BATCH_SIZE]
            success_count, batch_errors = self._upload_batch(batch)
            
            uploaded += success_count
            failed += len(batch) - success_count
            errors.extend(batch_errors)
            
            if progress_callback:
                progress = 0.5 + ((i + len(batch)) / len(processed_chunks)) * 0.5  # Second 50% for upload
                progress_callback(progress, f"Uploading... {uploaded}/{len(processed_chunks)} chunks")
        
        return {
            "total": total_chunks,
            "uploaded": uploaded,
            "failed": failed,
            "skipped": skipped,
            "errors": errors[:10]  # Limit error messages
        }


def create_streamlit_ui():
    """Create the Streamlit user interface."""
    st.set_page_config(
        page_title="Legal Document Chunk Uploader",
        page_icon="⚖️",
        layout="wide"
    )
    # ADD THE CSS HERE - RIGHT AFTER st.set_page_config()
    st.markdown("""
    <style>
    /* Luxurious Court Theme CSS for Legal Document Uploader */

    /* Main background with animated particles and legal motifs */
    .stApp {
        background: radial-gradient(circle at center, #0a0a0a, #111122, #0f0f25);
        position: relative;
        overflow: hidden;
        min-height: 100vh;
    }


    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, #0f0f25, #1a1a2e, #0f3460, #0a0a0a);
        background-size: 400% 400%;
        animation: moveGradient 100s ease infinite;
        z-index: -2;
        opacity: 0.6;
    }


    .stApp::after {
        content: '⚖️ § ⚖️ § ⚖️ § ⚖️ § ⚖️ § ⚖️ § ⚖️ § ⚖️ § ⚖️';
        position: fixed;
        top: 0;
        left: -100%;
        width: 300%;
        font-size: 22px;
        line-height: 100px;
        color: gold;
        opacity: 0.03;
        animation: legalSymbolsFloat 60s linear infinite;
        z-index: -1;
        white-space: nowrap;
        pointer-events: none;
    }


    @keyframes backgroundShift {
        0%, 100% { 
            background-position: 0% 0%, 100% 100%, 50% 50%; 
            filter: hue-rotate(0deg);
        }
        33% { 
            background-position: 100% 0%, 0% 100%, 25% 75%; 
            filter: hue-rotate(120deg);
        }
        66% { 
            background-position: 50% 100%, 100% 0%, 75% 25%; 
            filter: hue-rotate(240deg);
        }
    }

    @keyframes moveGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes legalSymbolsFloat {
        0% { transform: translateX(-100%) translateY(0px) rotate(0deg); }
        50% { transform: translateX(100%) translateY(-20px) rotate(180deg); }
        100% { transform: translateX(-100%) translateY(0px) rotate(360deg); }
    }


    /* Animated floating elements */
    .stApp .floating-elements {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }

    /* Create floating legal elements with CSS */
    .stApp .floating-elements::before,
    .stApp .floating-elements::after {
        content: '';
        position: absolute;
        width: 2px;
        height: 2px;
        background: radial-gradient(circle, rgba(255, 215, 0, 0.4), transparent);
        border-radius: 50%;
        box-shadow:
            0 0 6px rgba(255, 215, 0, 0.3),
            0 0 12px rgba(255, 215, 0, 0.2);
        animation: floatingParticles 25s linear infinite;
        z-index: -1;
    }

    .stApp .floating-elements::before {
        left: 10%;
        animation-delay: -8s;
        animation-duration: 18s;
    }

    .stApp .floating-elements::after {
        left: 85%;
        animation-delay: -12s;
        animation-duration: 22s;
    }

    @keyframes floatingParticles {
        0% {
            transform: translateY(100vh) translateX(0px) rotate(0deg) scale(0);
            opacity: 0;
        }
        10% {
            opacity: 1;
            scale: 1;
        }
        90% {
            opacity: 1;
            scale: 1;
        }
        100% {
            transform: translateY(-10vh) translateX(100px) rotate(360deg) scale(0);
            opacity: 0;
        }
    }

    /* Title glowing effect */
    .stApp h1 {
        text-shadow: 
            0 0 5px rgba(184, 134, 11, 0.5),
            0 0 10px rgba(184, 134, 11, 0.3),
            0 0 15px rgba(184, 134, 11, 0.2),
            0 0 20px rgba(184, 134, 11, 0.1);
        animation: titleGlow 3s ease-in-out infinite alternate;
        font-weight: 700;
        background: linear-gradient(45deg, #b8860b, #daa520, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    @keyframes titleGlow {
        from { 
            text-shadow: 
                0 0 5px rgba(184, 134, 11, 0.3),
                0 0 10px rgba(184, 134, 11, 0.2);
        }
        to { 
            text-shadow: 
                0 0 8px rgba(184, 134, 11, 0.6),
                0 0 16px rgba(184, 134, 11, 0.4),
                0 0 24px rgba(184, 134, 11, 0.2);
        }
    }

    /* Sidebar glowing effect */
    .stApp .css-1d391kg {
        background: rgba(10, 10, 10, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(184, 134, 11, 0.2);
        border-radius: 15px;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(184, 134, 11, 0.1);
    }

    /* Form containers */
    .stApp .stForm {
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(184, 134, 11, 0.3);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(184, 134, 11, 0.1);
        transition: all 0.3s ease;
    }

    .stApp .stForm:hover {
        border-color: rgba(184, 134, 11, 0.5);
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.5),
            0 0 20px rgba(184, 134, 11, 0.1);
        transform: translateY(-2px);
    }

    /* Button glowing effects */
    .stApp button {
        background: linear-gradient(45deg, rgba(184, 134, 11, 0.8), rgba(218, 165, 32, 0.8));
        border: 1px solid rgba(184, 134, 11, 0.6);
        border-radius: 8px;
        color: white;
        font-weight: 600;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
        box-shadow: 
            0 4px 15px rgba(184, 134, 11, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .stApp button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .stApp button:hover {
        background: linear-gradient(45deg, rgba(184, 134, 11, 1), rgba(218, 165, 32, 1));
        box-shadow: 
            0 6px 20px rgba(184, 134, 11, 0.5),
            0 0 30px rgba(184, 134, 11, 0.3);
        transform: translateY(-2px);
        border-color: rgba(184, 134, 11, 0.8);
    }

    .stApp button:hover::before {
        left: 100%;
    }

    /* Input fields glowing */
    .stApp input, .stApp select, .stApp textarea {
        background: rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(184, 134, 11, 0.3);
        border-radius: 6px;
        color: white;
        transition: all 0.3s ease;
    }

    .stApp input:focus, .stApp select:focus, .stApp textarea:focus {
        border-color: rgba(184, 134, 11, 0.7);
        box-shadow: 
            0 0 0 2px rgba(184, 134, 11, 0.2),
            0 4px 12px rgba(184, 134, 11, 0.1);
        background: rgba(0, 0, 0, 0.8);
    }
    
    /* Data frame styling */
    .stApp .stDataFrame {
        background: rgba(0, 0, 0, 0.8);
        border-radius: 10px;
        border: 1px solid rgba(184, 134, 11, 0.2);
        overflow: hidden;
    }

    /* Metric cards */
    .stApp .metric-container {
        background: rgba(0, 0, 0, 0.7);
        border: 1px solid rgba(184, 134, 11, 0.3);
        border-radius: 10px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .stApp .metric-container:hover {
        background: rgba(0, 0, 0, 0.8);
        border-color: rgba(184, 134, 11, 0.5);
        box-shadow: 0 4px 15px rgba(184, 134, 11, 0.2);
        transform: translateY(-3px);
    }

    /* Progress bar */
    .stApp .stProgress > div > div > div {
        background: linear-gradient(90deg, #b8860b, #daa520, #ffd700);
        box-shadow: 0 2px 10px rgba(184, 134, 11, 0.4);
    }

    /* Success/Error messages with glow */
    .stApp .stSuccess {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.2);
    }
    
    .stApp .stError {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);
    }

    .stApp .stWarning {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.2);
    }

    .stApp .stInfo {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }

    /* Hover effects for main containers */
    .stApp .block-container:hover {
        background: rgba(255, 255, 255, 0.02);
        transition: background 0.3s ease;
    }

    /* Expander styling */
    .stApp .streamlit-expanderHeader {
        background: rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(184, 134, 11, 0.2);
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stApp .streamlit-expanderHeader:hover {
        border-color: rgba(184, 134, 11, 0.4);
        background: rgba(0, 0, 0, 0.8);
        box-shadow: 0 2px 8px rgba(184, 134, 11, 0.1);
    }

    /* File uploader styling */
    .stApp .uploadedFile {
        background: rgba(0, 0, 0, 0.7);
        border: 1px solid rgba(184, 134, 11, 0.3);
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stApp .uploadedFile:hover {
        border-color: rgba(184, 134, 11, 0.5);
        box-shadow: 0 4px 12px rgba(184, 134, 11, 0.2);
    }

    /* Copyright footer */
    .copyright-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(90deg, 
            rgba(0, 0, 0, 0.8) 0%, 
            rgba(26, 26, 46, 0.9) 50%, 
            rgba(0, 0, 0, 0.8) 100%);
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(184, 134, 11, 0.3);
        padding: 8px 20px;
        z-index: 1000;
        font-size: 12px;
        color: rgba(255, 255, 255, 0.7);
        text-align: center;
        box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .copyright-footer a {
        color: rgba(184, 134, 11, 0.8);
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
        text-shadow: 0 0 5px rgba(184, 134, 11, 0.3);
    }
    
    .copyright-footer a:hover {
        color: rgba(218, 165, 32, 1);
        text-shadow: 0 0 10px rgba(184, 134, 11, 0.6);
    }
    
    /* Scrollbar styling */
    .stApp ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    .stApp ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 4px;
    }
    
    .stApp ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, rgba(184, 134, 11, 0.6), rgba(218, 165, 32, 0.6));
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    
    .stApp ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, rgba(184, 134, 11, 0.8), rgba(218, 165, 32, 0.8));
        box-shadow: 0 0 10px rgba(184, 134, 11, 0.4);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stApp h1 {
            font-size: 1.8rem;
        }
        
        .copyright-footer {
            font-size: 10px;
            padding: 6px 10px;
        }
        
        .stApp .floating-elements::before,
        .stApp .floating-elements::after {
            display: none; /* Hide particles on mobile for performance */
        }
    }
    
    /* Dark mode text colors */
    .stApp, .stApp * {
        color: rgba(255, 255, 255, 0.9);
    }

    .stApp .stMarkdown, .stApp .stText {
        color: rgba(255, 255, 255, 0.85);
    }

    /* Table styling */
    .stApp table {
        background: rgba(0, 0, 0, 0.6);
        border-radius: 8px;
        overflow: hidden;
    }

    .stApp th {
        background: rgba(184, 134, 11, 0.2);
        color: rgba(255, 255, 255, 0.9);
        font-weight: 600;
    }

    .stApp td {
        border-color: rgba(184, 134, 11, 0.1);
    }

    /* Interactive hover zones */
    .stApp .element-container:hover {
        transition: all 0.3s ease;
    }
    
    /* Special glow for important elements */
    .stApp .stButton > button[kind="primary"] {
        background: linear-gradient(45deg, 
            rgba(147, 51, 234, 0.8), 
            rgba(168, 85, 247, 0.8));
        box-shadow: 
            0 4px 15px rgba(147, 51, 234, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    .stApp .stButton > button[kind="primary"]:hover {
        box-shadow: 
            0 6px 20px rgba(147, 51, 234, 0.6),
            0 0 30px rgba(147, 51, 234, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)


    st.title("⚖️ Legal Document Chunk Uploader (RAG-ready)")
    st.markdown("""
    Upload legal documents and automatically parse them into chunks for Retrieval-Augmented Generation (RAG) applications.
    
    **Features:**
    -  Automatic section and article detection
    -  LegalBERT embeddings for semantic search
    -  Duplicate detection and prevention
    -  Batch processing with progress tracking
    -  Supabase pgvector integration

    **IMPORTANT NOTES**:
    -  The table designated for storing parsed and embedded legal content must be pre-created in your Supabase instance.
    -  Ensure that the table schema matches the expected structure (explained in the README file in the app's github repository.
    -  THE LEGAL DOCUMENTS SHPULD BE IN SECTIONS AND ARTICLES ONLY FOR EXAMPLE:
          section x : rights of ownership
          article z : ..... 
          article y : ....
    -  Both section titles and article headings may contain bracketed content.
    -  This project is licensed under the Apache License 2.0 .
    """)

    
    # Initialize session state
    if 'chunks' not in st.session_state:
        st.session_state.chunks = None
    if 'upload_stats' not in st.session_state:
        st.session_state.upload_stats = None
    if 'embedding_generator' not in st.session_state:
        st.session_state.embedding_generator = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Supabase credentials
        st.subheader("Supabase Settings")
        supabase_url = st.text_input(
            "Supabase Project URL",
            value=os.getenv("SUPABASE_URL", ""),
            help="Your Supabase project URL (e.g., https://xxx.supabase.co)"
        )
        supabase_key = st.text_input(
            "Supabase Service Key",
            type="password",
            value=os.getenv("SUPABASE_KEY", ""),
            help="Your Supabase service role key (keep this secret!)"
        )
        table_name = st.text_input(
            "Table Name",
            value=DEFAULT_TABLE_NAME,
            help="Name of the Supabase table to upload to"
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            batch_size = st.number_input(
                "Batch Size",
                min_value=10,
                max_value=200,
                value=BATCH_SIZE,
                help="Number of chunks to upload in each batch"
            )
            max_workers = st.number_input(
                "Max Workers",
                min_value=1,
                max_value=10,
                value=MAX_WORKERS,
                help="Number of concurrent workers for embedding generation"
            )
            skip_existing = st.checkbox(
                "Skip Existing Chunks",
                value=True,
                help="Skip chunks that already exist in the database"
            )
            
        # Model settings
        with st.expander("Model Settings"):
            model_name = st.text_input(
                "Embedding Model",
                value=DEFAULT_MODEL_NAME,
                help="Hugging Face model name for embeddings"
            )
            
            if st.button("Load Model"):
                with st.spinner("Loading embedding model..."):
                    try:
                        st.session_state.embedding_generator = EmbeddingGenerator(model_name)
                        st.success(" Model loaded successfully!")
                    except Exception as e:
                        st.error(f"Failed to load model: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(" Document Upload")
        
        with st.form("upload_form"):
            # File upload
            uploaded_file = st.file_uploader(
                "Choose a legal text file",
                type=["txt", "md"],
                help="Upload a text file containing legal document content"
            )
            
            # Metadata inputs
            st.subheader("Document Metadata")
            
            col_meta1, col_meta2 = st.columns(2)
            
            with col_meta1:
                document_type = st.selectbox(
                    "Document Type",
                    options=[dt.value for dt in DocumentType],
                    index=0
                )
                jurisdiction = st.text_input(
                    "Jurisdiction Code",
                    value="us",
                    max_chars=3,
                    help="2-3 letter jurisdiction code (e.g., 'us', 'uk', 'dz')"
                ).lower()
                language = st.text_input(
                    "Language Code",
                    value="en",
                    max_chars=2,
                    help="2-letter ISO language code (e.g., 'en', 'fr', 'ar')"
                ).lower()
            
            with col_meta2:
                source_url = st.text_input(
                    "Source URL",
                    value="",
                    help="URL where the document can be found (required)"
                )
                date_published = st.date_input(
                    "Date Published",
                    value=datetime.date.today(),
                    help="Publication date of the document"
                )
            
            # Submit button
            submitted = st.form_submit_button(
                " Parse Document",
                use_container_width=True
            )
        
        # Process uploaded file
        if submitted and uploaded_file:
            # Validate inputs
            if not source_url:
                st.error("Source URL is required")
            elif not source_url.startswith(('http://', 'https://')):
                st.error("Source URL must start with http:// or https://")
            else:
                try: 
                    
                    # Read file content
                    text = uploaded_file.read().decode("utf-8", errors="ignore")
                    
                    # Initialize parser
                    parser = LegalDocumentParser()
                    
                    # Parse document
                    with st.spinner("Parsing document..."):
                        chunks = list(parser.parse_legal_file(
                            text,
                            document_type=document_type,
                            jurisdiction=jurisdiction,
                            language=language,
                            source_url=source_url,
                            date_published=date_published
                        ))
                    
                    if chunks:
                        st.success(f" Successfully parsed {len(chunks)} chunks")
                        st.session_state.chunks = chunks
                        
                        # Display preview
                        st.subheader(" Preview")
                        
                        # Summary statistics
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("Total Chunks", len(chunks))
                        with col_stat2:
                            unique_sections = len(set(chunk.section_title for chunk in chunks))
                            st.metric("Unique Sections", unique_sections) 
                        with col_stat3:
                            avg_chunk_size = sum(len(chunk.body) for chunk in chunks) // len(chunks)
                            st.metric("Avg Chunk Size", f"{avg_chunk_size} chars")
                        
                        # Preview table
                        st.write("First 10 chunks:")
                        preview_data = []
                        for chunk in chunks[:10]:
                            preview_data.append({
                                "Section": chunk.section_title[:50] + "..." if len(chunk.section_title) > 50 else chunk.section_title,
                                "Article": chunk.article_number,
                                "Title": chunk.title[:50] + "..." if len(chunk.title) > 50 else chunk.title,
                                "Body Preview": chunk.body[:100] + "..." if len(chunk.body) > 100 else chunk.body,
                                "Tags": ", ".join(chunk.tags[:3]) if chunk.tags else "None",
                                "Hash": chunk.chunk_hash[:8] + "..."
                            })
                        
                        st.dataframe(preview_data, use_container_width=True)
                        
                        # Download parsed chunks as JSON
                        chunks_json = json.dumps([chunk.to_dict() for chunk in chunks], indent=2, default=str)
                        # Download parsed chunks as JSON
                        chunks_json = json.dumps([chunk.to_dict() for chunk in chunks], indent=2, default=str)
                        st.download_button(
                            label=" Download Parsed Chunks (JSON)",
                            data=chunks_json,
                            file_name="parsed_chunks.json",
                            mime="application/json"
                        )
                    else:
                        st.warning(" No valid chunks found in document")
                except Exception as e:
                    st.error(f" Document parsing failed: {str(e)}")
                    logger.exception("Document parsing error")
    
    with col2:
        st.header(" Upload to Supabase")
        
        if st.session_state.chunks is None:
            st.info("Parse a document first to enable upload")
        else:
            if st.button(" Upload Chunks", use_container_width=True):
                if not supabase_url or not supabase_key:
                    st.error(" Supabase credentials are required!")
                else:
                    try:
                        # Initialize clients
                        supabase_client = create_client(supabase_url, supabase_key)
                        uploader = SupabaseUploader(supabase_client, table_name)
                        
                        # Ensure embedding generator exists
                        if st.session_state.embedding_generator is None:
                            with st.spinner(" Loading embedding model..."):
                                st.session_state.embedding_generator = EmbeddingGenerator(model_name)
                        
                        # Setup progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def update_progress(progress, message):
                            progress_bar.progress(progress)
                            status_text.text(message)
                        
                        # Start upload
                        stats = uploader.upload_chunks_batch(
                            st.session_state.chunks,
                            st.session_state.embedding_generator,
                            progress_callback=update_progress,
                            skip_existing=skip_existing
                        )
                        
                        # Show results
                        st.success(f" Upload complete! Uploaded: {stats['uploaded']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
                        
                        if stats['errors']:
                            st.error(" Errors occurred:")
                            for error in stats['errors']:
                                st.error(error)
                                
                    except Exception as e:
                        st.error(f" Upload failed: {str(e)}")
                        logger.exception("Upload error")
        # Spacer to push footer down
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Copyright footer - GUARANTEED to be at the bottom
    st.markdown("""
    <div class="copyright-footer-alt">
        © 2024 Legal Document Uploader | Developed by <strong>Mohamed Benbouchama El Kamel</strong> | 
        Licensed under <a href="https://www.apache.org/licenses/LICENSE-2.0" target="_blank">Apache License 2.0</a> | 
        ⚖️ Empowering Legal Technology
    </div>
    """, unsafe_allow_html=True)


def main():
    create_streamlit_ui()

if __name__ == "__main__":
    main()
