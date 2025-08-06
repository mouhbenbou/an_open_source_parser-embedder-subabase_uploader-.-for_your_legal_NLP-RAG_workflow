# Legal Document Chunk Uploader

A Streamlit application for parsing legal documents into structured chunks, generating embeddings using LegalBERT, and uploading them to Supabase for RAG applications.

## Features

- **Document parsing**: Extracts sections and articles from legal texts using regex patterns
- **Embedding generation**: Uses LegalBERT (`nlpaueb/legal-bert-base-uncased`) to create 768D embeddings
- **Duplicate detection**: SHA-256 hashing prevents duplicate chunk uploads
- **Batch processing**: Uploads chunks in batches of 50 with parallel embedding generation
- **Metadata validation**: Enforces strict validation of jurisdiction codes, language codes, and URLs
- **Supabase integration**: Stores chunks with pgvector embeddings for semantic search

## Dependencies

### Python Packages
```
streamlit==1.32.0
numpy==1.26.4
torch==2.2.1
transformers==4.39.3
supabase==2.4.0
pgvector==0.2.6
tenacity==8.2.3
python-dotenv==1.0.1
```

### Environment Variables
```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_role_key
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/legal-document-uploader.git
cd legal-document-uploader
```



3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. you don't have to worry about envirmental variables since when you run the streamlit app on your terminal and provide it with both the url and anon key of your supabase database it will handle everything on its own .

## Usage

1. Start the Streamlit application:
```bash
streamlit run streamlit_app.py
```

2. Configure Supabase credentials in the sidebar:
   - Enter Supabase project URL
   - Provide service role key
   - Specify target table name (default: `legal_chunks`)
   - you specify other advanced parameters to suit your uploading goals such as the option to enable "skip existing .." and the name of the specific embedding model you want to use

3. Upload a legal text file (TXT or MD format) with the following structure:
```
Section 1 : General Provisions
Article 1 : This law governs...
Article 2 : Definitions include...
```

4. Provide document metadata:
   - Document type (civil code, criminal code, etc.)
   - Jurisdiction code (2-3 letter code like "us" or "uk")
   - Language code (2-letter ISO code like "en" or "fr")
   - Source URL
   - Publication date

5. Parse document and preview chunks

6. Upload chunks to Supabase

## Configuration

### Supabase Table Schema
The target table must have these columns:
```sql
create table public.legal_chunks (
  id uuid not null default gen_random_uuid (),
  document_type text not null,
  jurisdiction text not null,
  language text not null,
  section_title text not null,
  section_title_embedding vector(768) not null,
  article_number text not null,
  title text not null,
  body text not null,
  tags text[] null,
  article_embedding vector(768) not null,
  source_url text not null,
  date_published date not null,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  is_active boolean not null default true,
  constraint legal_chunks_pkey primary key (id)
) TABLESPACE pg_default;

create index IF not exists legal_chunks_embedding_idx on public.legal_chunks using hnsw (article_embedding vector_cosine_ops) with  (m = '16', ef_construction = '64') TABLESPACE pg_default;
create index IF not exists section_title_embedding_idx on public.legal_chunks using hnsw (section_title_embedding vector_cosine_ops) with  (m = '16', ef_construction = '64') TABLESPACE pg_default;
```

### Advanced Options (Sidebar)
- **Batch Size**: Number of chunks per upload batch (10-200)
- **Max Workers**: Concurrent embedding threads (1-10)
- **Skip Existing**: Toggle duplicate chunk skipping
- **Model Name**: Hugging Face model identifier

## API Reference

### `LegalDocumentParser`
```python
def parse_legal_file(text: str, **metadata) -> Generator[LegalChunk, None, None]
```
Parses legal text into structured chunks with metadata validation

**Parameters:**
- `text`: Raw document content
- `metadata`: Document metadata (jurisdiction, language, etc.)

**Returns:** Generator of `LegalChunk` objects

### `EmbeddingGenerator`
```python
def get_embedding(text: str) -> List[float]
```
Generates normalized LegalBERT embeddings

**Parameters:**
- `text`: Input text (max 256 tokens)

**Returns:** 768-dimensional embedding vector

### `SupabaseUploader`
```python
def upload_chunks_batch(
    chunks: List[LegalChunk],
    embedding_generator: EmbeddingGenerator,
    progress_callback=None,
    skip_existing: bool = True
) -> Dict[str, Any]
```
Uploads chunks to Supabase with embedding generation

**Parameters:**
- `chunks`: List of parsed chunks
- `embedding_generator`: Embedding generator instance
- `progress_callback`: Function for progress updates
- `skip_existing`: Skip duplicate chunks

**Returns:** Upload statistics dictionary

## Troubleshooting

| Error | Solution |
|-------|----------|
| `Invalid jurisdiction code` | Use 2-3 lowercase letters (e.g., "us", "uk") |
| `Invalid language code` | Use 2-letter ISO code (e.g., "en", "fr") |
| `Source URL validation failed` | URL must start with http:// or https:// |
| `No section markers found` | Ensure document follows "Section X: ..." format |
| `Duplicate key violation` | Enable "Skip Existing Chunks" in advanced settings |
| `Model loading failed` | Check Hugging Face model ID and internet connection |
| `Invalid embedding dimension` | Verify model outputs 768-dimensional vectors |
| `Null constraint violation` | Ensure all required metadata fields are provided |
