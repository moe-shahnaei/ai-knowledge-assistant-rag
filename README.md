# AI Knowledge Assistant with RAG

## Project Overview
This project is a simple Retrieval-Augmented Generation (RAG) knowledge assistant. It processes local text documents, splits them into cleaner chunks, creates embeddings, stores them in a Chroma vector database, retrieves relevant context based on a user question, and generates a grounded answer with source references.

## Why I Built This
I built this project to strengthen my practical understanding of AI application support, semantic search, document retrieval, embeddings, vector databases, and grounded assistant responses. The project is especially relevant to AI support, junior AI engineering, application support, and data/knowledge management roles.

## Current Features
- Loads `.txt` documents from a local knowledge base
- Splits documents into sentence-aware chunks
- Saves processed chunks locally
- Creates embeddings using `sentence-transformers`
- Stores embeddings in a Chroma vector database
- Retrieves semantically relevant chunks for a user question
- Generates a basic grounded answer using retrieved context
- Displays source files and chunk IDs for transparency

## RAG Pipeline
Documents → Chunks → Embeddings → Vector Database → Semantic Retrieval → Grounded Answer with Sources

## Tools and Technologies
- Python
- ChromaDB
- SentenceTransformers
- all-MiniLM-L6-v2 embedding model
- JSONL
- Git / GitHub
- VS Code

## Sample Question
**Question:** What are common data quality issues?

**Answer:** The assistant retrieves context from the data quality notes and identifies issues such as missing values, duplicate records, invalid dates, incorrect statuses, inconsistent formatting, outdated records, and missing documents.

**Sources:**
- data_quality_notes.txt | chunk 1
- data_quality_notes.txt | chunk 2

## What I Improved on Day 2
- Added `app.py` to run an assistant-style Q&A flow
- Updated `retrieve.py` so retrieval results are returned in a structured format
- Improved `ingest.py` by replacing character-based chunking with sentence-aware chunking
- Added source display to make answers traceable
- Tested the full question → retrieval → answer → sources pipeline

## Next Improvements
- Remove duplicate sentences caused by overlapping chunks
- Add a Streamlit interface
- Add more sample documents
- Add answer quality evaluation
- Add a troubleshooting/runbook section
