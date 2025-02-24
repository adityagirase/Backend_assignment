from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import os
import tempfile
import uvicorn
from pydantic import BaseModel
import datetime
import json

# Import our custom modules
from document_processor import DocumentProcessor
from text_chunker import TextChunker
from embedding_generator import EmbeddingGenerator
from weaviate_client import WeaviateClient

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Initialize services
app = FastAPI(title="RAG System API", description="Document Retrieval with Weaviate")
doc_processor = DocumentProcessor()
text_chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
embedding_generator = EmbeddingGenerator(api_key=OPENAI_API_KEY)
weaviate_client = WeaviateClient(url=WEAVIATE_URL, api_key=WEAVIATE_API_KEY)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class DocumentResponse(BaseModel):
    document_id: str
    title: str
    file_type: str
    upload_time: str
    metadata: Dict[str, Any]

class QueryRequest(BaseModel):
    query: str
    document_id: Optional[str] = None
    limit: int = 5

class QueryResult(BaseModel):
    content: str
    document_id: str
    metadata: Dict[str, Any]
    relevance_score: float

class QueryResponse(BaseModel):
    results: List[QueryResult]
    query: str

# Dependency for stats
async def get_stats():
    # Get total document and chunk counts
    docs_result = weaviate_client.client.query.aggregate("Document").with_meta_count().do()
    chunks_result = weaviate_client.client.query.aggregate("Chunk").with_meta_count().do()
    
    doc_count = docs_result["data"]["Aggregate"]["Document"][0]["meta"]["count"]
    chunk_count = chunks_result["data"]["Aggregate"]["Chunk"][0]["meta"]["count"]
    
    return {
        "document_count": doc_count,
        "chunk_count": chunk_count,
        "last_updated": datetime.datetime.now().isoformat()
    }

@app.get("/")
async def root(stats: Dict = Depends(get_stats)):
    """Root endpoint with system stats"""
    return {
        "status": "ok",
        "name": "RAG System with Weaviate",
        "stats": stats
    }

@app.post("/documents", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF, DOCX, JSON, TXT)
    """
    # Validate file type
    filename = file.filename
    file_extension = os.path.splitext(filename)[1].lower()
    
    if file_extension not in ['.pdf', '.docx', '.json', '.txt']:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")
    
    # Process the document
    try:
        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Process based on file type
        with open(temp_path, 'rb') as file_obj:
            document_data = doc_processor.process_document(file_obj, filename)
        
        # Store document in Weaviate and get ID
        document_id = weaviate_client.store_document(
            title=filename,
            file_type=file_extension[1:],  # Remove the dot
            metadata=document_data.get("metadata", {})
        )
        
        # Chunk the document content
        if file_extension == '.json' and 'raw_json' in document_data:
            # Special handling for JSON structure
            chunks = text_chunker.chunk_json(document_data['raw_json'])
        else:
            # Text-based chunking for other formats
            chunks = text_chunker.chunk_text(document_data['content'])
        
        # Generate embeddings for chunks
        chunks_with_embeddings = embedding_generator.batch_generate_embeddings(chunks)
        
        # Store chunks in Weaviate
        weaviate_client.store_chunks(chunks_with_embeddings, document_id)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return {
            "document_id": document_id,
            "title": filename,
            "file_type": file_extension[1:],
            "upload_time": datetime.datetime.now().isoformat(),
            "metadata": document_data.get("metadata", {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing error: {str(e)}")

@app.put("/documents/{document_id}", response_model=DocumentResponse)
async def update_document(document_id: str, file: UploadFile = File(...)):
    """
    Update a document by replacing it entirely
    """
    # Verify document exists
    try:
        doc_result = weaviate_client.client.data_object.get(
            document_id,
            "Document"
        )
        if not doc_result:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
    except Exception:
        raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
    
    # Delete existing document and its chunks
    weaviate_client.delete_document(document_id)
    
    # Process new document
    filename = file.filename
    file_extension = os.path.splitext(filename)[1].lower()
    
    if file_extension not in ['.pdf', '.docx', '.json', '.txt']:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")
    
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Process the new document
        with open(temp_path, 'rb') as file_obj:
            document_data = doc_processor.process_document(file_obj, filename)
        
        # Re-use the same document ID
        weaviate_client.store_document(
            title=filename,
            file_type=file_extension[1:],
            metadata=document_data.get("metadata", {}),
            document_id=document_id
        )
        
        # Chunk and embed the new content
        if file_extension == '.json' and 'raw_json' in document_data:
            chunks = text_chunker.chunk_json(document_data['raw_json'])
        else:
            chunks = text_chunker.chunk_text(document_data['content'])
        
        chunks_with_embeddings = embedding_generator.batch_generate_embeddings(chunks)
        
        # Store updated chunks
        weaviate_client.store_chunks(chunks_with_embeddings, document_id)
        
        # Clean up
        os.unlink(temp_path)
        
        return {
            "document_id": document_id,
            "title": filename,
            "file_type": file_extension[1:],
            "upload_time": datetime.datetime.now().isoformat(),
            "metadata": document_data.get("metadata", {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document update error: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its chunks
    """
    try:
        weaviate_client.delete_document(document_id)
        return {"status": "success", "message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.get("/documents")
async def list_documents():
    """
    List all documents in the system
    """
    try:
        result = weaviate_client.client.query.get(
            "Document", ["title", "fileType", "uploadedAt"]
        ).do()
        
        documents = result["data"]["Get"]["Document"]
        return {"documents": documents, "count": len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query for relevant document chunks
    """
    try:
        # Generate embedding for the query
        query_embedding = embedding_generator.generate_embedding(request.query)
        
        # Search in Weaviate
        results = weaviate_client.semantic_search(
            query_embedding,
            document_id=request.document_id,
            limit=request.limit
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result["content"],
                "document_id": result["documentId"],
                "metadata": result.get("metadata", {}),
                "relevance_score": result.get("_additional", {}).get("distance", 0)
            })
        
        return {
            "results": formatted_results,
            "query": request.query
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

# JSON Bonus Functionality
@app.post("/query/json")
async def query_json_document(
    document_id: str,
    operation: str = Query(..., description="Operation type: max, min, sum, avg"),
    field_path: str = Query(..., description="JSON path to field for operation"),
):
    """
    Perform operations on JSON document fields (Bonus functionality)
    """
    # First verify this is a JSON document
    doc_info = weaviate_client.client.data_object.get(document_id, "Document")
    if not doc_info or doc_info.get("fileType") != "json":
        raise HTTPException(
            status_code=400, 
            detail="Document not found or not a JSON document"
        )
    
    # Get all chunks for this document
    where_filter = {
        "path": ["documentId"],
        "operator": "Equal",
        "valueString": document_id
    }
    
    json_path_filter = {
        "path": ["metadata", "json_path"],
        "operator": "Like",
        "valueString": f"*{field_path}*"
    }
    
    # Combine filters
    combined_filter = {
        "operator": "And",
        "operands": [where_filter, json_path_filter]
    }
    
    # Query for the relevant chunks
    result = weaviate_client.client.query.get(
        "Chunk", ["content", "metadata"]
    ).with_where(combined_filter).do()
    
    chunks = result["data"]["Get"]["Chunk"]
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for field path: {field_path}"
        )
    
    # Extract values from chunks
    values = []
    for chunk in chunks:
        try:
            # For numerical operations, convert to float
            values.append(float(chunk["content"]))
        except ValueError:
            continue  # Skip non-numeric values
    
    if not values:
        raise HTTPException(
            status_code=400,
            detail=f"No numeric values found for field path: {field_path}"
        )
    
    # Perform the requested operation
    result = None
    if operation == "max":
        result = max(values)
    elif operation == "min":
        result = min(values)
    elif operation == "sum":
        result = sum(values)
    elif operation == "avg":
        result = sum(values) / len(values)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported operation: {operation}"
        )
    
    return {
        "document_id": document_id,
        "field_path": field_path,
        "operation": operation,
        "result": result,
        "count": len(values)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)