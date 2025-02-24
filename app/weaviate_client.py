import weaviate
from weaviate.util import generate_uuid5
import uuid
import datetime

class WeaviateClient:
    def __init__(self, url, api_key=None):
        self.client = weaviate.Client(
            url=url,
            auth_client_secret=weaviate.auth.AuthApiKey(api_key) if api_key else None
        )
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create the schema if it doesn't exist"""
        # Document class for storing whole documents
        if not self.client.schema.exists("Document"):
            document_class = {
                "class": "Document",
                "description": "A document uploaded to the RAG system",
                "properties": [
                    {
                        "name": "title",
                        "dataType": ["string"],
                        "description": "The document title or filename"
                    },
                    {
                        "name": "fileType",
                        "dataType": ["string"],
                        "description": "The file format (pdf, docx, json, txt)"
                    },
                    {
                        "name": "uploadedAt",
                        "dataType": ["date"],
                        "description": "When the document was uploaded"
                    }
                ]
            }
            self.client.schema.create_class(document_class)
        
        # Chunk class for storing document chunks with embeddings
        if not self.client.schema.exists("Chunk"):
            chunk_class = {
                "class": "Chunk",
                "description": "A chunk of text from a document with embedding",
                "vectorizer": "none",  # We'll provide our own vectors
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "The chunk text content"
                    },
                    {
                        "name": "documentId",
                        "dataType": ["string"],
                        "description": "Reference to parent document"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["object"],
                        "description": "Additional metadata about the chunk"
                    }
                ]
            }
            self.client.schema.create_class(chunk_class)
    
    def store_document(self, title, file_type, metadata=None):
        """Store document info and return the generated document ID"""
        doc_id = str(uuid.uuid4())
        
        doc_object = {
            "title": title,
            "fileType": file_type,
            "uploadedAt": datetime.datetime.now().isoformat(),
        }
        
        if metadata:
            doc_object.update(metadata)
        
        # Store document metadata
        self.client.data_object.create(
            doc_object,
            "Document",
            doc_id
        )
        
        return doc_id
    
    def store_chunks(self, chunks, document_id):
        """Store document chunks with their embeddings"""
        batch = self.client.batch.configure(batch_size=100)
        
        with batch:
            for chunk in chunks:
                # Create UUID based on content to avoid duplicates
                chunk_id = generate_uuid5(chunk["content"])
                
                # Prepare the data object with vector
                properties = {
                    "content": chunk["content"],
                    "documentId": document_id,
                    "metadata": chunk.get("metadata", {})
                }
                
                # Store with custom vector
                self.client.batch.add_data_object(
                    data_object=properties,
                    class_name="Chunk",
                    uuid=chunk_id,
                    vector=chunk["embedding"]
                )
    
    def delete_document(self, document_id):
        """Delete a document and all its chunks"""
        # First delete all chunks belonging to this document
        where_filter = {
            "path": ["documentId"],
            "operator": "Equal",
            "valueString": document_id
        }
        
        # Find all chunks for this document
        result = self.client.query.get(
            "Chunk", ["id"]
        ).with_where(where_filter).do()
        
        # Delete each chunk
        chunk_ids = [item["id"] for item in result["data"]["Get"]["Chunk"]]
        for chunk_id in chunk_ids:
            self.client.data_object.delete(
                uuid=chunk_id,
                class_name="Chunk"
            )
        
        # Then delete the document itself
        self.client.data_object.delete(
            uuid=document_id,
            class_name="Document"
        )
    
    def semantic_search(self, query_embedding, document_id=None, limit=5):
        """Search for similar chunks in Weaviate"""
        # Optional filter for specific document
        where_filter = None
        if document_id:
            where_filter = {
                "path": ["documentId"],
                "operator": "Equal",
                "valueString": document_id
            }
        
        # Perform the vector search
        result = self.client.query.get(
            "Chunk", ["content", "documentId", "metadata"]
        ).with_near_vector(
            {"vector": query_embedding}
        ).with_where(
            where_filter
        ).with_limit(limit).do()
        
        return result["data"]["Get"]["Chunk"]