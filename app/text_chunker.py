class TextChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text):
        """Split text into overlapping chunks of specified size."""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            # If we're not at the beginning, we include some overlap
            if start > 0:
                start = max(0, start - self.chunk_overlap)
            
            chunk = text[start:end]
            chunks.append({
                "content": chunk,
                "metadata": {
                    "start_idx": start,
                    "end_idx": end
                }
            })
            
            start = end
        
        return chunks
    
    def chunk_json(self, json_data, path_prefix=""):
        """Recursively chunk JSON data by fields/objects."""
        chunks = []
        
        if isinstance(json_data, dict):
            # Process dictionary items
            for key, value in json_data.items():
                current_path = f"{path_prefix}.{key}" if path_prefix else key
                
                if isinstance(value, (dict, list)):
                    # Recursively chunk nested structures
                    chunks.extend(self.chunk_json(value, current_path))
                else:
                    # Leaf node with primitive value
                    chunks.append({
                        "content": str(value),
                        "metadata": {
                            "json_path": current_path,
                            "type": type(value).__name__
                        }
                    })
        
        elif isinstance(json_data, list):
            # Process list items
            for idx, item in enumerate(json_data):
                current_path = f"{path_prefix}[{idx}]"
                if isinstance(item, (dict, list)):
                    chunks.extend(self.chunk_json(item, current_path))
                else:
                    chunks.append({
                        "content": str(item),
                        "metadata": {
                            "json_path": current_path,
                            "type": type(item).__name__,
                            "array_index": idx
                        }
                    })
        
        return chunks