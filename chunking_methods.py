import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
import numpy as np

class ChunkingMethods:
    def __init__(self):
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def fixed_size_chunking(self, text, chunk_size=500, chunk_overlap=50):
        """Fixed size chunking implementation"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        return splitter.split_text(text)
    
    def recursive_chunking(self, text, chunk_size=1000, separators=["\n\n", "\n", " ", ""]):
        """Recursive chunking based on separators"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=separators
        )
        return splitter.split_text(text)
    
    def semantic_chunking(self, text, threshold=0.7, min_size=100):
        """Semantic chunking based on embedding similarity"""
        # First split into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        sentences = [s for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return [text]
        
        # Get embeddings for all sentences
        embeddings = self.embedding_model.encode(sentences)
        
        chunks = []
        current_chunk = sentences[0]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            similarity = util.cos_sim(current_embedding, embeddings[i]).item()
            
            if similarity > threshold and len(current_chunk) + len(sentences[i]) < 1000:
                # Merge with current chunk
                current_chunk += " " + sentences[i]
                # Update embedding with weighted average
                current_embedding = (current_embedding * len(current_chunk.split()) + 
                                   embeddings[i] * len(sentences[i].split())) / \
                                  (len(current_chunk.split()) + len(sentences[i].split()))
            else:
                # Start new chunk
                chunks.append(current_chunk)
                current_chunk = sentences[i]
                current_embedding = embeddings[i]
        
        chunks.append(current_chunk)
        return chunks