import os
import psycopg2
import google.generativeai as genai
import numpy as np
from typing import List, Tuple
import tiktoken
import json
import logging

class RAGSystem:
    def __init__(self, gemini_api_key: str, pg_connection_params: dict, documentation_folder: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        logging.basicConfig(level=logging.ERROR)
        genai.configure(api_key=gemini_api_key)
        self.conn = psycopg2.connect(**pg_connection_params)
        self.cursor = self.conn.cursor()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._setup_pgvector()
        self.documentation_folder = documentation_folder

    def _setup_pgvector(self):
        self.cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id SERIAL PRIMARY KEY,
                document_path TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
        ''')
        self.conn.commit()

    def chunk_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk)
        return chunks

    def extract_text_from_txts(self) -> List[Tuple[str, List[str]]]:
        documents = []
        for filename in os.listdir(self.documentation_folder):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.documentation_folder, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()
                    chunks = self.chunk_text(text)
                    documents.append((filepath, chunks))
        return documents

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            if len(text) > 10000:
                text = text[:10000]
            result = genai.embed_content(model="models/text-embedding-004", content=text)
            embeddings.append(result['embedding'])
        return embeddings

    def store_document_embeddings(self):
        documents = self.extract_text_from_txts()
        all_chunks = []
        chunk_filepaths = []
        for filepath, chunks in documents:
            all_chunks.extend(chunks)
            chunk_filepaths.extend([filepath] * len(chunks))
        
        embeddings = self.generate_embeddings(all_chunks)

        for (filepath, text), embedding in zip(zip(chunk_filepaths, all_chunks), embeddings):
            embedding_str = json.dumps(embedding)
            self.cursor.execute(
                'INSERT INTO document_embeddings (document_path, content, embedding) VALUES (%s, %s, %s)',
                (filepath, text, embedding_str)
            )
        self.conn.commit()

    def retrieve_relevant_documents(self, query: str, selected_file: str, top_k: int = 3) -> List[Tuple[str, str]]:
        query_embedding = self.generate_embeddings([query])[0]
        self.cursor.execute('SELECT document_path, content, embedding FROM document_embeddings')
        results = self.cursor.fetchall()
        
        similarities = []
        for doc_path, content, embedding_str in results:
            if os.path.basename(doc_path) != selected_file:
                continue  

            stored_embedding = json.loads(embedding_str)
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            similarities.append((similarity, content, doc_path))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [(doc_path, doc) for _, doc, doc_path in similarities[:top_k]]

    def answer_query(self, query: str, selected_file: str) -> str:
        relevant_docs = self.retrieve_relevant_documents(query, selected_file)
        
        if not relevant_docs:
            return "No relevant documents found in the selected file."

        context = "\n\n".join(doc for _, doc in relevant_docs)
        
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Context:\n{context}\n\nQuery: {query}\n\nProvide a comprehensive answer based on the context:"
        
        try:
            response = model.generate_content(prompt)
            if hasattr(response, 'text') and response.text:
                citations = "\nCitations:\n" + "\n".join(doc_path for doc_path, _ in relevant_docs)
                return f"{response.text}\n\n{citations}"
            else:
                return "No valid response generated."
        except Exception as e:
            return f"An error occurred while generating the response: {str(e)}"

    def close_connection(self):
        self.cursor.close()
        self.conn.close()