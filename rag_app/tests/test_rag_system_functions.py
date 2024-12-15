import os
import pytest
import json
from unittest.mock import patch, MagicMock
import numpy as np

from rag_app.rag_system import RAGSystem

def test_rag_system_initialization(setup_test_documents):
    with patch('psycopg2.connect') as mock_connect, \
         patch('google.generativeai.configure') as mock_genai_config:
        
        rag_system = RAGSystem(
            'test_key', 
            {
                'dbname': os.getenv('DATABASE_NAME'),
                'user': os.getenv('DATABASE_USER'),
                'password': os.getenv('DATABASE_PASSWORD'),
                'host': os.getenv('DATABASE_HOST'),
                'port': os.getenv('DATABASE_PORT')
            }, 
            setup_test_documents
        )
        
        mock_genai_config.assert_called_once_with(api_key='test_key')
        mock_connect.assert_called_once()

def test_chunk_text():
    rag_system = RAGSystem(
        'test_key', 
        {
            'dbname': os.getenv('DATABASE_NAME'),
            'user': os.getenv('DATABASE_USER'),
            'password': os.getenv('DATABASE_PASSWORD'),
            'host': os.getenv('DATABASE_HOST'),
            'port': os.getenv('DATABASE_PORT')
        }, 
        './documents'
    )
    
    text = "This is a long text that needs to be chunked into smaller pieces for better processing and embedding."
    chunks = rag_system.chunk_text(text)
    
    assert len(chunks) > 0
    assert all(len(chunk) > 0 for chunk in chunks)

@patch('google.generativeai.embed_content')
def test_generate_embeddings(mock_embed_content):
    
    mock_embed_content.return_value = {
        'embedding': [0.1, 0.2, 0.3] * 10  
    }
    
    rag_system = RAGSystem(
        'test_key', 
        {
            'dbname': os.getenv('DATABASE_NAME'),
            'user': os.getenv('DATABASE_USER'),
            'password': os.getenv('DATABASE_PASSWORD'),
            'host': os.getenv('DATABASE_HOST'),
            'port': os.getenv('DATABASE_PORT')
        }, 
        './documents'
    )
    
    texts = ["Test text 1", "Test text 2"]
    embeddings = rag_system.generate_embeddings(texts)
    
    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) == 30  

def test_extract_text_from_txts(setup_test_documents):
    rag_system = RAGSystem(
        'test_key', 
        {
            'dbname': os.getenv('DATABASE_NAME'),
            'user': os.getenv('DATABASE_USER'),
            'password': os.getenv('DATABASE_PASSWORD'),
            'host': os.getenv('DATABASE_HOST'),
            'port': os.getenv('DATABASE_PORT')
        }, 
        setup_test_documents
    )
    
    documents = rag_system.extract_text_from_txts()
    
    assert len(documents) == 1
    assert isinstance(documents[0][1], list)  

@patch('google.generativeai.GenerativeModel')
def test_answer_query(mock_genai_model):
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value.text = "Test answer"
    mock_genai_model.return_value = mock_model_instance
    
    rag_system = RAGSystem(
        'test_key', 
        {
            'dbname': os.getenv('DATABASE_NAME'),
            'user': os.getenv('DATABASE_USER'),
            'password': os.getenv('DATABASE_PASSWORD'),
            'host': os.getenv('DATABASE_HOST'),
            'port': os.getenv('DATABASE_PORT')
        }, 
        './documents'
    )
    
    answer = rag_system.answer_query("Test query")
    
    assert "Test answer" in answer