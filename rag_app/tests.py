import os
import time
import unittest
import statistics
from unittest.mock import patch
from dotenv import load_dotenv
from .rag_system import RAGSystem

load_dotenv()

class TestRAGSystem(unittest.TestCase):
    
    @patch('psycopg2.connect')
    @patch('google.generativeai.embed_content')
    def setUp(self, mock_embed_content, mock_connect):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.pg_connection_params = {
            'dbname': os.getenv('DATABASE_NAME'),
            'user': os.getenv('DATABASE_USER'),
            'password': os.getenv('DATABASE_PASSWORD'),
            'host': os.getenv('DATABASE_HOST'),
            'port': os.getenv('DATABASE_PORT')
        }
        self.documentation_folder = './documents'
        
        self.rag_system = RAGSystem(self.gemini_api_key, self.pg_connection_params, self.documentation_folder)

    def test_chunk_text(self):
        text = "This is a simple text to be chunked."
        expected_chunks = ["This is a simple text to be chunked."]
        chunks = self.rag_system.chunk_text(text)
        self.assertEqual(chunks, expected_chunks)

    def test_chunk_empty_text(self):
        text = ""
        expected_chunks = []
        chunks = self.rag_system.chunk_text(text)
        self.assertEqual(chunks, expected_chunks)

    def test_chunk_invalid_input(self):
        with self.assertRaises(TypeError):
            self.rag_system.chunk_text(123)

    @patch('google.generativeai.embed_content', side_effect=[{'embedding': [0.1]}, {'embedding': [0.2]}])
    def test_generate_embeddings_multiple_texts(self, mock_embed_content):
        texts = ["Text one.", "Text two."]
        embeddings = self.rag_system.generate_embeddings(texts)
        expected_embeddings = [[0.1], [0.2]]
        self.assertEqual(embeddings, expected_embeddings)

    @patch('google.generativeai.embed_content', return_value={'embedding': [0.1]})
    def test_generate_embeddings(self, mock_embed_content):
        texts = ["This is a sample text."]
        embeddings = self.rag_system.generate_embeddings(texts)
        expected_embeddings = [[0.1]]
        self.assertEqual(embeddings, expected_embeddings)

    def test_generate_embeddings_empty_list(self):
        texts = []
        embeddings = self.rag_system.generate_embeddings(texts)
        expected_embeddings = []
        self.assertEqual(embeddings, expected_embeddings)

    @patch('google.generativeai.embed_content', return_value={'embedding': [0.1]})
    def test_performance_generate_embeddings(self, mock_embed_content):
        texts = ["Sample text"] * 1000
        
        timing_runs = []
        for _ in range(5):
            start_time = time.time()
            self.rag_system.generate_embeddings(texts)
            duration = time.time() - start_time
            timing_runs.append(duration)
        
        avg_time = statistics.mean(timing_runs)
        std_dev = statistics.stdev(timing_runs)
        
        print(f"\nEmbedding Generation Performance:")
        print(f"Average Time: {avg_time:.4f} seconds")
        print(f"Standard Deviation: {std_dev:.4f} seconds")
        print(f"Timing Runs: {timing_runs}")
        
        self.assertLess(avg_time, 2.0, "Average embedding generation time should be under 2 seconds")
        self.assertLess(std_dev, 0.5, "Performance should be consistent across runs")

    def test_timing_chunk_text(self):
        test_cases = [
            "Short text",
            "Medium length text with multiple words and some complexity",
            "A" * 1000,
        ]
        
        timing_results = []
        for text in test_cases:
            start_time = time.time()
            self.rag_system.chunk_text(text)
            duration = time.time() - start_time
            timing_results.append(duration)
        
        print("\nText Chunking Performance:")
        for i, (text, duration) in enumerate(zip(test_cases, timing_results), 1):
            print(f"Text {i} (Length: {len(text)}): {duration:.6f} seconds")
        
        self.assertTrue(all(t < 0.01 for t in timing_results), "Chunking should be very fast")

if __name__ == '__main__':
    unittest.main()