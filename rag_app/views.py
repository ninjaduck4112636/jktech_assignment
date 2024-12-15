from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_http_methods
from django.shortcuts import render
from django.http import JsonResponse
import os
from .rag_system import RAGSystem
from dotenv import load_dotenv

load_dotenv()

def index(request):
    DOCUMENTATION_FOLDER = './documents'
    files = [f for f in os.listdir(DOCUMENTATION_FOLDER) if f.endswith('.txt')]
    return render(request, 'index.html', {'files': files})

@csrf_protect
@require_http_methods(["POST"])
def query_documents(request):
    query = request.POST.get('query', '')
    selected_file = request.POST.get('file', '')
    
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    PG_CONFIG = {
        'dbname': os.getenv('DATABASE_NAME'),
        'user': os.getenv('DATABASE_USER'),
        'password': os.getenv('DATABASE_PASSWORD'),
        'host': os.getenv('DATABASE_HOST'),
        'port': os.getenv('DATABASE_PORT')
    }
    DOCUMENTATION_FOLDER = './documents'
    
    rag_system = RAGSystem(GEMINI_API_KEY, PG_CONFIG, DOCUMENTATION_FOLDER)
    
    try:
        relevant_docs = rag_system.retrieve_relevant_documents(query, selected_file)
        
        if not relevant_docs:
            rag_system.store_document_embeddings()
        
        answer = rag_system.answer_query(query, selected_file)
        
        return JsonResponse({
            'answer': answer,
            'selected_file': selected_file
        })
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)
    finally:
        rag_system.close_connection()
