__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path
import os


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    directories = os.listdir(".")
    
    for name in directories:
        path = f"./{name}"

        if not os.path.isdir(path):
            continue

        try:
            current_dir = os.listdir(path)
            # help from Claude to understand globbing the sqlite files 02/04/2026
            print(current_dir)
            sqlite = any('.sqlite3' in f for f in current_dir)
            if sqlite:
                client = chromadb.PersistentClient(path=path)
                collections = client.list_collections()
                print(f"list_collections returned: {collections}")
                for collection in collections:
                    backends[f"{path}:{collection.name}"] = {
                        "path": str(path),
                        "collection": collection,
                        "collection_name": collection.name,
                        "directory": str(path),
                        "display_name": f"{collection.name} ({path})",
                        "doc_count": collection.count()
                    }
        except Exception as e:
            print("Excpetion", e)
            continue

    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(name=collection_name)
    return collection, True, None

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    filtering = None
    if mission_filter and mission_filter.lower() != "all":
        filtering = {"mission": mission_filter}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=filtering
    )

    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    # TODO: Initialize list with header text for context section
    context = ["-------EXTRACTED CONTEXT-------"]
    for text, meta in zip(documents, metadatas):
        mission = ""
        try:
            mission = meta['mission'] 
        except KeyError:
            mission = 'Unknown'

        mission = mission.replace("_", " ")
        mission = mission.capitalize()

        source = ""
        try:
            source = meta['source'] 
        except KeyError:
            source = 'Unknown'

        category = ""
        try:
            category = meta['category'] 
        except KeyError:
            category = 'Unknown'

        category = category.replace("_", " ")
        category = category.capitalize()

        header = f"{meta['chunk_index']}:{mission}:{source}:{category}"

        context.append(header)

        max_length = 1000
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        context.append(text)
        context.append("")

    return "\n".join(context)