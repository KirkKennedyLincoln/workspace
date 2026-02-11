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
            sqlite = any('.sqlite3' in f for f in current_dir)
            if sqlite:
                client = chromadb.PersistentClient(path=path)
                collections = client.list_collections()
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

def deduplicate_results(documents: List[str], metadatas: List[Dict]) -> tuple:
    # claude ai helped with this helper function 02/11/2026, my original approach was using dict key strings to dedupe.
    seen = []
    deduped_docs = []
    deduped_meta = []

    for text, meta in zip(documents, metadatas):
        source = metadata.get("source", "")
        chunk_idx = metadata.get("chunk_index", -1)

        is_adjacent = any(
            s == source and abs(c - chunk_index) <= 1
            for s, c in seen
        )

        # could add in ranged 80-90% of the text being mimicked help from claude
        # is_correlated = any(
        #     len(set(text.split())) == len(set(d.split())) / max(len(set(text.split())), 1) > 0.9
        #     for d in deduped_docs
        # )

        if is_adjacent:
            continue

        seen.append((source, chunk_idx))
        deduped_docs.append(text)
        deduped_meta.append(meta)

    return (deduped_docs, deduped_meta)

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    deduped_docs, deduped_meta = deduplicate_results(documents, metadatas)
    # TODO: Initialize list with header text for context section
    context = ["-------EXTRACTED CONTEXT-------"]
    for text, meta in zip(deduped_documents, deduped_metadatas):
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