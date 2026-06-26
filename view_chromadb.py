#!/usr/bin/env python3
"""
Script to view ChromaDB contents and statistics.
Run: python view_chromadb.py
"""

import chromadb
from tabulate import tabulate
import json

def view_chromadb():
    """Display all documents in ChromaDB"""
    
    print("=" * 80)
    print("ChromaDB Viewer - Investor Intelligence Platform")
    print("=" * 80)
    
    try:
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path="./chroma_data")
        
        # List all collections
        collections = client.list_collections()
        print(f"\nTotal Collections: {len(collections)}")
        
        if not collections:
            print("❌ No collections found in ChromaDB")
            return
        
        # Process each collection
        for col in collections:
            collection = client.get_collection(name=col.name)
            count = collection.count()
            
            print(f"\n{'─' * 80}")
            print(f"📦 Collection: {col.name}")
            print(f"📊 Total Documents: {count}")
            print(f"{'─' * 80}")
            
            if count == 0:
                print("   (empty)")
                continue
            
            # Get all documents
            results = collection.get(
                include=["documents", "metadatas"]
            )
            
            # Prepare table data
            table_data = []
            for i, (doc_id, metadata, document) in enumerate(zip(
                results['ids'],
                results['metadatas'],
                results['documents']
            ), 1):
                company = metadata.get('company', 'N/A')
                year = metadata.get('year', 'N/A')
                source = metadata.get('source_file', 'N/A')
                doc_preview = document[:60] + "..." if len(document) > 60 else document
                
                table_data.append([
                    i,
                    company,
                    year,
                    source,
                    doc_preview
                ])
            
            # Display table
            headers = ["#", "Company", "Year", "Source File", "Content Preview"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            
            # Display metadata summary
            print(f"\n📋 Metadata Summary:")
            companies = set(m.get('company') for m in results['metadatas'] if m.get('company'))
            years = set(m.get('year') for m in results['metadatas'] if m.get('year'))
            sources = set(m.get('source_file') for m in results['metadatas'] if m.get('source_file'))
            
            print(f"   Companies: {', '.join(sorted(companies)) if companies else 'N/A'}")
            print(f"   Years: {', '.join(sorted(map(str, years))) if years else 'N/A'}")
            print(f"   Source Files: {len(sources)} unique file(s)")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure ChromaDB data exists at ./chroma_data")

if __name__ == "__main__":
    view_chromadb()
