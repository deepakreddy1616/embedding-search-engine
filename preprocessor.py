import hashlib
import re
import os
from bs4 import BeautifulSoup
from pathlib import Path

class DocumentPreprocessor:
    def clean_text(self, text):
        """Clean and normalize text"""
        text = BeautifulSoup(text, "lxml").get_text()
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def compute_hash(self, text):
        """Compute SHA-256 hash for cache validation"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def process_file(self, filepath):
        """Process a single document file"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()
        cleaned = self.clean_text(raw)
        meta = {
            "filename": os.path.basename(filepath),
            "hash": self.compute_hash(cleaned),
            "length": len(cleaned)
        }
        return {"text": cleaned, "metadata": meta}

    def process_dir(self, directory):
        """Process all .txt files in a directory"""
        docs = []
        for file in Path(directory).glob("*.txt"):
            d = self.process_file(str(file))
            docs.append(d)
        print(f"Processed {len(docs)} documents")
        return docs
