import os
import re
import pickle
import numpy as np
import faiss
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI

# Load API Key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_FOLDER = "Data/"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

def clean_text(text):
    """Cleans legal text noise."""
    text = re.sub(r"THE PAKISTAN PENAL CODE", "", text, flags=re.IGNORECASE)
    text = re.sub(r"CONSTITUTION OF PAKISTAN", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Page \d+ of \d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()

def get_chunks(text, chunk_size=1000, overlap=200):
    """Splits text into chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def get_embedding(text):
    """Generates embedding using OpenAI."""
    response = client.embeddings.create(
        input=text.replace("\n", " "),
        model="text-embedding-3-small"  # Cheaper and better than ada-002
    )
    return response.data[0].embedding

def main():
    print("Loading PDFs...")
    all_chunks = []
    
    # 1. Read and Chunk PDFs
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".pdf"):
            print(f"Processing {file}...")
            reader = PdfReader(os.path.join(DATA_FOLDER, file))
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
            
            cleaned_text = clean_text(full_text)
            chunks = get_chunks(cleaned_text)
            
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "source": file
                })

    print(f"Total chunks created: {len(all_chunks)}")

    # 2. Create Embeddings
    print("Generating Embeddings (this may take a moment)...")
    embeddings = []
    for i, doc in enumerate(all_chunks):
        emb = get_embedding(doc["text"])
        embeddings.append(emb)
        if i % 50 == 0:
            print(f"Embedded {i}/{len(all_chunks)} chunks...")

    # 3. Save to FAISS
    print("Saving to FAISS...")
    dimension = len(embeddings[0])
    np_embeddings = np.array(embeddings).astype('float32')
    
    index = faiss.IndexFlatL2(dimension)
    index.add(np_embeddings)

    # Save Index and Metadata
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    print("✅ Ingestion Complete! 'faiss_index.bin' and 'metadata.pkl' created.")

if __name__ == "__main__":
    main()