import os
import pickle
import json
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from langdetect import detect

# ---------------- 1. CONFIGURATION ----------------
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("CRITICAL ERROR: OPENAI_API_KEY not found. Check .env file.")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

print("Loading Knowledge Base...")
try:
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        chunks_metadata = pickle.load(f)
    print("✅ Knowledge Base Loaded.")
except Exception as e:
    print(f"❌ Error loading database: {e}")
    index = None
    chunks_metadata = []

app = FastAPI()

# ---------------- 2. DATA MODELS (FRIEND'S UI REQUIREMENT) ----------------
class Query(BaseModel):
    query: str

class StructuredAnswer(BaseModel):
    crime_title_en: str
    crime_title_ur: str
    laws: list # List of objects [{"section": "302", "title_en": "Murder", "title_ur": "..."}]
    punishment_en: str
    punishment_ur: str
    explanation_en: str
    explanation_ur: str

# ---------------- 3. HELPERS ----------------
def get_embedding(text: str):
    res = client.embeddings.create(
        input=text.replace("\n", " "),
        model="text-embedding-3-small"
    )
    return res.data[0].embedding

def translate_if_urdu(text: str):
    try:
        if detect(text) == "ur":
            print("Urdu detected. Translating...")
            t = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "Translate to English:"}, 
                          {"role": "user", "content": text}]
            )
            return t.choices[0].message.content.strip()
    except:
        pass
    return text

# ---------------- 4. THE JUNIOR LAWYER (FULL POWER) ----------------
def refine_query_to_legal_terms(user_query):
    """
    Translates layman stories into PRECISE legal keywords.
    Uses 10-Shot Prompting to ensure high accuracy.
    """
    system_prompt = """
    You are the Chief Justice of the Supreme Court of Pakistan.
    Your job is to translate a layman's story into PRECISE SEARCH QUERIES for the Pakistan Penal Code (PPC).
    
    STRATEGY:
    1. Identify the ACT (What was done?).
    2. Identify the HARM (What was the result? Death, Injury, Loss?).
    3. Identify the INTENT (Was it on purpose, accidental, or negligent?).
    4. Convert these into legal keywords and section numbers.
    
    OUTPUT FORMAT:
    Legal Concept 1 | Legal Concept 2 | Legal Concept 3
    
    --- EXAMPLES (STUDY THESE CAREFULLY) ---
    
    Input: "Someone stole my bike from outside my house."
    Output: Theft Section 378 | Punishment for theft Section 379
    
    Input: "Two men stopped me on the road, showed a gun, and took my wallet."
    Output: Robbery Section 390 | Punishment for Robbery Section 392 | Snatching
    
    Input: "A shopkeeper sold expired milk and the customer died after drinking it."
    Output: Sale of noxious food or drink Section 273 | Qatl-bis-sabab Section 321 | Punishment for Qatl-bis-sabab Section 322
    
    Input: "My neighbor built a wall on my land without permission."
    Output: Criminal Trespass Section 441 | Punishment for criminal trespass Section 447 | Illegal Possession
    
    Input: "He hit me with a stick and broke my tooth."
    Output: Hurt Shajjah Section 337 | Itlaf-i-udw Section 333 | Punishment for Itlaf-i-udw Section 334
    
    Input: "A man is following my daughter and harassing her on the street."
    Output: Assault or criminal force to woman Section 354 | Sexual harassment | Loitering
    
    Input: "They kidnapped my brother to ask for ransom money."
    Output: Kidnapping for extorting property Section 365A | Punishment for kidnapping Section 363
    
    Input: "Someone signed my name on a fake property document."
    Output: Forgery Section 463 | Making a false document Section 464 | Forgery for purpose of cheating Section 468
    
    Input: "A mob attacked the police station and burned it down."
    Output: Mischief by fire Section 436 | Rioting Section 146 | Unlawful Assembly Section 141
    
    Input: "He gave me a check but it bounced because he had no money."
    Output: Dishonestly issuing a cheque Section 489F | Cheating Section 415
    
    --- END OF EXAMPLES ---
    
    Analyze the User Input below and generate the best search queries separated by '|'.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_query}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except:
        return user_query

# ---------------- 5. THE API ENDPOINT (MERGED LOGIC) ----------------
@app.post("/ask", response_model=StructuredAnswer)
async def ask(user: Query):
    if index is None:
        raise HTTPException(status_code=503, detail="System not ready. Run ingest.py first.")

    # A. PRE-PROCESSING
    original_query = user.query
    processed_query = translate_if_urdu(original_query)

    # B. JUNIOR LAWYER (Your 'Brain')
    # We use the Chief Justice persona to find "Section 321" even if user didn't say it.
    legal_search_terms = refine_query_to_legal_terms(processed_query)
    print(f"--- Search Terms: {legal_search_terms} ---")
    
    search_queries = [q.strip() for q in legal_search_terms.split('|')]

    # C. FAN-OUT SEARCH (Your 'Search Strategy')
    all_retrieved_indices = []
    for query in search_queries:
        if query:
            vec = np.array([get_embedding(query)], dtype="float32")
            _, idxs = index.search(vec, k=5)
            all_retrieved_indices.extend(idxs[0])

    unique_indices = list(set(all_retrieved_indices))

    # D. CONTEXT BUILDING
    context_text = ""
    for i in unique_indices:
        if i < len(chunks_metadata):
            context_text += chunks_metadata[i]["text"] + "\n"

    # E. SUPERVISOR (Merged: Your 'Gap Filling' + His 'JSON Format')
    # We apply the 'Chief Justice' persona here too for authority.
    
    system_prompt = """
    You are the Chief Justice of the Supreme Court of Pakistan.
    
    PHASE 1: JUDICIAL ANALYSIS (Internal Thought)
    1. Analyze the User Case vs. PDF Context.
    2. **CRITICAL:** Check for GAPS. If the user mentions DEATH but the PDF only has 'Noxious Food' (Section 273), you MUST ADD Section 321/322 (Qatl-bis-sabab) from your internal knowledge.
    3. Ensure the punishment matches the severity of the crime described.
    
    PHASE 2: FORMATTING (Output)
    Output the result in STRICT JSON format matching the schema below.
    
    JSON SCHEMA:
    {
     "crime_title_en": "Short Title (e.g. Sale of Noxious Food & Accidental Death)",
     "crime_title_ur": "Urdu Translation",
     "laws": [
       {"section": "Section 273", "title_en": "Sale of noxious food", "title_ur": "Urdu Title"},
       {"section": "Section 321", "title_en": "Qatl-bis-sabab", "title_ur": "Urdu Title"}
     ],
     "punishment_en": "Explain imprisonment and fines clearly.",
     "punishment_ur": "Urdu translation.",
     "explanation_en": "Simple explanation of why these laws apply.",
     "explanation_ur": "Urdu translation."
    }
    """
    
    user_message = f"User Case: {processed_query}\n\nRetrieved PDF Context:\n{context_text}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0
    )

    # F. VALIDATION (To ensure his App doesn't crash)
    try:
        data = json.loads(response.choices[0].message.content)
        return data
    except Exception as e:
        print(f"JSON Error: {e}")
        # Fallback if AI messes up JSON (rare with temp=0)
        raise HTTPException(status_code=500, detail="AI failed to format response.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)