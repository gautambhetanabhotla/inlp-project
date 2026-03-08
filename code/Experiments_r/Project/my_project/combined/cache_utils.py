import os
import json
import re
import pickle
import spacy
from tqdm import tqdm

try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except:
    os.system("python3 -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def extract_legal_concepts(text):
    textStr = text.lower()
    concepts = []
    
    citations = re.findall(r'\b\d{4}\s+(?:scc|air|scr|ilr|scale|jt)\s+(?:sc|hc)?\s*\d+\b', textStr)
    statutes = re.findall(r'\b(?:section|sec\.|article|art\.|rule|order)\s+[a-z0-9ivx]+\b', textStr)
    acts = re.findall(r'\b(?:indian penal code|ipc|crpc|cpc|evidence act|constitution(?:\s+of\s+india)?|income tax act)\b', textStr)
    
    concepts.extend([c.replace(" ", "_") for c in citations])
    concepts.extend([s.replace(" ", "_") for s in statutes])
    concepts.extend([a.replace(" ", "_") for a in acts])
    
    # Process text in chunks to prevent SpaCy memory errors
    max_chunk = 500000 
    text_chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    for chunk in text_chunks:
        doc = nlp(chunk)
        for token in doc:
            if token.is_stop or token.is_punct or len(token.lemma_) <= 2:
                continue
            if token.pos_ in ["NOUN", "PROPN", "VERB"]:
                if not token.lemma_.isnumeric():
                    concepts.append(token.lemma_.lower())
    return " ".join(concepts)


def extract_legal_entities(text):
    """ Same as concepts but strictly citations and statutes for graph diffusion """
    text = text.lower()
    entities = []
    citations = re.findall(r'\b\d{4}\s+(?:scc|air|scr|ilr|scale|jt)\s+(?:sc|hc)?\s*\d+\b', text)
    statutes = re.findall(r'\b(?:section|sec\.|article|art\.|rule|order)\s+[a-z0-9ivx]+\b', text)
    acts = re.findall(r'\b(?:indian penal code|ipc|crpc|cpc|evidence act|constitution.*?india|income tax act)\b', text)
    entities.extend([c.replace(" ", "_") for c in citations])
    entities.extend([s.replace(" ", "_") for s in statutes])
    entities.extend([a.replace(" ", "_") for a in acts])
    return " ".join(entities)


def load_raw_documents(folder_path):
    docs = {}
    if not os.path.exists(folder_path): return docs
    filenames = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    for filename in tqdm(filenames, desc=f"Reading {os.path.basename(folder_path)}", leave=False):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
            docs[filename] = f.read()
    return docs

def get_or_generate_feature(doc_dict, cache_file, extraction_fn, desc):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
            
    feature_dict = {}
    for doc_id, text in tqdm(doc_dict.items(), desc=desc, leave=False):
        feature_dict[doc_id] = extraction_fn(text)
        
    with open(cache_file, "wb") as f:
        pickle.dump(feature_dict, f)
    return feature_dict
