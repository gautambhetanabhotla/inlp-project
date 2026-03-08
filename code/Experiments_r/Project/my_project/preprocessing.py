import os
import re
import spacy
from tqdm import tqdm

# Load light-weight spacy model for tokenization and lemmatization
# Disable NER and Parser to speed up processing
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(text):
    """
    Cleans the raw legal text by removing numbers, punctuation,
    and converting to lowercase.
    """
    # Remove standard punctuation and special characters, keep words and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    """
    Applies spacy to tokenize, lemmatize, and remove stopwords.
    Returns a list of clean tokens.
    """
    cleaned = clean_text(text)
    # We increase max_length because legal documents can be very long
    nlp.max_length = max(len(cleaned) + 100, 2000000)
    
    doc = nlp(cleaned)
    tokens = []
    for token in doc:
        # Keep only words that are not stopwords and have length > 2
        if not token.is_stop and len(token.lemma_) > 2:
            tokens.append(token.lemma_)
            
    return tokens

def load_documents(directory):
    """
    Loads all .txt documents from a given directory.
    Returns a dictionary of {doc_id: raw_text}
    """
    docs = {}
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return docs
        
    for filename in tqdm(os.listdir(directory), desc=f"Loading docs from {os.path.basename(directory)}"):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                docs[filename] = f.read()
    return docs

def preprocess_corpus(docs_dict):
    """
    Preprocesses a dictionary of {doc_id: raw_text}.
    Returns a dictionary of {doc_id: list_of_tokens}.
    """
    preprocessed = {}
    for doc_id, text in tqdm(docs_dict.items(), desc="Preprocessing documents"):
        preprocessed[doc_id] = preprocess_text(text)
    return preprocessed
