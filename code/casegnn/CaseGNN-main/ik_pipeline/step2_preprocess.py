"""
Step 2 – Extract features (facts / issues) from raw Indian legal case text,
         run NER + dependency-parse-based relation extraction,
         and produce structured CSVs consumed by later stages.

For each case file the script produces:
  output/ie/{split}_{feat}/result/{case_id}.csv

CSV format (matches CaseGNN expectation):
  Type,Entity1,Relationship,Type,Entity2
"""
import csv
import json
import os
import re
import sys
from tqdm import tqdm

import spacy

from config import (
    IK_TRAIN_DIR, IK_TEST_DIR, IE_DIR, LABEL_DIR,
    SPACY_MODEL, ensure_dirs
)

# ────────────────────────────────────────────────────────────────────────────
# Section-based feature extraction for Indian legal judgments
# ────────────────────────────────────────────────────────────────────────────

# Common section headers in Indian Supreme Court cases
_SECTION_RE = re.compile(
    r'\b(PETITIONER|RESPONDENT|DATE OF JUDGMENT|BENCH|CITATION|'
    r'CITATOR INFO|ACT|HEADNOTE|JUDGMENT)\s*:', re.IGNORECASE)


def split_sections(text):
    """Split an Indian legal case into named sections."""
    parts = _SECTION_RE.split(text)
    sections = {}
    i = 0
    while i < len(parts):
        if _SECTION_RE.match(parts[i].strip() + ":"):
            key = parts[i].strip().upper()
            val = parts[i + 1] if i + 1 < len(parts) else ""
            sections[key] = val.strip()
            i += 2
        else:
            sections.setdefault("PREAMBLE", "")
            sections["PREAMBLE"] += parts[i]
            i += 1
    return sections


def extract_fact_issue(text):
    """
    Return (fact_text, issue_text) from a raw Indian legal case.
    Fact  = JUDGMENT section  (the factual narrative / reasoning)
    Issue = HEADNOTE + ACT    (legal issues, provisions, head-notes)
    Falls back to a simple length-based split when sections are absent.
    """
    sections = split_sections(text)

    fact_text = sections.get("JUDGMENT", "").strip()
    issue_parts = []
    if "HEADNOTE" in sections:
        issue_parts.append(sections["HEADNOTE"])
    if "ACT" in sections:
        issue_parts.append(sections["ACT"])
    issue_text = " ".join(issue_parts).strip()

    # Fallback: if either is empty, split text roughly in half
    if not fact_text and not issue_text:
        mid = len(text) // 2
        fact_text = text[:mid]
        issue_text = text[mid:]
    elif not fact_text:
        fact_text = issue_text  # duplicate if only one available
    elif not issue_text:
        issue_text = fact_text

    # Truncate to avoid extremely long texts (spaCy can be slow)
    MAX_CHARS = 50_000
    fact_text = fact_text[:MAX_CHARS]
    issue_text = issue_text[:MAX_CHARS]

    return fact_text, issue_text


# ────────────────────────────────────────────────────────────────────────────
# spaCy-based NER + relation (SVO) extraction
# ────────────────────────────────────────────────────────────────────────────

def extract_entities(doc):
    """Return {entity_text: entity_label} from a spaCy Doc."""
    ent_dict = {}
    for ent in doc.ents:
        ent_dict[ent.text] = ent.label_
    return ent_dict


def extract_svo_triplets(doc):
    """
    Extract (subject, verb/relation, object) triplets via dependency parsing.
    Returns list of (subj_text, rel_text, obj_text).
    """
    triplets = []
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
            subject = _expand_compound(token)
            verb = token.head
            # Collect direct objects, prepositional objects, attributes
            for child in verb.children:
                if child.dep_ in ("dobj", "pobj", "attr", "oprd", "acomp"):
                    obj = _expand_compound(child)
                    triplets.append((subject, verb.lemma_, obj))
            # Also look at prepositional phrases
            for child in verb.children:
                if child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            obj = _expand_compound(pobj)
                            triplets.append((subject, verb.lemma_ + " " + child.text, obj))
    return triplets


def _expand_compound(token):
    """Expand a token to include its compound modifiers."""
    parts = []
    for child in token.children:
        if child.dep_ in ("compound", "amod", "det"):
            parts.append(child.text)
    parts.append(token.text)
    return " ".join(parts)


def _sanitize(s):
    """Remove commas (CSV delimiter) and newlines from a string."""
    return s.replace(",", " ").replace("\n", " ").replace("\r", " ").strip()


def build_structured_csv(triplets, ent_dict):
    """
    Convert triplets + NER dict into rows:
      [(Type, Entity1, Relationship, Type, Entity2), ...]
    """
    rows = []
    for subj, rel, obj in triplets:
        type1 = ent_dict.get(subj, "None")
        type2 = ent_dict.get(obj, "None")
        rows.append((
            _sanitize(type1),
            _sanitize(subj),
            _sanitize(rel),
            _sanitize(type2),
            _sanitize(obj),
        ))
    return rows


# ────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ────────────────────────────────────────────────────────────────────────────

def read_case_text(ik_dir, case_name):
    for subdir in ["query", "candidate"]:
        p = os.path.join(ik_dir, subdir, case_name)
        if os.path.isfile(p):
            with open(p, "r", errors="replace") as f:
                return f.read()
    return ""


def process_split(split_name, ik_dir, nlp):
    """Process all cases for one split (train or test)."""
    case_list_path = os.path.join(LABEL_DIR, f"{split_name}_case_list.json")
    with open(case_list_path, "r") as f:
        case_list = json.load(f)

    for feat_type in ["fact", "issue"]:
        out_dir = os.path.join(IE_DIR, f"{split_name}_{feat_type}", "result")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n── Processing {split_name} / {feat_type}  ({len(case_list)} cases) ──")
        for case_name in tqdm(case_list):
            case_id = case_name.replace(".txt", "")
            csv_path = os.path.join(out_dir, f"{case_id}.csv")

            if os.path.exists(csv_path):
                continue  # skip already processed

            raw_text = read_case_text(ik_dir, case_name)
            if not raw_text.strip():
                # Write minimal CSV (header only) so downstream doesn't crash
                with open(csv_path, "w") as fout:
                    fout.write("Type,Entity1,Relationship,Type,Entity2\n")
                continue

            fact_text, issue_text = extract_fact_issue(raw_text)
            text = fact_text if feat_type == "fact" else issue_text

            # spaCy processing (limit to first 100k chars to keep memory reasonable)
            doc = nlp(text[:100_000])
            ent_dict = extract_entities(doc)
            triplets = extract_svo_triplets(doc)
            rows = build_structured_csv(triplets, ent_dict)

            with open(csv_path, "w", newline="") as fout:
                fout.write("Type,Entity1,Relationship,Type,Entity2\n")
                writer = csv.writer(fout)
                for row in rows:
                    writer.writerow(row)


def main():
    ensure_dirs()

    print(f"Loading spaCy model: {SPACY_MODEL}")
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        print(f"Model '{SPACY_MODEL}' not found. Trying en_core_web_sm …")
        nlp = spacy.load("en_core_web_sm")

    # Increase max text length
    nlp.max_length = 200_000

    process_split("train", IK_TRAIN_DIR, nlp)
    process_split("test", IK_TEST_DIR, nlp)

    print("\nDone. Structured CSVs saved to:", IE_DIR)


if __name__ == "__main__":
    main()
