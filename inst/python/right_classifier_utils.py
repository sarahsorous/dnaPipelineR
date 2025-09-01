# right_classifier_utils.py

# Load libraries
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import spacy

# POS tag set used to build frequency features (match training)
POS_TAGS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'AUX', 'PRON', 'ADP', 'CCONJ', 'DET', 'NUM', 'PROPN']

def extract_pos_features(texts, nlp):
    feats = []
    # match training steup: disable parser/ner, keep tagger for POS only
    for doc in nlp.pipe(texts, disable=["ner", "parser"]):
        counts = {t: 0 for t in POS_TAGS}
        for tok in doc:
            # Count only POS we track
            if tok.pos_ in counts:
                counts[tok.pos_] += 1
        # Convert counts -> normalised frequencies (avoid divide by 0)
        total = sum(counts.values()) + 1e-6
        feats.append([counts[t] / total for t in POS_TAGS])
    return np.array(feats, dtype=np.float32)

def assign_right_labels(
    input_csv: str = "actor_org_statements_confidence.csv",
    output_csv: str = "statement_rights.csv",
    model_path: str = "right_svm.joblib",                 # trained linear SVM (saved during training)
    encoder_path: str = "right_label_encoder.joblib",     # fitted LabelEncoder for string ↔ int mapping
    sbert_path: str = None                                # kept for backward compat; ignored
):
    print("🔍 Assigning rights...")

    # Load trained SVM and label encoder
    svm = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    # SBERT eoncdoer
    sbert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    # Load spaCy model for POS tagging
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as e:
        # Tells users to install model if error occurs
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is required for POS features. "
            "Install with: python -m spacy download en_core_web_sm"
        ) from e

    # Read input statements
    df = pd.read_csv(input_csv)
    texts = df["statement"].astype(str).tolist()

    # Encode features exactly like training: SBERT embeddings + POS frequencies (concatenated)
    print("🔡 Encoding SBERT embeddings...")
    embeddings = sbert.encode(texts, show_progress_bar=True)
    pos_feats = extract_pos_features(texts, nlp)
    X = np.hstack([embeddings, pos_feats])

    # Predict class with the trained SVM and map back to original labels
    preds = svm.predict(X)
    pred_labels = label_encoder.inverse_transform(preds)

    # Save output
    out = df.copy()
    out["right"] = pred_labels
    out[["speaker_org", "statement", "right"]].to_csv(output_csv, index=False)
    print(f"✅ Right labels assigned and saved to {output_csv}")

    return out[["speaker_org", "statement", "right"]]
