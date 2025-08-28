# statement_segmenter_utils.py

import pandas as pd
import numpy as np
import re
import joblib
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

def extract_features(text, nlp):
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]

    return {
        "starts_with_modal": int(bool(re.match(r"^(should|must|can|will|would|shall)\b", text.lower()))),
        "contains_question_mark": int("?" in text),
        "length_tokens": len(text.split()),
        "contains_numbers": int(bool(re.search(r"\d", text))),
        "starts_with_amendment_term": int(bool(re.match(r"^(amendment|clause|section)\b", text.lower()))),
        "has_actor_reference": int(bool(re.search(r"\b(we|government|minist(er|ry))\b", text.lower()))),
        "has_modal_verb": int(bool(re.search(r"\b(should|must|can|could|might|will|shall|would|may)\b", text.lower()))),
        "has_adj_or_adv": int("ADJ" in pos_tags or "ADV" in pos_tags)
    }

def split_into_sentences(text, nlp):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def starts_with_anaphora(sentence):
    anaphora = ["this", "that", "these", "those", "they", "them", "he", "she", "it", "such", "the former", "the latter"]
    first_word = sentence.lower().split()[0] if sentence else ""
    return first_word in anaphora

def group_similar_sentences(sentences, sbert, threshold=0.45, max_group_size=3):
    if not sentences:
        return []

    embeddings = sbert.encode(sentences, convert_to_numpy=True)
    groups = []
    current_group = [sentences[0]]

    for i in range(1, len(sentences)):
        # If current_group was flushed last iteration, start anew
        if not current_group:
            current_group = [sentences[i]]
            continue

        sim = cosine_similarity(embeddings[i:i+1], embeddings[i-1:i])[0][0]
        if sim >= threshold or starts_with_anaphora(sentences[i]):
            current_group.append(sentences[i])
        else:
            groups.append(" ".join(current_group))
            current_group = [sentences[i]]

        if len(current_group) >= max_group_size:
            groups.append(" ".join(current_group))
            current_group = []

    if current_group:
        groups.append(" ".join(current_group))

    return groups

def extract_statements(input_csv, model_path="MLP_statement_segmenter.joblib", output_csv="actor_org_statements_confidence.csv"):
    print("🔍 Extracting statements...")

    artifact = joblib.load(model_path)
    if isinstance(artifact, dict):
        mlp = artifact["model"]
        scaler = artifact.get("scaler", None)
    else:
        mlp = artifact
        scaler = None

    sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    nlp = spacy.load("en_core_web_sm")

    df = pd.read_csv(input_csv)
    output = []

    for _, row in df.iterrows():
        speaker = row.get("speaker_org") or "UNKNOWN"
        speech = str(row.get("speech", ""))
        sentences = split_into_sentences(speech, nlp)
        chunks = group_similar_sentences(sentences, sbert, threshold=0.45, max_group_size=3)

        for chunk in chunks:
            # Handcrafted features -> vector (preserve order from function)
            feats = extract_features(chunk, nlp)
            feat_vector = np.array([list(feats.values())], dtype=np.float32)

            # Apply the same scaling used in training
            if scaler is not None:
                feat_vector = scaler.transform(feat_vector)

            # Embedding (L2-normalised to match training)
            embed_vector = sbert.encode([chunk], convert_to_numpy=True)
            embed_vector = normalize(embed_vector, norm="l2")  # shape (1, dim)

            # Concatenate
            X = np.hstack([embed_vector, feat_vector])

            # Predict
            proba = mlp.predict_proba(X)[0][1]
            label = mlp.predict(X)[0]

            if label == 1:
                output.append({
                    "speaker_org": speaker,
                    "statement": chunk,
                    "confidence": round(float(proba), 4)
                })

    output_df = pd.DataFrame(output)
    output_df.to_csv(output_csv, index=False)
    print(f"✅ Statements extracted and saved to {output_csv}")

    return output_df
