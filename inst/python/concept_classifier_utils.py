# concept_classifier_utils.py

import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

def _softmax(a: np.ndarray) -> np.ndarray:
    """Stable row-wise softmax for decision_function scores."""
    a = np.asarray(a, dtype=np.float64)
    a = a - np.max(a, axis=1, keepdims=True)
    np.exp(a, out=a)
    a_sum = np.sum(a, axis=1, keepdims=True)
    # avoid divide-by-zero in degenerate cases
    a_sum[a_sum == 0.0] = 1.0
    a /= a_sum
    return a

def assign_concepts(
    input_csv: str = "actor_org_statements_confidence.csv",
    output_csv: str = "statement_concepts.csv",
    threshold: float = 0.6,
    model_path: str = "concept_svm.joblib",
    encoder_path: str = "concept_label_encoder.joblib",
    sbert_path: str = None  # kept for backward compatibility; ignored
):
    print("🔍 Assigning concepts...")

    # Load artifacts
    svm = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    # Instantiate SBERT by name
    sbert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    # Read input
    df = pd.read_csv(input_csv)
    texts = df["statement"].astype(str).tolist()

    # Encode with SBERT (to match training)
    print("🔡 Encoding statements with SBERT...")
    X = sbert.encode(texts, show_progress_bar=True)

    # SVM was saved with probability=False -> use decision_function
    scores = svm.decision_function(X)  # shape: (n_samples, n_classes) or (n_samples,) for binary

    # Normalize shape for binary case to (n_samples, 2): class 0 score is -score, class 1 is +score
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)

    # Convert margins to pseudo-probabilities (keeps your threshold semantics)
    probs = _softmax(scores)

    # Top-k indices by prob (k up to 3, but never exceeding #classes)
    n_classes = probs.shape[1]
    k = min(3, n_classes)
    topk_idx = np.argsort(probs, axis=1)[:, -k:][:, ::-1]

    # Map column indices -> actual encoded class ids -> original string labels
    class_ids = svm.classes_  # encoded integer labels used to train the SVM
    topk_class_ids = class_ids[topk_idx]
    # Vectorized inverse transform back to label strings
    flat_labels = label_encoder.inverse_transform(topk_class_ids.ravel())
    topk_labels = flat_labels.reshape(topk_class_ids.shape)

    # Build results
    results = []
    for i, row in df.iterrows():
        speaker = row.get("speaker_org", "UNKNOWN")
        text = row["statement"]

        top1_idx = topk_idx[i, 0]
        top1_prob = probs[i, top1_idx]

        if top1_prob >= threshold:
            concept = topk_labels[i, 0]
        else:
            concept = "; ".join(topk_labels[i, :k].tolist())

        results.append({
            "speaker_org": speaker,
            "statement": text,
            "concept": concept
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"✅ Concepts assigned and saved to {output_csv}")

    return results_df
