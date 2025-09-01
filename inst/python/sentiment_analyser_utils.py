# sentiment_analyser_utils.py

# Load libraries
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def analyse_sentiment(
    input_csv="actor_org_statements_confidence.csv",
    output_csv="actor_org_statements_with_sentiment.csv",
    model_name="cardiffnlp/twitter-roberta-base-sentiment",
    map_labels=True
):
    print("💬 Performing sentiment analysis...", flush=True)

    # Load input data
    df = pd.read_csv(input_csv)

    # Initialise HuggingFace sentiment-analysis
    # downloads models on first use, then cached locally
    classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

    # Apply sentiment analysis on each statement
    # converts to string to avoid errors
    df['sentiment'] = df['statement'].apply(lambda x: classifier(str(x))[0]['label'])

    if map_labels:
        # Map raw model labels to human-readable classes
        label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
        df['sentiment_label'] = df['sentiment'].map(label_map)

    # Save results with sentiment column
    df.to_csv(output_csv, index=False)
    print(f"✅ Sentiment saved to {output_csv}", flush=True)

    return df
