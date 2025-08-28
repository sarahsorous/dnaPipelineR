import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def analyse_sentiment(
    input_csv="actor_org_statements_confidence.csv",
    output_csv="actor_org_statements_with_sentiment.csv",
    model_name="cardiffnlp/twitter-roberta-base-sentiment",
    map_labels=True
):
    print("💬 Performing sentiment analysis...", flush=True)

    # Load the data
    df = pd.read_csv(input_csv)

    # Load sentiment pipeline (cached after first use)
    classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

    # Apply sentiment analysis
    df['sentiment'] = df['statement'].apply(lambda x: classifier(str(x))[0]['label'])

    if map_labels:
        label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
        df['sentiment_label'] = df['sentiment'].map(label_map)

    # Export
    df.to_csv(output_csv, index=False)
    print(f"✅ Sentiment saved to {output_csv}", flush=True)

    return df
