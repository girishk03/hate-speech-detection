import pandas as pd
import re
import nltk
import nlpaug.augmenter.word as naw

# -----------------------------
# Download required NLTK resources
# -----------------------------
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv('Datasets/ytdata.csv')

# -----------------------------
# Cleaning
# -----------------------------
df = df.dropna(subset=['text'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Remove duplicates
df = df.drop_duplicates(subset='clean_text')

# Keep only required columns
df = df[['clean_text', 'label']]

print("After cleaning:")
print(df['label'].value_counts())

# -----------------------------
# Augmenter
# -----------------------------
aug = naw.SynonymAug(aug_src='wordnet')

# -----------------------------
# Function to generate synthetic data
# -----------------------------
def generate_samples(df, label, target):
    subset = df[df['label'] == label]

    current_size = len(subset)

    # If already enough data → no augmentation
    if current_size >= target:
        return pd.DataFrame(columns=['clean_text', 'label'])

    needed = target - current_size
    new_rows = []

    for _ in range(needed):
        text = subset.sample(1)['clean_text'].values[0]

        try:
            aug_text = aug.augment(text)

            if isinstance(aug_text, list):
                aug_text = aug_text[0]

            # Avoid duplicates
            if aug_text not in subset['clean_text'].values:
                new_rows.append({
                    "clean_text": aug_text,
                    "label": label
                })

        except:
            continue

    return pd.DataFrame(new_rows)

# -----------------------------
# Target size (balanced dataset)
# -----------------------------
TARGET = df['label'].value_counts().max()

# -----------------------------
# Generate synthetic data
# -----------------------------
positive_syn = generate_samples(df, "positive", TARGET)
neutral_syn = generate_samples(df, "neutral", TARGET)
negative_syn = generate_samples(df, "negative", TARGET)

# -----------------------------
# Merge datasets
# -----------------------------
df_final = pd.concat([
    df,
    positive_syn,
    neutral_syn,
    negative_syn
], ignore_index=True)

# Shuffle
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# -----------------------------
# Save dataset
# -----------------------------
df_final.to_csv("Datasets/final_balanced_dataset.csv", index=False)

# -----------------------------
# Final distribution
# -----------------------------
print("\nFinal dataset distribution:")
print(df_final['label'].value_counts())

print("\n Synthetic dataset saved successfully!")