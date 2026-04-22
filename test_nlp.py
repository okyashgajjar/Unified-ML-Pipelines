import pandas as pd
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from Superwised_Classification.nlp_data.nlp_class import nlp_classification

# Load data
df = pd.read_csv('email.csv')
print(f"Loaded {len(df)} rows from email.csv")

# Run classification
print("Starting NLP classification...")
results = nlp_classification(df, 'Category')

# Display results
print("\n--- NLP Classification Results ---")
if 'accuracy' in results.columns:
    print(results.drop(columns=['pipeline']).sort_values(by='accuracy', ascending=False))
else:
    print(results)
