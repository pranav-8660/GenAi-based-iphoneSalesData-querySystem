import pandas as pd
import numpy as np
import faiss
import openai
import time
import ast
import openaiapikeyPv as apiKeyy

# Set OpenAI API key
openai.api_key = apiKeyy.provideApiKey()

# Paths
DATASET_PATH = r"iphone.csv"  # Input dataset
EMBEDDINGS_PATH = r"iphone_sales_data_with_embeddings.csv"  # Output embeddings file
INDEX_PATH = r"iphone_sales_index.index"  # Output FAISS index file

# Load dataset
print("Loading dataset...")
try:
    dataset = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded successfully with {len(dataset)} rows.")
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATASET_PATH}. Please check the path.")
    exit()

# Function to combine columns into a single text string
def concatenate_columns(row):
    return " ".join([
        str(row.get('productAsin', '')),
        str(row.get('country', '')),
        str(row.get('date', '')),
        str(row.get('isVerified', '')),
        str(row.get('ratingScore', '')),
        str(row.get('reviewTitle', '')),
        str(row.get('reviewDescription', '')),
        str(row.get('reviewUrl', '')),
        str(row.get('reviewedIn', '')),
        str(row.get('variant', '')),
        str(row.get('variantAsin', ''))
    ])

# Create a combined_text column
print("Creating combined text column...")
dataset['combined_text'] = dataset.apply(concatenate_columns, axis=1)

# Function to generate embeddings with retries
def get_embeddings_batch(texts, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [embedding['embedding'] for embedding in response['data']]
        except openai.error.RateLimitError:
            retries += 1
            wait_time = 10 * (2 ** retries)
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds... (Retry {retries}/{max_retries})")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            break
    raise Exception("Max retries reached for OpenAI API.")

# Generate embeddings for each row
print("Generating embeddings...")
embeddings = []
batch_size = 10  # Process 10 rows at a time to optimize API calls
for i in range(0, len(dataset), batch_size):
    batch_texts = dataset['combined_text'][i:i + batch_size].tolist()
    try:
        batch_embeddings = get_embeddings_batch(batch_texts)
        embeddings.extend(batch_embeddings)
    except Exception as e:
        print(f"Error processing batch {i // batch_size + 1}: {e}")
        embeddings.extend([[]] * len(batch_texts))  # Append empty lists for failed batches

# Check if all embeddings were generated
if len(embeddings) != len(dataset):
    print(f"Error: Mismatch in embeddings ({len(embeddings)}) and dataset rows ({len(dataset)}). Exiting.")
    exit()

# Add embeddings to the dataset
dataset['embedding'] = embeddings

# Save embeddings to CSV
print(f"Saving embeddings to {EMBEDDINGS_PATH}...")
dataset.to_csv(EMBEDDINGS_PATH, index=False)
print("Embeddings saved successfully.")

# Prepare data for FAISS
print("Creating FAISS index...")
try:
    embedding_array = np.array([e for e in embeddings if len(e) > 0]).astype('float32')  # Exclude empty embeddings
    index = faiss.IndexFlatL2(embedding_array.shape[1])
    index.add(embedding_array)
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index created and saved to {INDEX_PATH}.")
except Exception as e:
    print(f"Error creating FAISS index: {e}")
