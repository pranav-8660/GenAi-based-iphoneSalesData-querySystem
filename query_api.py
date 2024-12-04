from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
import openai
import ast
import openaiapikeyPv as apiKeyy
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your frontend URL (replace if different)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Set OpenAI API key
openai.api_key = apiKeyy.provideApiKey()

# Paths
EMBEDDINGS_PATH = r"C:\Users\Pranav\Desktop\finalLLMModel\dataset\iphone_sales_data_with_embeddings.csv"
INDEX_PATH = r"C:\Users\Pranav\Desktop\finalLLMModel\dataset\iphone_sales_index.index"

# Load embeddings and FAISS index
print("Loading embeddings...")
dataset = pd.read_csv(EMBEDDINGS_PATH)
dataset['embedding'] = dataset['embedding'].apply(ast.literal_eval)
embeddings = np.array(dataset['embedding'].tolist()).astype('float32')

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

# Define query model
class Query(BaseModel):
    question: str
    top_k: int = 3

# Function to generate embedding for a query
def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']

# Function to interact with GPT-3.5 Turbo
def query_gpt3(question, context):
    prompt = f"Question: {question}\nContext: {context}"
    
    # Check if the prompt is overly repetitive or too long
    if len(prompt.split()) > 2000:  # You can adjust this limit
        print("Prompt is too long. Reducing context.")
        # Trim context or summarize it
        context = context[:1000]  # Example of trimming context

    print(f"Prompt:\n{prompt[:500]}...")  # Print the first 500 characters for review (optional)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except openai.error.InvalidRequestError as e:
        print(f"Error with request: {e}")
        return None


# Search function
@app.post("/query/")
def search_sales_data(query: Query):
    # Step 1: Convert the question to an embedding
    question_embedding = np.array(get_embedding(query.question)).astype('float32').reshape(1, -1)
    
    # Step 2: Search FAISS index for the most relevant entries
    distances, indices = index.search(question_embedding, query.top_k)
    
    if not indices.any():
        raise HTTPException(status_code=404, detail="No relevant results found.")
    
    # Step 3: Retrieve the results
    results = dataset.iloc[indices[0]].copy()
    results['distance'] = distances[0]
    
    # Combine the context for GPT-3
    context = "\n".join(results['combined_text'].tolist())
    
    # Step 4: Query GPT-3.5 Turbo with the results as context
    gpt_response = query_gpt3(query.question, context)
    
    return {
        "question": query.question,
        "context_used": context,
        "gpt_response": gpt_response
    }

# Example query for debugging
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
