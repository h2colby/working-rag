from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
import torch
from transformers import AutoTokenizer, AutoModel
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Define Pinecone index name
PINECONE_INDEX_NAME = "h2safety-embeddings"

# Connect to the Pinecone serverless instance
serverless_spec = ServerlessSpec(cloud="aws", region="us-east1-gcp")
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist. Please check your Pinecone setup.")

pinecone_index = pc.Index(name=PINECONE_INDEX_NAME, spec=serverless_spec)

# Load Llama model and tokenizer for local embeddings
model_path = "/Users/colbydeweese/Downloads/llama-3.2-transformers-3b-v1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Set up the device and tokenizer settings
device = torch.device("cpu")
model.to(device)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Define Local Embedding Class
class LocalLlamaEmbedding:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def embed(self, text):
        """
        Generate embeddings for the input text using the Llama model.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs.to(self.device))
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Take mean pooling
        return embeddings.squeeze().tolist()

# Initialize the embedding model
embedding_model = LocalLlamaEmbedding(model=model, tokenizer=tokenizer, device=device)

# Create Pinecone vector store
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    embedding=embedding_model
)

# Initialize OpenAI model
llm = OpenAI(
    model="gpt-4o-mini",  # Supports up to 8,192 tokens in most configurations
    temperature=0.0,
    max_tokens=6000,  # High token limit for detailed responses
    top_p=1.0,
    frequency_penalty=0.2,
    presence_penalty=0.0,
    api_key=os.environ["OPENAI_API_KEY"],
)

# Updated Query Optimization Prompt
QUERY_OPTIMIZATION_PROMPT = (
    "Expand the following user query into a database search query to maximize relevant results. "
    "Add synonyms, related keywords, and context to improve retrieval precision for hydrogen information. "
    "The optimized query should include multiple related terms to broaden the search scope while maintaining focus.\n\n"
    "User Query: {user_query}\n\n"
    "Optimized Query:"
)

# Explicit QA prompt for the final answer
QUESTION_ANSWER_PROMPT = (
    "Using the user query and retrieved documents, list all hydrogen safety standards, codes, best practices, or other relevant information, not everything requires a standard to be cited:\n\n"
    "1. **Directly quote** from the documents.\n"
    "2. Provide exact references (e.g., standard name, section number).\n\n"
    "User Query: {query_str}\n\n"
    "Final Answer:"
)

custom_prompt = PromptTemplate(template=QUESTION_ANSWER_PROMPT)

# Set up vector store index
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

query_engine = index.as_query_engine(
    embed_model=embedding_model,
    similarity_top_k=15,  # Retrieve more documents for robust answers
    response_mode="tree_summarize",  # Ensure thorough answers
    text_qa_template=custom_prompt,
    debug=True
)

def query_pinecone(query: str, top_k: int = 15):
    """
    Embed the query and perform a similarity search in Pinecone.
    """
    try:
        # Generate query embeddings using the local Llama model
        query_embedding = embedding_model.embed(query)

        # Perform similarity search
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True  # Include metadata in the results
        )

        return results
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return None

def optimize_question(user_query: str) -> str:
    """
    Use OpenAI to optimize the user's query by expanding it with additional context and keywords.
    """
    try:
        optimization_prompt = QUERY_OPTIMIZATION_PROMPT.format(user_query=user_query)
        print(f"Optimization Prompt Sent to OpenAI:\n{optimization_prompt}")

        # Call OpenAI to optimize the query
        optimization_response = llm.complete(optimization_prompt)

        # Extract the optimized query from the response
        optimized_query = optimization_response.text.strip()
        print(f"Optimized Query: {optimized_query}")
        return optimized_query

    except Exception as e:
        print(f"Error optimizing question: {e}")
        return user_query  # Fallback to the original query

def answer_question(user_query: str) -> dict:
    """
    Answer the user's question by querying Pinecone.
    Returns only the AI-generated final answer.
    """
    try:
        optimized_query = optimize_question(user_query)
        print(f"Optimized Query: {optimized_query}")

        # Query Pinecone for relevant documents
        pinecone_results = query_pinecone(optimized_query)
        if not pinecone_results or not pinecone_results.get("matches"):
            retrieved_context = "No relevant context was retrieved from the vector store."
        else:
            retrieved_context = "\n\n".join(
                match.get("metadata", {}).get("text", "")
                for match in pinecone_results["matches"]
                if match.get("metadata")
            )

        # Send the retrieved context and query to OpenAI for a final answer
        final_prompt = (
            f"Using the user query and retrieved documents, put toogether what you believe is the most reasonable response to the question including: all hydrogen safety standards, "
            f"codes, or best practices, or other relevant information, not everything requires a standard to be cited. :\n\n"
            f"User Query: {user_query}\n\n"
            f"Context:\n{retrieved_context}\n\n"
            f"Final Answer:"
        )

        final_answer = llm.complete(final_prompt).text.strip()

        return {
            "final_response": final_answer
        }

    except Exception as e:
        print(f"Error in answer_question: {e}")
        return {
            "final_response": f"An error occurred: {e}"
        }

# Example query
if __name__ == "__main__":
    question = "What hydrogen safety standards are recommended for Oklahoma facilities?"
    result = answer_question(question)

    print("\n=== Final Response ===")
    print(result["final_response"])