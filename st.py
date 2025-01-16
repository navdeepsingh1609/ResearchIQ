import streamlit as st
import requests

# Configuration: Replace these placeholders with your API keys and endpoints
CORTEX_SEARCH_ENDPOINT = "https://api.cortexsearch.com/v1/search"
CORTEX_API_KEY = "your_cortex_api_key"
MISTRAL_ENDPOINT = "https://api.mistralllm.com/v1/generate"
MISTRAL_API_KEY = "your_mistral_api_key"

def search_documents(query):
    """Fetch relevant documents from Cortex Search."""
    headers = {"Authorization": f"Bearer {CORTEX_API_KEY}"}
    payload = {"query": query, "top_k": 5}  # Fetch top 5 results
    response = requests.post(CORTEX_SEARCH_ENDPOINT, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        st.error("Error retrieving documents from Cortex Search.")
        return []

def generate_summary(context, query):
    """Generate a summary or response using Mistral LLM."""
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
    payload = {
        "context": context,
        "prompt": f"Based on the following context, answer the query: {query}",
        "max_tokens": 300,
    }
    response = requests.post(MISTRAL_ENDPOINT, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("text", "")
    else:
        st.error("Error generating response with Mistral LLM.")
        return ""

# Streamlit UI
st.title("AI-Powered Research Assistant")
st.subheader("Search and summarize information with cutting-edge AI technology.")

# User query input
query = st.text_input("Enter your research query:", placeholder="e.g., Climate change impacts on biodiversity")

if st.button("Search and Summarize"):
    if query.strip():
        with st.spinner("Fetching documents..."):
            documents = search_documents(query)
        
        if documents:
            st.success("Documents retrieved successfully!")
            st.subheader("Top Retrieved Documents:")
            combined_context = ""
            
            for idx, doc in enumerate(documents, start=1):
                st.markdown(f"**{idx}. {doc['title']}**")
                st.write(doc['snippet'])
                combined_context += f"{doc['title']}: {doc['snippet']}\n"
            
            with st.spinner("Generating summary..."):
                summary = generate_summary(combined_context, query)
            
            st.subheader("Generated Summary:")
            st.write(summary)
        else:
            st.warning("No documents found. Please refine your query.")
    else:
        st.warning("Please enter a query to proceed.")
