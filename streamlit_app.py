import streamlit as st
from huggingface_hub import InferenceClient
from langchain_community.retrievers import WikipediaRetriever

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Global Guide", page_icon="🌍")

# ==========================================
# 2. SIDEBAR & SETUP
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Securely get token
    if "HF_TOKEN" in st.secrets:
        hf_token = st.secrets["HF_TOKEN"]
    else:
        hf_token = st.text_input("Hugging Face Token:", type="password")

    st.divider()

    # Country Selector
    try:
        import pycountry
        countries = sorted([c.name for c in pycountry.countries])
    except ImportError:
        # Fallback if pycountry isn't installed
        countries = ["Worldwide", "USA", "France", "Egypt", "Japan", "Italy"]
        
    country = st.selectbox("Destination:", ["Worldwide"] + countries)

# ==========================================
# 3. MAIN APP LOGIC
# ==========================================
st.title("🌍 AI Tour Guide")
st.caption(f"Running on Streamlit Cloud | Focus: **{country}**")

# Initialize History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    
    # 1. Check for Token
    if not hf_token:
        st.error("❌ Missing Hugging Face Token. Add it in Sidebar or Secrets.")
        st.stop()

    # 2. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Generate Answer
    with st.chat_message("assistant"):
        with st.spinner(f"Searching Wikipedia for info about {country}..."):
            try:
                # --- A. RETRIEVE CONTEXT (Wikipedia) ---
                # top_k=5 to get more context
                retriever = WikipediaRetriever(top_k_results=5, doc_content_chars_max=2000)
                
                # specific search query to help Wikipedia find the right country context
                search_query = f"{prompt} {country}" 
                
                docs = retriever.invoke(search_query)
                context_text = "\n\n".join([doc.page_content for doc in docs])

                # DEBUG: Show what Wikipedia found (Click to expand)
                with st.expander("🕵️ View Retrieved Source Text"):
                    st.write(context_text)

                # --- B. PREPARE PROMPT (Your Custom "Warm" Prompt) ---
                system_prompt = f"""
You are a warm, enthusiastic, and knowledgeable tour guide specializing in {country}.
Your goal is to help the traveler have the best experience possible, whether they ask about history, logistics, culture, or hidden gems.

Here is some information retrieved from the guidebook (Wikipedia):
Context:
{context_text}

---
User's Question: {prompt}

Instructions:
1. Answer the question specifically for {country}.
2. Use a friendly, conversational tone (like a helpful local friend).
3. If the retrieved context contains the answer, summarize it clearly.
4. If the context is empty or irrelevant, use your general knowledge to help, but keep it grounded in reality.
5. Avoid technical jargon; speak like a human guide.

Guide's Answer:
"""

                # --- C. CALL MODEL ---
                client = InferenceClient(
                    "HuggingFaceH4/zephyr-7b-beta", 
                    token=hf_token
                )

                # Send messages
                response_stream = client.chat_completion(
                    messages=[
                        {"role": "user", "content": system_prompt} 
                    ],
                    max_tokens=512,
                    temperature=0.5, # 0.3 allows for a friendly tone without being too random
                    stream=False
                )

                # Extract Answer
                answer = response_stream.choices[0].message.content
                
                # Display & Save
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error: {e}")
