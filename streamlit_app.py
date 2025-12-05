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
        countries = ["Worldwide", "USA", "France", "Egypt", "Japan", "Italy"]
        
    country = st.selectbox("Destination:", ["Worldwide"] + countries)

# ==========================================
# 3. MAIN APP LOGIC
# ==========================================
st.title("🌍 AI Tour Guide")
st.caption(f"Your friendly neighbourhood tour guide |📍: **{country}**")

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
        with st.spinner(f"Searching info about {country}..."):
            try:
                # --- A. RETRIEVE CONTEXT (Wikipedia) ---
                retriever = WikipediaRetriever(top_k_results=5, doc_content_chars_max=2000)
                search_query = f"{prompt} {country}" 
                docs = retriever.invoke(search_query)
                context_text = "\n\n".join([doc.page_content for doc in docs])

                # (Debug View REMOVED as requested)

                # --- B. PREPARE PROMPT ---
                system_prompt = f"""
You are a warm, enthusiastic tour guide specializing in {country}.

Here is information from the guidebook:
Context:
{context_text}

---
User's Question: {prompt}

Instructions:
1. Answer the question specifically for {country}.
2. KEEP IT SHORT (aim for 3-5 sentences). Do not ramble.
3. Use a friendly, enthusiastic tone with emojis (e.g., 🏛️, 🌊, ✨) to make it lively.
4. If listing recommendations, use bullet points for readability.
5. No technical jargon. Speak simply like a local friend.
6. If the context contains the answer, summarize it. If not, use general knowledge but stay realistic.
7. End with a short, engaging follow-up question to keep the chat going.
8.Never invent facts. If you don't know, suggest a related popular spot in {country} instead

Guide's Answer:
"""

                # --- C. CALL MODEL ---
                client = InferenceClient( "mistralai/Mistral-7B-Instruct-v0.2", 
                    token=hf_token
                )

                response_stream = client.chat_completion(
                    messages=[
                        {"role": "user", "content": system_prompt} 
                    ],
                    max_tokens=512,
                    temperature=0.5, # UPDATED: Set to 0.5
                    stream=False
                )

                # Extract Answer
                answer = response_stream.choices[0].message.content
                # --- CLEANUP FIX ---
                # This removes the glitched tags if they appear
                answer = answer.replace("[/USER]", "").replace("[/ASS]", "").strip()
                
                # Display & Save
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
             

            except Exception as e:
                st.error(f"Error: {e}")





