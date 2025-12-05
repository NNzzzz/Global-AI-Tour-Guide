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
    except:
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
        with st.spinner("Searching Global Database..."):
            try:
                # --- A. RETRIEVE CONTEXT (Wikipedia) ---
                # We use LangChain just for the Wikipedia search (it's good at that)
                retriever = WikipediaRetriever(top_k_results=3, doc_content_chars_max=2000)
                
                # Construct query
                search_query = f"{prompt} in {country}" if country != "Worldwide" else prompt
                
                # Fetch docs
                docs = retriever.invoke(search_query)
                context_text = "\n\n".join([doc.page_content for doc in docs])

                # --- B. PREPARE PROMPT ---
                # We build the chat message manually
                system_prompt = f"""You are a friendly, enthusiastic tour guide specializing in {country}.
                Use the following context to answer the user's question. 
                If the answer isn't in the context, use your general knowledge.
                
                Context:
                {context_text}"""

                # --- C. CALL MODEL (Using InferenceClient) ---
                # This uses the "Chat" API which is supported on Free Tier!
                client = InferenceClient(
                    "mistralai/Mistral-7B-Instruct-v0.3", 
                    token=hf_token
                )

                # Send messages
                response_stream = client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=512,
                    temperature=0.7,
                    stream=False
                )

                # Extract Answer
                answer = response_stream.choices[0].message.content
                
                # Display & Save
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error: {e}")
