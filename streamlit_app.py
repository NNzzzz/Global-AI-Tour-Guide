import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import WikipediaRetriever
from operator import itemgetter

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
                # --- A. SETUP MODEL ---
                repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
                llm = HuggingFaceEndpoint(
                    repo_id=repo_id,
                    max_new_tokens=512,
                    temperature=0.7,
                    huggingfacehub_api_token=hf_token
                )

                # --- B. SETUP RETRIEVER ---
                retriever = WikipediaRetriever(top_k_results=3, doc_content_chars_max=2000)

                # --- C. SETUP TEMPLATE ---
                template = """
                You are a friendly tour guide specializing in {country}.
                Context:
                {context}
                
                Question: {question}
                
                Answer:
                """
                prompt_template = PromptTemplate.from_template(template)

                # --- D. BUILD CHAIN ---
                chain = (
                    {
                        "context": itemgetter("question") | retriever,
                        "question": itemgetter("question"),
                        "country": itemgetter("country")
                    }
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )

                # --- E. RUN ---
                search_query = f"{prompt} in {country}" if country != "Worldwide" else prompt
                
                response = chain.invoke({
                    "question": search_query, 
                    "country": country
                })
                
                st.markdown(response)

                # ✅✅✅ THIS IS THE LINE YOU WERE MISSING ✅✅✅
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:

                st.error(f"Error: {e}")
