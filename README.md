# 🌍 Worldwide AI Tour Guide

> A smart, context-aware travel assistant powered by RAG (Retrieval-Augmented Generation) and Mistral-7B.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-v0.2-green)

## 📖 Overview
The **Worldwide AI Tour Guide** is a serverless application that acts as a personalized local guide for any country on Earth. Unlike standard chatbots that hallucinate facts, this app uses **RAG** to fetch real-time data from Wikipedia, ensuring accurate responses about history, culture, and logistics.

The system is built on **Streamlit** for the frontend and uses **Hugging Face Inference Endpoints** to run the **Mistral-7B** Large Language Model without requiring local GPUs.

## 🚀 Features
* **🌍 Global Knowledge:** Select any country from the sidebar to customize the guide's personality and expertise.
* **🧠 RAG Architecture:** Dynamically retrieves relevant context from Wikipedia to ground the AI's answers in reality.
* **⚡ Serverless Deployment:** Runs entirely on the cloud using Streamlit and Hugging Face API (No local GPU required).
* **💬 Contextual Memory:** Remembers the conversation history for a natural chat experience.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Orchestration:** LangChain
* **LLM:** Mistral-7B-Instruct-v0.2 (via Hugging Face API)
* **Retriever:** Wikipedia API
* **Utilities:** Pycountry

## ⚙️ How it Works

1.  **User Query:** The user asks a question (e.g., "What is the best food in Egypt?").
2.  **Context Injection:** The app identifies the selected country and appends it to the search query.
3.  **Retrieval:** The `WikipediaRetriever` fetches the top 3 most relevant articles about that topic and country.
4.  **Augmentation:** The retrieved text + the user's question are sent to the LLM (Mistral).
5.  **Generation:** Mistral generates a friendly, fact-based response.

## 📦 Installation & Local Setup

If you want to run this locally instead of on the cloud:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/My-AI-Tour-Guide.git](https://github.com/your-username/My-AI-Tour-Guide.git)
    cd My-AI-Tour-Guide
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run streamlit_app.py
    ```

4.  **Authentication:**
    The app requires a **Hugging Face Token**. You can enter it in the sidebar or save it in a `.streamlit/secrets.toml` file:
    ```toml
    HF_TOKEN = "your_token_here"
    ```

## 📂 Project Structure
```text
My-AI-Tour-Guide/
├── streamlit_app.py       # Main application code (Serverless RAG)
├── requirements.txt       # List of Python dependencies
└── README.md              # Project documentation
