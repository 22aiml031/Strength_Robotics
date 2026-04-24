# Research Paper RAG Agent

An AI-powered assistant designed to simplify research by allowing users to chat directly with academic papers. Built with a focus on privacy and speed using local LLMs.

---

### 🚀 Quick Start (2-minute setup)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Pull Local Models (Ollama)**
   ```bash
   ollama pull all-minilm
   ollama pull gemma3:4b
   ```

3. **Configure & Run**
   - Rename `.env.example` to `.env` and add your Supabase keys.
   - Run: `python ingest.py --paper all`
   - Start: `streamlit run app.py`

---

### 📝 Project Overview

This application uses **Retrieval-Augmented Generation (RAG)** to transform dense academic PDFs into interactive knowledge bases. 

- **What it does:** Indexes research papers into a vector database and uses a local LLM to answer questions.
- **Why it is useful:** Eliminates the need to manually skim hundreds of pages to find specific data or methodologies.
- **Example Use-case:** A researcher asking, *"What were the specific error rates for the XGBoost model in the imputation study?"* and getting an instant, cited answer.

---

### 💡 Why This Project?

*   **Problem:** Reading and synthesizing information from multiple research papers is time-consuming and error-prone.
*   **Solution:** By using RAG, the agent "reads" the papers for you and provides answers grounded strictly in the source text.
*   **Impact:** Faster insights, better data extraction, and a more intuitive way to interact with scientific literature.

---

### ✨ Features

*   **Privacy-First:** All embeddings and LLM inference run locally via Ollama.
*   **High Precision:** Vector search ensures only relevant sections are used for answers.
*   **Citations included:** Every answer provides page and section references for easy verification.
*   **Multi-Paper Context:** Supports separate indexing and querying for multiple studies.

---

### 📂 Project Layout

| File / Folder | Description |
| :--- | :--- |
| **`app.py`** | The main Streamlit dashboard for asking questions. |
| **`ingest.py`** | Script to process PDFs, create chunks, and store embeddings. |
| **`src/`** | Core RAG logic, including PDF parsing and database helpers. |
| **`sql/`** | Database schema and `pgvector` search functions. |
| **`data/`** | Storage for the research PDFs to be indexed. |

---

### 🔍 Demo / Example Queries

Try asking the agent these questions:
*   *"What tracker performed best in the barbell trajectory study and why?"*
*   *"What kinematic variables are most critical for assessing injury risk?"*
*   *"How does the proposed imputation method handle missing longitudinal data?"*

---

### 🛠 Tech Stack

*   **Python:** Core logic and processing.
*   **Streamlit:** Clean, interactive user interface.
*   **Supabase (pgvector):** High-performance vector storage and retrieval.
*   **Ollama:** Local hosting for LLMs (`Gemma`) and embeddings.
*   **Sentence Transformers:** State-of-the-art text vectorization.

---

### ⚙️ Setup (Full Instructions)

#### 1. Database Setup (Supabase)
- Create a new project at [Supabase](https://supabase.com).
- Open the SQL Editor and run the contents of `sql/schema.sql` to enable vectors and create the required tables.

#### 2. Environment Variables
Copy `.env.example` to `.env` and replace the placeholders:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_role_key
```

#### 3. Local Model Installation
Ensure [Ollama](https://ollama.com) is installed and running, then pull the models:
```bash
ollama pull all-minilm
ollama pull gemma3:4b
```

#### 4. Data Ingestion
Process the provided PDFs and store them in the vector database:
```bash
python ingest.py --paper all
```

---

### ⚠️ Limitations

*   **PDF Formatting:** Accuracy depends on the structure and text extractability of the PDF.
*   **Local Hardware:** Performance is tied to your machine's ability to run Ollama.
*   **Context Scope:** The agent can only answer based on the papers currently indexed.

---

### 👤 Author & Notes

Developed to demonstrate the practical application of RAG in specialized domains like Exercise Science and Data Imputation.

---
