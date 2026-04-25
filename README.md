This repository contains three main parts: research paper analysis, a RAG-based research assistant, and an advanced exercise science coaching agent.

---

## 📄 Research Paper Task

The file **`Research_paper_task.pdf`** contains a detailed analysis of two research papers. The purpose of this task is to understand the research deeply and translate it into practical insights.

### Paper 1: Barbell Trajectory Analysis (Computer Vision)

This paper focuses on analyzing weightlifting performance using computer vision techniques. The system detects and tracks the barbell using object detection and tracking algorithms. It extracts trajectory-based features such as vertical displacement and movement patterns to evaluate lifting performance.

The key idea is to automate coaching by analyzing bar path instead of relying on manual observation. However, the system has limitations such as sensitivity to camera angles, occlusion issues, and lack of 3D depth understanding.

### Paper 2: Data Imputation for Athlete Monitoring

This paper addresses the issue of missing athlete data in performance monitoring systems. It proposes a machine learning-based imputation pipeline using feature selection, clustering, and multiple XGBoost models.

The system predicts missing values based on similarity between athlete profiles. While effective, it depends on data quality and can struggle with noisy or limited datasets.

### Purpose of This Task

This task builds the foundation for the project by:

* Understanding real-world problems in sports science
* Identifying limitations of current approaches
* Translating research into implementable AI systems

---

## 🧠 Task 1: Research Paper RAG Agent

The folder **`Task_1_rag/`** contains a Retrieval-Augmented Generation (RAG) system that allows users to interact with research papers using natural language queries.

### What This Task Does

Instead of manually reading research papers, users can ask questions and receive answers generated directly from the paper content. The system ensures that all responses are grounded in the actual research.

### Papers Used in This Task

The system works on two research papers:

* Imputation-based athlete monitoring paper
* Weightlifting computer vision tracking paper

These papers are stored inside the **`data/`** folder and are processed during ingestion.

### How the System Works

1. PDF papers are processed and converted into text
2. Text is split into smaller chunks
3. Each chunk is converted into embeddings using a local model
4. Embeddings are stored in Supabase (pgvector)
5. When a user asks a question, relevant chunks are retrieved
6. A local LLM (Gemma via Ollama) generates the answer

### Folder Structure Overview

* `app.py` → Streamlit interface for asking questions
* `ingest.py` → Processes PDFs and stores embeddings
* `data/` → Contains research papers
* `sql/` → Database schema and vector search functions
* `src/` → Core logic (embedding, retrieval, processing)
* `requirements.txt` → Dependencies

### Important Note

This task includes its own **README.md inside the Task_1_rag folder**, which provides detailed step-by-step instructions on how to install dependencies, configure the environment, run ingestion, and launch the application.

---

## 🏋️‍♂️ Task 2: Exercise Science Coaching Agent

The folder **`Task_2_rag/`** contains an advanced RAG-based system that generates research-grounded coaching advice for squat performance.

### What This Task Does

This system goes beyond simple question answering. It analyzes biomechanical concepts from multiple research papers and generates structured coaching insights for athletes.

### Key Features

* Multi-paper retrieval and reasoning
* Biomechanics-based analysis
* Structured coaching output (4-paragraph format)
* Focus on performance optimization and injury prevention

### Research Focus

This task uses multiple biomechanics papers covering:

* Trunk angle and bar path
* High-bar vs low-bar squat differences
* Hip and knee joint loading
* Movement efficiency and stability

### How the System Works

1. Multiple research papers are ingested and indexed
2. A diversity-based retrieval system ensures multiple sources are used
3. The model synthesizes insights across papers
4. A structured response is generated including:

   * Problem context
   * Research findings
   * Biomechanical explanation
   * Practical coaching advice

### Folder Structure Overview

* `app.py` → Streamlit coaching interface
* `ingest.py` → Processes and indexes research papers
* `generator.py` → Generates structured coaching responses
* `retriever.py` → Handles multi-paper retrieval
* `data/` → Research paper PDFs
* `requirements.txt` → Dependencies

### Important Note

This task also includes its own **README.md inside the Task_2_rag folder**, which explains how to set up the environment, install models, run ingestion, and launch the coaching application.

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Supabase (pgvector)
* Ollama (local LLMs like Gemma)
* Sentence Transformers
* Machine Learning (XGBoost, clustering)
* Computer Vision concepts

---

## ⚙️ General Setup Overview

Each task has its own setup instructions in its respective README file. In general, the steps include:

* Install dependencies using requirements.txt
* Install and run Ollama locally
* Pull required models (all-minilm, gemma3:4b)
* Configure Supabase credentials
* Run ingestion scripts
* Launch Streamlit application

---

## 📈 Project Outcome

This project demonstrates a complete pipeline:

* Understanding research papers
* Converting them into structured data
* Building AI systems on top of them
* Generating real-world insights

---
