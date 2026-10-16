# Enterprise Knowledge Assistant: Small RAG Project for Interview

A deliberately small, explainable Retrieval-Augmented Generation (RAG) app. It uses Gemini to create embeddings and answer questions, ChromaDB as a persistent local vector store, and Streamlit for a simple UI.

## What the project demonstrates

- Ingest PDF, TXT, and Markdown documents.
- Extract document text and split it into overlapping chunks.
- Generate document embeddings with Gemini `gemini-embedding-001` or `gemini-embedding-2`.
- Persist vectors, chunk text, and source metadata in ChromaDB.
- Embed a user's question and retrieve top-K relevant chunks using cosine distance.
- Send the question plus retrieved passages to Gemini Flash.
- Ask the model to answer only from retrieved evidence and cite source labels such as `[S1]`.
- Show the retrieved chunks and similarity values so retrieval can be inspected instead of treating the model as a black box.

## Architecture

```text
PDF / TXT / MD
      |
      v
Text extraction --> Chunking (overlap)
      |                    |
      |                    v
      |             Gemini embedding API
      |                    |
      |                    v
      |              ChromaDB (local)
      |
      +-------------------------- indexing time

User question --> Gemini query embedding --> ChromaDB top-K retrieval
                                               |
                                               v
                                 Question + retrieved chunks
                                               |
                                               v
                                      Gemini Flash model
                                               |
                                               v
                                  Answer with source labels
```

## Requirements

- Python 3.10 or newer recommended.
- A Gemini API key with access to the configured embedding and generation models.
- Internet access for Gemini API calls.

## Setup on Windows PowerShell

Open PowerShell in this project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Paste your key into `.env` by replacing `replace_with_your_key`:

```dotenv
GEMINI_API_KEY=your_actual_key_here
GEMINI_CHAT_MODEL=gemini-3.8-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

Do not add quotes unless your shell or environment manager requires them. Never commit `.env` or paste the key into chat, screenshots, GitHub, or code.

Run the app:

```powershell
python -m streamlit run app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`.

## Setup on macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
# Edit .env and paste your Gemini API key locally.
python -m streamlit run app.py
```

## Demo flow

1. Start the app.
2. Leave **Include the built-in fictional company handbook** selected.
3. Select **Build / Rebuild Knowledge Index**.
4. Ask: `What should an engineer do after a failed production deployment?`
5. Observe the generated answer and open the evidence expanders. The citations `[S1]`, `[S2]` refer to the passages supplied to the LLM.
6. Try an unrelated question such as `What is the company's parental leave policy?`. The app should either say the documents do not contain enough information or make the lack of supporting evidence clear.
7. Upload a text-based PDF, `.txt`, or `.md` file and rebuild the index to demonstrate custom data.

**Note:** Every rebuild replaces the current local collection with the currently selected sample and uploaded files. The PDF reader extracts embedded text; OCR for image-only scanned documents is not included.

## What to say in the interview

> I built a small RAG knowledge assistant using Streamlit, Gemini, and ChromaDB. During ingestion, it extracts text from source documents, splits the text into overlapping chunks, and creates embeddings with Gemini's embedding model. Those vectors, along with the source metadata and chunk text, are persisted in a local ChromaDB collection. At query time, the user's question is embedded, the vector store retrieves the top-K chunks by cosine similarity, and the application sends the question and retrieved evidence to Gemini Flash. The prompt asks the model to answer only from those passages and cite their source labels. I also display the retrieved chunks so I can inspect whether an incorrect answer came from retrieval or generation.

## Why each component?

- **Gemini embedding model:** Converts text and questions into vectors for semantic retrieval.
- **Embedding model compatibility:** The engine uses `task_type` with `gemini-embedding-001`; with `gemini-embedding-2`, it uses separate `Content` objects and task prefixes so each chunk receives its own vector.
- **ChromaDB:** Provides a local persistent vector index and query API without requiring a separately managed database server.
- **Gemini Flash:** Generates the natural-language response from the retrieved context.
- **Streamlit:** Makes the pipeline easy to demo and inspect without building a separate frontend.
- **Local vector store:** Keeps this interview project easy to run. It is not a claim that a local index is the right production architecture for every workload.

## Known limitations (be transparent)

- No login, tenant isolation, or document-level access-control implementation. Do not use with confidential documents in the demo.
- No OCR for scanned PDFs or extraction of complex tables and layouts.
- No automated change detection, incremental index updates, or production deletion workflow.
- No reranker or hybrid keyword/vector search.
- No formal benchmark for retrieval precision, answer accuracy, latency, or cost.
- Local ChromaDB persistence is suitable for a demo, not automatically suitable for a distributed production service.
- Prompt instructions reduce risk but do not prove the answer is correct or eliminate prompt injection.

## Useful next improvements

1. Add a small evaluation set with expected source chunks and answerable/unanswerable questions.
2. Add document IDs, versions, incremental updates, and deletion handling.
3. Add tenant-aware metadata filters and authentication before retrieval.
4. Add hybrid keyword + vector retrieval or a reranker if evaluation shows it helps.
5. Log latency per stage and estimate embedding / generation costs.

## Official references

- Gemini embeddings: https://ai.google.dev/gemini-api/docs/embeddings
- Gemini generate content: https://ai.google.dev/api/generate-content
- Google Gen AI Python SDK: https://googleapis.github.io/python-genai/
- Streamlit file uploader: https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
