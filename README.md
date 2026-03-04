# Simple Local RAG (Python + Streamlit)

A minimal Retrieval-Augmented Generation app with:
- Streamlit UI
- Hugging Face embedding model (`sentence-transformers/all-MiniLM-L6-v2`)
- Local vector database (ChromaDB persisted to `./chroma_db`)
- OpenRouter chat model for answers

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set your API key:

```bash
set OPENROUTER_API_KEY=your_key_here
```

## Run

```bash
streamlit run app.py
```

Then open the Streamlit URL in your browser.

## Usage

1. Upload `.txt`, `.md`, or `.pdf` files in the sidebar.
2. Click **Index Uploaded Files**.
3. Add OpenRouter API key and model in sidebar.
4. Chat in the main panel.
5. View retrieved chunks for the last turn.

## Notes

- First run downloads the embedding model from Hugging Face.
- Vector DB is local and persistent in `./chroma_db`.
- Click **Reset Vector DB** in the sidebar to clear indexed data.
