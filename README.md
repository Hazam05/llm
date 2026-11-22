# CampusQuery

## Run in VS Code

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501

Upload academic PDFs, build the knowledge base, and ask questions.
The first run downloads the Sentence Transformer and FLAN-T5 models.
