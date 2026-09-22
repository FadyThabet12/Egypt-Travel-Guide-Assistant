# Publishing checklist (Phase 5 + deliverables)

1. Run `notebooks/rag_pipeline.ipynb` (Restart & Run All). It writes `backend/data/vector_store/`,
   `docs/evaluation_results.md` and fills the results table in `README.md`.
2. Edit the "Observed on this run" paragraph at the end of section 2.6 (failure analysis) with what *you* saw.
3. Start the backend and frontend, ask real questions, and save 3 screenshots into `docs/screenshots/`:
   `chat-answer.png`, `chat-refusal.png`, `swagger.png` (http://localhost:8000/docs).
4. `cd backend && pytest` must pass.
5. Publish:
   ```bash
   git init
   git add .
   git status            # check: no .venv, no .env, no data/raw/*.pdf
   git commit -m "RAG assistant: notebook, FastAPI backend, frontend"
   git remote add origin https://github.com/<your-username>/rag-assistant-app.git
   git branch -M main
   git push -u origin main
   ```
6. "Verify like a stranger": clone into a fresh folder, follow only the README, fix the README if any step fails.
7. Record the video walkthrough and prepare the live demo (question -> API -> retrieval -> LLM -> cited answer).
