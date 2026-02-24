# Codex prompt — hayfind: migrate embeddings to google-genai (2026-02-24)

Repo: `~/projects/hayfind` (public GitHub repo chrisfonte/hayfind)

Problem:
- Runtime indexing fails. Server 500 because `google-generativeai` embed_content call returns NotFound for `models/text-embedding-004`.
- Goal: migrate embeddings provider to the new Google Gen AI SDK (`google-genai`) and use the doc-aligned embeddings model.

Context docs (read if helpful):
- Private impact plan: `~/operations-chris-fonte/todo-files/hayfind-google-genai-migration-impact-plan.md`
- Public research: `~/operations/docs/04-professional/research/ai-tools-research/google-gemini/google-genai-sdk-migration-for-embeddings.md`

Hard requirements:
1) Replace `google-generativeai` usage with `google-genai`.
   - Import style: `from google import genai`
   - Create client via `genai.Client()` (env var pickup) or `genai.Client(api_key=...)` when key is read from file.
2) Update embeddings call to use the new SDK method:
   - `client.models.embed_content(model='gemini-embedding-001', contents=...)`
   - Keep model name centralized/configurable (env var or constant).
3) Preserve current API key lookup order:
   - env `GEMINI_API_KEY` or `GOOGLE_API_KEY`
   - else file `~/.credentials/gemini/api_key`
4) Normalize the SDK response into plain Python vectors so callers get `list[list[float]]`.
   - Don’t leak pydantic objects through the rest of the code.
5) Update dependencies in `pyproject.toml` accordingly.
6) Update / add tests:
   - Add a unit test that mocks the embeddings client and verifies we parse to vectors.
   - Avoid requiring real API keys in tests.
7) Update README if needed (remove deprecation warning, update install notes).

Acceptance:
- In a fresh venv: `pip install -e '.[dev]'` works.
- `ruff check .` passes.
- `pytest` passes.
- End-to-end smoke test passes:
  - `hayfind serve` (background)
  - `hayfind index ~/projects/hayfind` succeeds
  - `hayfind search "embed" --repo hayfind` returns results

Make small, clean commits and keep code style consistent.
