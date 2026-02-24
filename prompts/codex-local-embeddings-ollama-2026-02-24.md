Task: Implement local embeddings provider in Hayfind using Ollama.

Context:
- Repo: ~/projects/hayfind
- OpenAI provider + fallback already added.
- Ollama is installed/running on gateway machine and model `nomic-embed-text` is pulled.
- We need `HAYFIND_EMBED_PROVIDER=local` to work now.

Requirements:
1) Implement local provider in `src/hayfind/embeddings.py`:
   - Provider key: `local`
   - Backend: Ollama HTTP API
   - Endpoint default: `http://127.0.0.1:11434/api/embeddings`
   - Env vars:
     - `HAYFIND_LOCAL_EMBED_URL` (default above)
     - `HAYFIND_LOCAL_EMBED_MODEL` (default `nomic-embed-text`)
   - Should support batch embedding (`embed_documents`) efficiently.
     - If endpoint only supports one prompt at a time, loop in batches but keep behavior clear.
2) Keep output normalization consistent with existing providers: `list[list[float]]` for docs and `list[float]` for query.
3) Keep existing provider/fallback behavior intact:
   - `HAYFIND_EMBED_PROVIDER=local` should function
   - Fallback can still route to gemini/openai when configured
4) Add tests for:
   - local provider response parsing
   - get_embedder with local provider
5) Update README:
   - local embedding env vars + example
   - mention Ollama install + model pull quick commands
6) Run quality gates in repo venv:
   - `ruff check .`
   - `pytest -q`
7) Commit and push with clear message.

Constraints:
- Minimal, focused changes.
- No breaking changes to existing Gemini/OpenAI flows.
- Do not include private paths/secrets in docs.

Output back:
- Files changed
- Test/lint results
- Commit hash
