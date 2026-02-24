Task: Add OpenAI embeddings support + provider fallback for Hayfind indexing reliability.

Context:
- Repo: ~/projects/hayfind
- Current embedding path works with google-genai (gemini-embedding-001) but full indexing can hit 429 quota.
- We want pluggable embedding providers and a practical fallback path.

Requirements:
1) Add provider selection via env vars/config:
   - HAYFIND_EMBED_PROVIDER = gemini|openai|local (local can be TODO/not implemented with clear error)
   - HAYFIND_EMBED_FALLBACK_PROVIDER (optional): openai or gemini
2) Add OpenAI embeddings backend:
   - OPENAI_API_KEY
   - HAYFIND_OPENAI_EMBED_MODEL default: text-embedding-3-small
   - Use batched requests and normalize output to same vector list format used currently.
3) Fallback behavior:
   - If primary provider errors with retry-exhausted rate/quota errors, and fallback provider configured, continue with fallback provider for remaining batches.
   - Log/surface provider switch clearly.
4) Keep existing Gemini behavior unchanged when provider=gemini and no fallback.
5) Tests:
   - Unit tests for provider selection and response normalization
   - Unit test for fallback switch logic (mock failures then success)
6) Docs:
   - Update README env var section + examples.
   - Add a short migration note: OpenAI API key required; ChatGPT subscription alone is not API access.
7) Quality gates:
   - ruff check .
   - pytest -q

Implementation constraints:
- Minimal churn; don’t rewrite unrelated modules.
- Keep public-safe docs (no personal paths/secrets).
- Commit your changes with a clear commit message.

Output back:
- Summary of files changed
- Test/lint results
- Any caveats