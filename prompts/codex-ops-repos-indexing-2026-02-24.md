Task: Finish Hayfind local-embeddings indexing across all operations repos and harden any remaining runtime issues.

Repo: ~/projects/hayfind

Current state/context:
- Local provider support (Ollama) is implemented.
- Auto-separate Chroma path by provider/model is implemented.
- Service is flaky when running giant synchronous /index calls from CLI wrappers; direct HTTP calls are more reliable.
- Goal now is operational reliability + complete indexing coverage for:
  1) /Users/chrisfonte/operations
  2) /Users/chrisfonte/operations-chris-fonte
  3) /Users/chrisfonte/operations-fontastic
  4) /Users/chrisfonte/clawd

Deliverables:
1) Implement robust indexing strategy in code (if needed):
   - avoid giant monolithic blocking calls where possible
   - add/ensure chunked repo indexing behavior
   - better progress/error reporting for long runs
2) Verify local provider config defaults and docs for this workflow.
3) Run smoke verification queries for each repo and capture results.
4) Ensure tests/lint pass.
5) Commit and push changes.

Required checks:
- ruff check .
- pytest -q

Output required in final response:
- What changed (files + why)
- Exact commands used to index all four repos
- Verification search examples + results summary
- Commit hash

Constraints:
- Minimal, focused changes.
- Keep existing Gemini/OpenAI behavior intact.
- Don’t touch unrelated code.
