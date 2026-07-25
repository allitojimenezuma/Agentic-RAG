Interesting finding — `"entitie" + "s"` happens to equal `"entities"` by coincidence, so `upsert_entry` still routes to correct category. But the stored `IndexEntry.type` field is still `"entitie"` — latent bug for Phase 2/3.

Final count: 57 tests pass. 2 blockers (path traversal, type corruption), 2 data fidelity issues. Let me write the acceptance report.

---

```
## Review
- Correct: All 57 unit tests pass (0.94s). All §8 Pydantic models present with correct fields. §7 MarkItDown wrapper handles optional LLM image description. §10 log prefix format correct. Atomic temp+rename writes in wiki_io.py and index_manager.py. markdown_parser regex handles [[Target]] and [[Target|alias]]. slugify handles unicode. agents_md returns file content or sensible default. Index parsing works for both [[link]] and [Title](path) entry styles.

- Fixed: Not edited (review-only instructions).

- Blocker: wiki_io.py:27-28,36,72,79 — Path traversal: no slug validation. `wiki_path / f"{slug}.md"` with slug `../../etc/passwd` resolves outside wiki_path. Defense-in-depth requires validation at io layer, not just Phase 3 middleware.

- Blocker: index_manager.py:96 — `section_name.rstrip("s")` produces `"entitie"` instead of `"entity"` for "## Entities" section. Stored `IndexEntry.type` is wrong. Accidentally mitigated for category routing (since `"entitie"+"s"=="entities"`), but will break Phase 2/3 code comparing `entry.type == "entity"`.

- Note: index_manager.py:110-113 — Source entry format generates `[slug].md` instead of `sources/[slug].md` per §10 spec. Also loses original file extension on round-trip (cv.pdf → cv.md).

- Note: index_manager.py:118-121 — `_format_entry` re-derives display name from slug via `.title()`, losing original casing. [[MLX]] round-trips to [[Mlx]].
```