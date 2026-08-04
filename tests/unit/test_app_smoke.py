"""AppTest smoke tests for the restructured multi-page app (Task 4).

Renders ``frontend/app.py`` (the ``st.navigation`` entry point) and each
existing ``frontend/app_pages/*.py`` page under Streamlit's ``AppTest`` with a
fake ``OPENAI_API_KEY`` env. Pages render only — no chat inputs are driven, so
the LLM is never contacted (agent builds are API-free).

T4 ships only ``query.py``; ingest/lint/fix land in T5/T6. app.py builds its
navigation from the page files that exist, so the entry point renders with
just query.py and picks up the remaining pages automatically as their files
land — the T4 smoke test requires only app.py + query.py to render, and never
asserts on page files it doesn't ship. T5 adds a dedicated ingest render case
(chat input + raw picker, render-only).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
APP_PY = FRONTEND / "app.py"
QUERY_PAGE = FRONTEND / "app_pages" / "query.py"
INGEST_PAGE = FRONTEND / "app_pages" / "ingest.py"

# Every page shipped so far renders standalone (grows as T5/T6 land).
EXISTING_PAGES = sorted(FRONTEND.glob("app_pages/*.py"))


@pytest.fixture
def fake_api_key(monkeypatch):
    """Smoke pages never hit the LLM; the fake key only satisfies Settings()."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")


def _render(page: Path) -> AppTest:
    """Run one page under AppTest and assert it produced no script exceptions."""
    at = AppTest.from_file(str(page))
    at.run()
    assert len(at.exception) == 0, [
        exc.message for exc in at.exception
    ]
    return at


class TestAppEntryPoint:
    def test_entry_point_renders_default_page(self, fake_api_key):
        """app.py renders the nav shell and the default (query) page inline."""
        at = _render(APP_PY)
        assert [t.value for t in at.title] == ["Wiki Q&A"]
        # The query chat affordance is present (render-only — nothing submitted).
        assert len(at.chat_input) == 1
        # The sidebar thread manager rendered.
        assert any("New chat" in b.label for b in at.sidebar.button)


class TestQueryPage:
    def test_query_page_renders_standalone(self, fake_api_key):
        """query.py works standalone under AppTest.from_file (repo-root bootstrap)."""
        at = _render(QUERY_PAGE)
        assert [t.value for t in at.title] == ["Wiki Q&A"]
        assert "Streaming chat over the wiki query agent" in [c.value for c in at.caption]
        assert len(at.chat_input) == 1
        assert any("New chat" in b.label for b in at.sidebar.button)


class TestIngestPage:
    def test_ingest_page_renders_standalone(self, fake_api_key):
        """ingest.py renders standalone — chat input + raw picker, render-only.

        No chat input is driven, so the ingest agent is never built and the LLM
        is never contacted. The raw-source picker reads ``raw/`` ONLY; when the
        (gitignored) ``raw/`` dir holds files, the ``Source in raw/`` selectbox
        must render.
        """
        at = _render(INGEST_PAGE)
        assert [t.value for t in at.title] == ["Ingest"]
        assert len(at.chat_input) == 1
        assert any("New chat" in b.label for b in at.sidebar.button)
        raw_dir = Path.cwd() / "raw"
        if raw_dir.is_dir() and any(p.is_file() for p in raw_dir.rglob("*")):
            assert any(s.label == "Source in raw/" for s in at.selectbox)


class TestEveryExistingPage:
    def test_every_shipped_page_renders(self, fake_api_key):
        """Each frontend/app_pages/*.py currently shipped renders without exceptions.

        Self-maintaining: T5/T6 pages are added to this loop automatically once
        their files land; the loop never requires a page that does not exist.
        """
        assert EXISTING_PAGES, "no app_pages shipped yet"
        for page in EXISTING_PAGES:
            _render(page)
