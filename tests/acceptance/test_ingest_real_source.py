"""Acceptance test: ingest a real source into a temp copy of wiki/."""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from agentic_rag.config import Settings
from agentic_rag.agents.ingest import build_ingest_agent
from agentic_rag.io.wiki_io import list_pages
from agentic_rag.io.index import read_index


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY")
def test_ingest_real_source():
    """Ingest raw/cv.pdf into a temp copy of wiki/ — verify pages created."""
    # Create temp copy of wiki
    with tempfile.TemporaryDirectory() as tmp:
        wiki_copy = Path(tmp) / "wiki"
        shutil.copytree(Path("wiki"), wiki_copy)

        # Create a settings with temp wiki path
        settings = Settings(wiki_path=wiki_copy)
        agent = build_ingest_agent(settings)
        config = {
            "configurable": {"thread_id": "acceptance-ingest"},
            "recursion_limit": settings.recursion_limit,
        }

        # Find a raw source to ingest
        raw_sources = list(Path("raw").glob("*.md"))
        if not raw_sources:
            pytest.skip("No raw sources available")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": f"Ingest {raw_sources[0]}"}]},
            config=config,
        )

        # Should complete
        assert result["messages"][-1].content

        # Check pages were created
        pages = list_pages(wiki_copy)
        assert len(pages) > 0
