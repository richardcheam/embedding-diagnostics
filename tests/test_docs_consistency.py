"""Documentation invariants, enforced so they cannot silently rot.

These exist because the failure they guard against already happened. A decision
heuristic was withdrawn before any experiment ran, and then reappeared in the
results section described as "pre-registered" and "applied as written" -- in
seven files, across a fortnight, with a passing test suite the whole time.
Prose is not covered by unit tests unless someone writes them, so here they are.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PROSE_GLOBS = ("*.md", "report/**/*.tex", "docs/**/*.md")

# Files whose job is to record the withdrawal, or to narrate history. They may
# name the withdrawn rule; everything else may not present it as current.
HISTORY_FILES = {
    "docs/development-log.md",
    "report/sections/provenance.tex",
    "report/sections/results.tex",
    "report/sections/experiments.tex",
    "AGENTS.md",
    "docs/primer.md",
    "README.md",
    "docs/STATUS.md",
}

# Phrases that would mean the withdrawn rule is being presented as the protocol.
REINSTATEMENT_PHRASES = (
    "pre-registered claim rule",
    "preregistered claim rule",
    "the claim rule was fixed in advance",
    "applied as written",
)

# Verdict words the pre-registration explicitly prohibited attaching to results.
# Checked only in the report, where a reader takes prose as a claim.
PROHIBITED_VERDICTS = ("statistically significant", "proves that", "demonstrates conclusively")


def prose_files():
    seen = []
    for pattern in PROSE_GLOBS:
        for path in ROOT.glob(pattern):
            if ".git" in path.parts or "node_modules" in path.parts:
                continue
            if path.is_file():
                seen.append(path)
    return sorted(set(seen))


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def test_there_are_prose_files_to_check():
    """A glob that silently matches nothing would make every test below pass."""
    assert len(prose_files()) >= 5


@pytest.mark.parametrize("phrase", REINSTATEMENT_PHRASES)
def test_the_withdrawn_heuristic_is_not_presented_as_the_protocol(phrase):
    """The exact regression: `96184e2` withdrew the 2xSD/binomial rule before any
    Phase-A data existed, and `5d1e47f` reinstated it as 'pre-registered'."""
    offenders = []
    for path in prose_files():
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if phrase not in text:
            continue
        # Allowed only where the surrounding text marks it as an error or history.
        context_ok = any(
            marker in text
            for marker in ("withdrawn", "was false", "earlier version", "mistakenly", "residue")
        )
        if not (relative(path) in HISTORY_FILES and context_ok):
            offenders.append(relative(path))
    assert not offenders, (
        f"{phrase!r} presented as current protocol in: {offenders}. "
        "The rule in force is paired Student-t intervals; see report/sections/provenance.tex."
    )


def test_the_report_names_the_protocol_actually_in_force():
    """A negative check alone would pass on an empty document."""
    text = (ROOT / "report" / "sections" / "experiments.tex").read_text()
    assert "Student-t" in text
    assert "No verdicts" in text or "no verdict" in text.lower()


def test_the_provenance_section_exists_and_is_included():
    """It is the single place separating confirmatory from exploratory work."""
    section = ROOT / "report" / "sections" / "provenance.tex"
    assert section.exists()
    body = section.read_text()
    for required in ("Fixed before", "withdrawn", "exploratory"):
        assert required.lower() in body.lower(), required
    assert "input{sections/provenance}" in (ROOT / "report" / "main.tex").read_text()


@pytest.mark.parametrize("phrase", PROHIBITED_VERDICTS)
def test_the_report_avoids_verdict_language(phrase):
    """The pre-registration prohibits labelling results significant or proven."""
    offenders = [
        relative(path)
        for path in (ROOT / "report" / "sections").glob("*.tex")
        if phrase in path.read_text(encoding="utf-8", errors="ignore").lower()
    ]
    assert not offenders, f"{phrase!r} appears in {offenders}"


def test_the_probe_defect_is_disclosed_wherever_probe_numbers_are_quoted():
    """Unstandardized probe endpoints are provisional until recomputed. Any
    document quoting them must say so, or a reader will take them as settled."""
    for name in ("README.md", "docs/STATUS.md", "docs/primer.md"):
        text = (ROOT / name).read_text(encoding="utf-8", errors="ignore").lower()
        if "unscaled" not in text and "unstandardized" not in text:
            continue
        # Any of these constitutes disclosure. The list is deliberately broad
        # about WORDING and strict about PRESENCE: the requirement is that a
        # reader meeting a probe number also meets the caveat, not that every
        # document phrases it identically.
        assert any(
            marker in text
            for marker in (
                "under recomputation",
                "under revision",
                "provisional",
                "withdrawn",
                "being recomputed",
            )
        ), f"{name} quotes probe numbers without disclosing the defect"


def test_condition_count_is_stated_consistently():
    """AGENTS.md said 'all four conditions' long after the grid became seven."""
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8", errors="ignore").lower()
    assert "all four conditions" not in text
    assert "four conditions run through" not in text
