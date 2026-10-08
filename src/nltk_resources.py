"""Starter-provided NLTK resource setup; no downloads happen on app startup.

From the repository root:
    python -m src.nltk_resources --download
    python -m src.nltk_resources --download --include-pos
    python -m src.nltk_resources

Resources are stored in the ignored repository-local .nltk_data directory.
Standard NLTK locations, including NLTK_DATA, remain usable for offline setup.
"""
import argparse
from functools import lru_cache
from pathlib import Path

import nltk
from nltk.corpus import stopwords

LOCAL_DATA_DIR = Path(__file__).resolve().parents[1] / ".nltk_data"
CORE_RESOURCES = ("stopwords", "wordnet")
POS_RESOURCE = "averaged_perceptron_tagger_eng"
RESOURCE_PATHS = {
    "stopwords": ("corpora/stopwords", "corpora/stopwords.zip"),
    "wordnet": ("corpora/wordnet", "corpora/wordnet.zip"),
    POS_RESOURCE: (f"taggers/{POS_RESOURCE}",),
}


class NLTKResourceError(RuntimeError):
    """An installed data resource is missing or cannot be read."""


def configure_resource_path() -> None:
    """Prefer local workshop data without changing environment variables."""
    local_path = str(LOCAL_DATA_DIR)
    if local_path not in nltk.data.path:
        nltk.data.path.insert(0, local_path)


def resource_available(name: str) -> bool:
    """Verify directories or ZIP corpora without downloading anything."""
    configure_resource_path()
    for path in RESOURCE_PATHS[name]:
        try:
            nltk.data.find(path)
            return True
        except LookupError:
            continue
    return False


def require_resource(name: str) -> None:
    """Give callers an actionable classroom setup message."""
    if not resource_available(name):
        optional_flag = " --include-pos" if name == POS_RESOURCE else ""
        raise NLTKResourceError(
            f"Missing NLTK resource '{name}'. From the repository root run: "
            f"python -m src.nltk_resources --download{optional_flag}. "
            "For an offline classroom, copy the prepared .nltk_data folder "
            "into the repository. See workshop/SETUP_CHECKLIST.md."
        )


def missing_resources(include_pos: bool = False) -> list[str]:
    """Report missing core resources and, optionally, the POS tagger."""
    names = CORE_RESOURCES + ((POS_RESOURCE,) if include_pos else ())
    return [name for name in names if not resource_available(name)]


@lru_cache(maxsize=1)
def english_stopwords() -> frozenset[str]:
    """Load the provided English set once; students implement filtering only."""
    require_resource("stopwords")
    try:
        return frozenset(stopwords.words("english"))
    except LookupError as error:
        raise NLTKResourceError(
            "The English stopwords resource cannot be read. Run "
            "python -m src.nltk_resources --download after repairing or "
            "replacing the incomplete .nltk_data resource folder."
        ) from error


def setup_resources(include_pos: bool = False) -> None:
    """Download only missing resources, exclusively during explicit setup."""
    missing = missing_resources(include_pos)
    if missing:
        LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in missing:
        print(f"Installing {name}...")
        installed = nltk.download(
            name, download_dir=str(LOCAL_DATA_DIR), quiet=True, raise_on_error=True
        )
        if not installed:
            raise NLTKResourceError(f"Installation failed for {name}.")
    english_stopwords.cache_clear()
    for name in CORE_RESOURCES + ((POS_RESOURCE,) if include_pos else ()):
        require_resource(name)


def main() -> int:
    """One classroom command for setup or offline verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Install missing data once.")
    parser.add_argument("--include-pos", action="store_true", help="Include the optional POS demo.")
    args = parser.parse_args()
    try:
        if args.download:
            setup_resources(args.include_pos)
        missing = missing_resources(args.include_pos)
        if missing:
            require_resource(missing[0])
        print("NLTK resources ready. No startup downloads are needed.")
        return 0
    except (NLTKResourceError, OSError, ValueError) as error:
        print(f"Setup could not finish: {error}")
        print("Check internet/proxy access, folder permissions, or use a prepared offline copy.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
