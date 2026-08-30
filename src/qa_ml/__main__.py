"""Module entry point so the pipeline runs as ``python -m qa_ml <command>``."""

from qa_ml.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
