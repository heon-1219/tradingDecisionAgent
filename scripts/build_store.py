"""Build data/store/products.db from data/raw/*.xlsx.  Usage: python scripts/build_store.py"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from finagent.etl.build_store import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
