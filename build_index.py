"""
build_index.py
One-time script to pre-build the FAISS index from catalog.json.
Run this LOCALLY before deploying to Render so the index is
committed to the repo and loaded instantly at cold start.

Usage:
    python build_index.py

Output:
    data/faiss.index
    data/faiss_meta.pkl
"""

import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    # Verify catalog exists
    catalog_path = "data/catalog.json"
    if not os.path.exists(catalog_path):
        logger.error(f"Catalog not found at {catalog_path}")
        logger.error("Place your catalog.json in the data/ directory and re-run.")
        sys.exit(1)

    from catalog import load_catalog, build_catalog_index
    from retriever import SHLRetriever

    logger.info("Loading catalog...")
    catalog = load_catalog(catalog_path)
    name_index = build_catalog_index(catalog)
    logger.info(f"Loaded {len(catalog)} assessments.")

    logger.info("Building FAISS index")
    retriever = SHLRetriever()
    retriever.build_index(
        catalog=catalog,
        name_index=name_index,
        force_rebuild=True,   # always rebuild when running this script
    )

    logger.info("Done. Files written:")
    logger.info("  data/faiss.index")
    logger.info("  data/faiss_meta.pkl")


if __name__ == "__main__":
    main()
