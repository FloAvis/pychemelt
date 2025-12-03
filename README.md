# Install from Source - Development

git clone https://github.com/osvalB/pychemelt.git
cd pychemelt
uv sync --extra dev

# Verify Installation
uv run pytest
uv run build_docs.py


