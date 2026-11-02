# Initializing the repo

1. Download `gemma-4-E2B-it.litertlm` from https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/tree/main and place it in the repo root
2. Create the Python venv: `python3 -m venv .venv`
3. Activate the Python venv: `source .venv/bin/activate`
4. Install the required package: `pip3 install litert_lm`

You can start indexing for example your `Downloads` folder: `python3 inlex.py ~/Downloads`.

After indexing is done you can start your server: `python3 inlex.py ~/Downloads --serve`.
