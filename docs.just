build:
    uv run --no-default-groups --extra cli --with-requirements docs/requirements.txt sphinx-build -E -a -b html -W --keep-going docs docs/_build/html

open: build
    uv run --no-sync python -c 'import pathlib, webbrowser; webbrowser.open(pathlib.Path("docs/_build/html/index.html").resolve().as_uri())'
