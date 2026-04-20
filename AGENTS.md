Whenever you need to use Python, especially to run tests, ALWAYS use the `eztaox-env-cpu-ns` conda environment.

Use `/Users/colinburke/miniforge3/envs/eztaox-env-cpu-ns/bin/python` directly for Python commands.

Run tests through that interpreter, for example:
- `/Users/colinburke/miniforge3/envs/eztaox-env-cpu-ns/bin/python -m pytest ...`
- `/Users/colinburke/miniforge3/envs/eztaox-env-cpu-ns/bin/python path/to/script.py`

Do not use the system `python`, `python3`, or bare `pytest` for this repository.

Do not ever create a new environment or venv.
