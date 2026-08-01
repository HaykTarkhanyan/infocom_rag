# Windows + Hugging Face + Armenian: three environment traps

All three bite immediately and have nothing to do with the model.

## 1. `huggingface_hub` dies on symlinks without admin

```
OSError: [WinError 1314] A required privilege is not held by the client:
  '..\\..\\blobs\\4eca68d8...' -> 'C:\\hf_models\\hub\\models--...\\snapshots\\...'
```

The HF cache symlinks blobs into snapshot directories, which needs Developer Mode
or admin rights on Windows. Fix -- copy instead, costs disk but always works:

```python
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
```

## 2. `HF_HOME` must be set before the import, not after

`huggingface_hub` and `transformers` read `HF_HOME` at **import time**. Setting it
afterwards is silently ignored and the download lands in the default cache.

```python
load_dotenv()                     # must come first
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
from transformers import AutoTokenizer   # only now
```

This forces the imports below the dotenv call, which linters flag as E402. That
is correct here; the ordering is load-bearing.

## 3. `PYTHONIOENCODING=utf-8` for any script printing Armenian

Without it, Windows uses `cp1252` for stdout and any Armenian character raises:

```
UnicodeEncodeError: 'charmap' codec can't encode characters in position 29-34
```

This hits `print()` only -- file writes are fine as long as every `open()` and
`FileHandler` passes `encoding="utf-8"` explicitly, which they must.

```bash
PYTHONIOENCODING=utf-8 python src/whatever.py
```
