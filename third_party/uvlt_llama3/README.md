# UVLT Llama3 migration package

This directory contains the minimal Llama3 source copied from the user's UVLT repository:

- `llama/model.py`
- `llama/tokenizer.py`
- `llama/generation_feat.py`
- `llama/__init__.py`

The copied source is kept as a third-party dependency. The wsovvis-facing feature builder does **not** use the old UVLT `corr_feats` as final features, because the old generation-time feature capture is shifted by one token.

The corrected extraction policy is implemented in:

```text
tools/build_lvvis_llama3_text_bank.py
```

Corrected policy:

1. Generate assistant token ids with Llama3.
2. Re-run a full forward pass on `prompt_tokens + generated_tokens`.
3. Slice hidden states exactly at the generated assistant-token span.
4. Mean-pool token hidden states per response.
5. Mean-pool prompt/repeat views per class and L2-normalize.

Prompt profiles are stored in:

```text
third_party/uvlt_llama3/prompts/prompt_profiles.json
```
