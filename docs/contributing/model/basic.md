# Basic Model Support

This page explains the minimum path for adding a new model implementation.

## Typical flow

1. Identify the model family and its architecture.
2. Add or reuse the tokenizer and configuration mapping.
3. Wire the model into the loading path.
4. Validate the forward pass and generation path.

## What to document here

- Required config fields
- Minimal code changes
- Common failure modes
- Smoke tests that should pass before merging
