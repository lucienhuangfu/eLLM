# Multimodal Models

This page tracks the extra work needed when a model accepts more than text.

## Include the following

- Vision or audio encoder integration
- Input preprocessing and batching
- Special tokens or prompt formatting
- Tests for text-only and multimodal paths

## Notes

Keep multimodal-specific logic separate from the base text model path whenever
possible so the implementation stays easier to review.
