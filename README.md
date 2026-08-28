# Missing-Aware Latent Recovery

Stage-1 research implementation for multivariate long-term forecasting when the input history contains random variable-wise missing blocks.

The missing-aware front-end recovers damaged patch latents and then reuses PatchTST positional encoding, channel-independent Transformer encoder, and forecast head. ETTh1 and Weather are the first supported datasets.

## Development

```powershell
python -m pip install -e ".[dev]"
python -m pytest -v
```

Generated datasets, masks, prototype banks, checkpoints, and experiment runs are intentionally excluded from Git.

