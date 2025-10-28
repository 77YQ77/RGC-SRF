# Hybrid Access (BIOMEDICA)

Local-first + cloud streaming without full downloads. Designed for large-scale medical VLM training.

## Quick Start
```bash
# Requires HF token
export HF_TOKEN=hf_xxx
cd data_access/hybrid_access
python demo_hybrid_training.py --help
```

## Outputs
- Models: `models/best_model_e*.pt`
- Logs: `logs/` (TensorBoard)

## Notes
- Ensure dataset terms accepted on Hugging Face
- Stable network recommended for streaming
- Proxy (CN): `HTTP(S)_PROXY=http://127.0.0.1:7890`

See `demo_hybrid_training.py` for parameters and inline notes.
