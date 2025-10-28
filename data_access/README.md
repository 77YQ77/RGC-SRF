# Data Access

Tools and scripts to access project datasets (e.g., BIOMEDICA).

## Hybrid Access (Recommended)
- Local cache + cloud streaming for large-scale training
- Saves storage and supports resume

### Quick Start
```bash
# Set token (Hugging Face)
export HF_TOKEN=hf_xxx

cd data_access/hybrid_access
python demo_hybrid_training.py --help
```

### Troubleshooting
- Token/auth issues: re-login `huggingface-cli login`
- Slow speed (CN): set proxy `HTTP(S)_PROXY=http://127.0.0.1:7890`
- Mirror: `export HF_ENDPOINT=https://hf-mirror.com`

See `hybrid_access/README.md` for details.

