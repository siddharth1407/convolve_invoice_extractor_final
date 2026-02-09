# 📄 Invoice Extractor Pro

**AI-Powered Invoice Data Extraction using Qwen2-VL + YOLOv8**

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud%20Ready-red)](https://streamlit.io/)

Automatically extract structured data from invoices using Vision-Language Models and Object Detection. Supports multiple languages, OCR fallbacks, and real-time processing with confidence scoring.

---

## 🎯 Features

- ✅ **Multi-Language Support**: English, Hindi, Marathi, Gujarati
- ✅ **Hybrid OCR Engine**: PaddleOCR + EasyOCR with intelligent fallback
- ✅ **Vision-Language Model**: Qwen2-VL-2B-Instruct for text extraction
- ✅ **Object Detection**: YOLOv8 for stamp and signature detection
- ✅ **Confidence Scoring**: Geometric mean-based perplexity scoring
- ✅ **Batch Processing**: Process multiple PDFs and images
- ✅ **Real-Time Live Results**: Stream processing with visual feedback
- ✅ **Annotated Images**: Auto-generated images with detected boxes
- ✅ **Offline Mode**: All models cached, zero internet required
- ✅ **Production Ready**: Docker-compatible, easy deployment

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- 8GB RAM (16GB recommended)
- 5GB disk space

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/invoice-extractor.git
cd invoice_extractor_convolve

# Setup
python -m venv venv
source venv/bin/activate  # macOS/Linux: venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Download models (only if you want to run offline otherwise not needed)
python setup_models.py
```

### Run

**Web UI:**
```bash
streamlit run app.py
```

**CLI Demo:**
```bash
python demo_runner.py
```

---

## 📋 Extracted Fields

| Field | Type | Example | Validation |
|-------|------|---------|-----------|
| `dealer_name` | String | "National Motors" | Non-empty, not brand |
| `model_name` | String | "575 SP PLUS" | Non-empty |
| `horse_power` | Integer | 50 | Range: 20-99 |
| `asset_cost` | Integer | 500000 | Range: 300K-2M |
| `stamp` | Object | `{present: true, bbox: [...]}` | Valid coordinates |
| `signature` | Object | `{present: true, bbox: [...]}` | Valid coordinates |

**Plus:** Confidence score (0.0-1.0) for every extraction

---

## 🏗️ Architecture

```
Invoice (PDF/PNG/JPG)
    ↓
    ├─ OCR Path: Resize → 3 versions → Multi-OCR
    ├─ Vision Path: YOLOv8 detection → Stamps/Signatures
    └─ Merge & Validate → JSON + Annotated Image
```

**Stack:**
- Qwen2-VL-2B-Instruct (VLM)
- PaddleOCR + EasyOCR (OCR)
- YOLOv8 (Detection)
- OpenCV (Preprocessing)
- Streamlit (Web UI)
- PyTorch (Inference)

---

## 📖 Usage

### Web Interface

1. **Upload** (Tab 1): Upload invoices
2. **Process**: Click "Process Documents"
3. **Results** (Tab 2): View extracted fields
4. **Visualize** (Tab 3): See annotated images
5. **Export**: Download as JSON

### Python API

```python
from executable import load_model, process_single_image

model, processor, device = load_model()
result = process_single_image(model, processor, device, "invoice.jpg", "invoice.jpg")

print(f"Dealer: {result['fields']['dealer_name']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Batch Processing

```bash
python demo_runner.py
# Results → sample_output/result.json
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Processing Time | 2-3 sec/doc |
| Model Load Time | ~15 sec (first) |
| Dealer Accuracy | ~94% |
| Model Accuracy | ~93% |
| HP Accuracy | ~95% |
| Cost Accuracy | ~92% |
| Stamp Detection | ~88% |
| Signature Detection | ~85% |
| Memory Usage | 6-8GB |
| Formats | PDF, PNG, JPG, JPEG |

---

## 🔒 Offline Mode (No Internet Required)

### First Setup (Requires Internet)
```bash
python setup_models.py
# Downloads: Qwen2-VL (~4GB), YOLO, OCR models (~1GB)
```

### Running Offline
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
streamlit run app.py
# ✅ 100% offline after setup!
```

### Cached Models

| Component | Size | Status |
|-----------|------|--------|
| Qwen2-VL-2B | ~4GB | ✅ |
| YOLOv8 | ~100MB | ✅ |
| PaddleOCR | ~400MB | ✅ |
| **Total** | **~5GB** | **One-time** |

### Deploy Offline

```bash
# Pre-download on machine with internet
python setup_models.py

# Copy to air-gapped server
scp -r .huggingface_cache/ user@server:/app/
scp -r model_weights/ user@server:/app/

# Run offline on server
export HF_HUB_OFFLINE=1
streamlit run app.py
# ✅ Zero internet needed!
```

### Perfect For
- 🏢 Corporate networks (restricted)
- 🚢 Remote locations
- 🔒 Air-gapped systems
- ✈️ Offline deployment

---
## 🐛 Troubleshooting

**CUDA out of memory:**
```bash
# Comment GPU initialization or reduce resolution
```

**Model not found:**
```bash
python setup_models.py
```

**Low confidence:**
- Check image quality
- Try different preprocessing methods

**Stamp/signature not detected:**
- YOLO trained on specific invoice types
- Consider fine-tuning with your data

---

## 🚀 Deployment

### Streamlit Cloud
```bash
git push origin main
# Go to share.streamlit.io → New App → Select repo
```

### Docker
```bash
docker build -t invoice-extractor .
docker run -p 8501:8501 invoice-extractor
```

### AWS/Azure/GCP
See [RUN_APP.md](RUN_APP.md) for details

---

## 🤝 Contributing

```bash
git checkout -b feature/your-feature
git commit -m 'Add feature'
git push origin feature/your-feature
```

---

## 📄 License

MIT - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- **Qwen2-VL** by Alibaba Cloud
- **YOLOv8** by Ultralytics
- **PaddleOCR** by PaddlePaddle
- **Streamlit** framework

---

⭐ If this helps you, please star the repo!

