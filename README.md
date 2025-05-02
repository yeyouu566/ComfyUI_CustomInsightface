# ComfyUI Custom InsightFace Node

This is a custom ComfyUI node that uses [`insightface`](https://github.com/deepinsight/insightface) to detect and analyze faces.

---

## 🧩 Features

- Loads InsightFace model (`buffalo_l`)
- Accepts PIL or NumPy (ndarray) image input
- Returns facial bounding box info, embeddings, landmarks, etc.
- Designed for easy use in ComfyUI with plug-and-play setup

---

## 📦 Installation

### 1. Clone this repo inside your `ComfyUI/custom_nodes/` directory

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yeyouu566/ComfyUI_CustomInsightface.git
