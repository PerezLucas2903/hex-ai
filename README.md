# Hex-AI

A research project exploring neural network agents for playing the game **Hex**, with interactive visualization and (hopefully) model interpretability.

---

## 🧱 Project Structure

- `src/game`: Core Hex implementation.
- `src/models`: Neural network architectures and training code.
- `src/interpretability`: Explainability tools (saliency, SHAP, etc.).
- `src/interface`: Visualization and user interaction.
- `data`: Datasets for training/evaluation.
- `notebooks`: Experiments and analysis.
- `tests`: Unit tests.

---

## 🚀 Getting Started

### 1. Clone and install
```bash
git clone https://github.com/PerezLucas2903/hex-ai.git
cd hex-ai
pip install -r requirements.txt
```

### 2. Play Hex against models
Play against random moves:
```
python play_human_vs_model.py --adversary-type random
```

Play against a trained model
```
python play_human_vs_model.py --adversary-type nn --model-path path/to/model.pth
```