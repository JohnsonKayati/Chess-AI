#  Chess-AI

A hybrid neural network-based chess AI built with PyTorch, combining the spatial reasoning of Convolutional Neural Networks (CNNs) with the sequential pattern recognition of Transformers. Trained on high-level games (~2000 ELO), the model predicts what a strong player would likely play in a given position.

---

##  About the Model

This project is an advanced chess move predictor that mimics the decision-making of a 2000-rated player. It uses:

- **CNNs** to analyze spatial features from the board — piece positions, castling rights, and board structure.
- **Transformers** to capture the sequential flow of the game from previous moves.

Together, this hybrid architecture excels at both static evaluation and recognizing patterns across move histories.

---

##  Training & Performance

- **Input Format**: 8×8×14 board tensors — 14 channels encode piece types, player turn, and castling rights.
- **Target**: Predict the most likely move a 2000 ELO player would play based on the current board and recent moves.
- **Top-K Accuracy**:
  - **Training Set (Top-5)**: 95%
  - **Test Set (Top-5)**: 94%
- **Positional Encoding**: Board states are embedded as tensors and processed via CNNs before being passed to the Transformer module.

---

##  Tech Stack

- **Framework**: PyTorch
- **Libraries**: `torch`, `numpy`, `pygame`, `pickle`

---

##  Play Against the AI

A playable interface is included via `GM_play.py`, allowing users to test their skills against the trained AI.

```bash
python GM_play.py
