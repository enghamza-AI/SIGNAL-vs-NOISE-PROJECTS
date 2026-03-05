# Signal vs Noise – Hands-On ML Intuition Projects

This repository contains 5 beginner-friendly Python projects designed to build deep intuition about **signal**, **noise**, **irreducible error**, and why models hallucinate patterns in random data.

Each project uses only NumPy, Matplotlib, and scikit-learn — no deep learning required.

## Why these projects?

Most ML tutorials jump straight to fancy models. These experiments force you to answer the first real question every good ML engineer asks:

> "Is there actually something meaningful to learn here?"

## Projects Overview

| # | Project Name                        | Core Lesson                                      | Key Visual / Output                          |
|---|-------------------------------------|--------------------------------------------------|----------------------------------------------|
| 1 | Noise Injection Lab                 | When noise drowns signal, models hallucinate     | Slope instability & exploding test MSE       |
| 2 | Fake Pattern Generator              | Models fit lines even to pure random data        | Confident green line through scattered dots  |
| 3 | Signal Strength Detector            | Detect strong / weak / no signal before modeling | R² + |slope| classification (strong/weak/none) |
| 4 | Irreducible Error Simulator         | Some error is permanent — more data hits a wall  | Test MSE flattening at noise variance floor  |
| 5 | Feature Corruption Experiment       | Destroy signal-carrying features → collapse     | R² crash & MSE explosion heatmap-like curve |


# Run any project (each is a standalone .py file)
python project1_noise_injection.py
# or jupyter notebook if you prefer notebooks
