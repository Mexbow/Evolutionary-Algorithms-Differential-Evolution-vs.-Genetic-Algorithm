# 🔬 Evolutionary Algorithms: Differential Evolution vs. Genetic Algorithm

This project implements and compares two powerful evolutionary optimization algorithms: **Differential Evolution (DE)** and **Genetic Algorithm (GA)**. Both algorithms are applied to a suite of benchmark functions to explore their performance in solving complex, non-linear optimization problems.

## 📌 Benchmark Functions

- 🌀 **Ackley**
- 🏔️ **Bukin**
- ✖️ **Cross-in-Tray**
- 🌊 **Drop-Wave**
- 🍳 **EggHolder**

Each function is visualized in both **2D** and **3D** plots, highlighting the best solution found by each algorithm on the function’s surface.

---

## ✅ Features

- 🔁 Differential Evolution (DE) implementation
- 🧬 Genetic Algorithm (GA) with:
  - Crossover
  - Mutation
  - Roulette Wheel Selection
- 🧠 Optimization over 5 well-known benchmark functions
- 📈 2D & 3D function visualization using **Matplotlib**
- 🐍 Fully implemented in **Python** with **NumPy**

---

## 📊 Output

- Prints **best fitness per generation** for each algorithm
- Displays **2D contour plots** and **3D surface plots**
- Shows best-found solution points for DE and GA on each function

---

## 🛠️ Getting Started

Clone the repo and run:

```bash
EA.py
