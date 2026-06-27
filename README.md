# SRL — Structural Reinforcement Learning

A pedagogical [JAX](https://github.com/jax-ml/jax) implementation of
policy-gradient methods for heterogeneous-agent macro models in general
equilibrium, including with aggregate risk.

SRL goes with the paper *Structural Reinforcement Learning* by Chiyuan Wang,
Yucheng Yang, Andreas Schaab, and Benjamin Moll. The point is that you can read
it. A second-year macro PhD student should be able to clone the repo, read the
notebooks top to bottom, and follow both the economics and the method without us
explaining it.

## The tutorials

Start with the tutorials. They take one household problem and grow it across the
sequence, one new idea at a time, checking each step against the one before. Read
them in order from notebook 0. [TUTORIAL.md](TUTORIAL.md) is the full syllabus:
who it's for, what each notebook teaches and assumes, and the checks it runs.

| # | Notebook | Open in Colab |
|---|----------|:-------------:|
| 0 | [Household problem, in NumPy](tutorials/00_household_numpy.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreasschaab/SRL/blob/main/tutorials/00_household_numpy.ipynb) |
| 1 | [The same problem, in JAX](tutorials/01_household_jax.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreasschaab/SRL/blob/main/tutorials/01_household_jax.ipynb) |
| 2 | [The same problem, by policy gradient](tutorials/02_household_policy_gradient.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreasschaab/SRL/blob/main/tutorials/02_household_policy_gradient.ipynb) |
| 3 | [The household with moving prices](tutorials/03_household_moving_prices.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreasschaab/SRL/blob/main/tutorials/03_household_moving_prices.ipynb) |
| 4 | [The Huggett economy](tutorials/04_ge_huggett.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreasschaab/SRL/blob/main/tutorials/04_ge_huggett.ipynb) |
| 5 | [Huggett with aggregate risk](tutorials/05_ge_huggett_agg_risk.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreasschaab/SRL/blob/main/tutorials/05_ge_huggett_agg_risk.ipynb) |

Each notebook runs top to bottom on a free Colab GPU (click a badge) or locally
on a CPU, slower. On Colab the first cell clones the repo and installs the
package, so there's no token and no setup notebook.

## Install (local)

```sh
pip install -e .
```

Python ≥ 3.11. JAX is pinned (`jax==0.10.0`) so results reproduce exactly across
machines.

## What's here

- **`srl/`**: the package. The policy-gradient solver (`SPGSolver`), a
  value-function-iteration baseline (`VFISolver`), and the discretization and
  plotting helpers (`srl.utils`).
- **`tutorials/`**: the notebook sequence (percent-format `.py` sources and the
  generated `.ipynb`), the shared `calibration.py`, and the SSJ reference data.
  The notebooks are generated from their `.py` sources by
  `tutorials/build_notebooks.py`, so don't hand-edit a `.ipynb`.
- **`TUTORIAL.md`**: the syllabus.

## License & citation

MIT, see [LICENSE](LICENSE). If you use SRL in your work, please cite the paper
and this repo (see [CITATION.cff](CITATION.cff)).
