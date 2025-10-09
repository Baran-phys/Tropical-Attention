# 🌴 Tropical-Attention [NeurIPS 2025]
**Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms.** ([paper](https://arxiv.org/abs/2505.17190))

Dynamic programming (DP) algorithms for combinatorial optimization problems work with taking maximization, minimization, and classical addition in their recursion algorithms. The associated value functions correspond to convex polyhedra in the max plus semiring. Existing Neural Algorithmic Reasoning models, however, rely on softmax-normalized dot-product attention where the smooth exponential weighting blurs these sharp polyhedral structures and collapses when evaluated on out-of-distribution (OOD) settings. We introduce Tropical attention, a novel attention function that operates natively in the max-plus semiring of tropical geometry. We prove that Tropical attention can approximate tropical circuits of DP-type combinatorial algorithms. We then propose that using Tropical transformers enhances empirical OOD performance in both length generalization and value generalization, on algorithmic reasoning tasks, surpassing softmax baselines while remaining stable under adversarial attacks. We also present adversarial-attack generalization as a third axis for Neural Algorithmic Reasoning benchmarking. Our results demonstrate that Tropical attention restores the sharp, scale-invariant reasoning absent from softmax.

---
We experiment on 11 combinatorial tasks:

1. **Floyd–Warshall** - All-pairs shortest paths on a weighted undirected graph. (Both Regression and Classification)
2. **Quickselect** — Find the k-th smallest elements in a set. (Classification)
3. **3SUM (Decision)** — Decide if there exist a, b, c with a+b+c=T. (Classification)
4. **Balanced Partition** - Split numbers into two subsets with equal sum. (NP-complete Classification)
5. **Convex Hull** - Given 2D points, identify the hull. (Classification)
6. **Subset Sum (Decision)** - Decide if some subset sums to a target T. (NP-complete Classification)
7. **0/1 Knapsack** — Maximize value under capacity with binary item choices. (NP-hard Classification)
8. **Fractional Knapsack** — Items can be taken fractionally; predict optimal value. (Regression)
9. **Strongly Connected Components (SCC)** Decompose a directed graph with community structure; predict pairwise same-component. (Classification)
10. **Bin Packing** — Pack items into the fewest bins of fixed capacity. (NP-hard Classification)
11. **Min Coin Change** — Minimum number of coins to reach a target T with each coin used at most once. (Classification)

---

## Tropical kernel

If you just want the **Tropical Attention kernel**, use **`TropicalAttention.py`**.

<summary>Example (instantiate inside your model)</summary>

```python
from TropicalAttention import TropicalAttention
import torch

attn = TropicalAttention(
    d_model=128,
    n_heads=8,
    device="cuda" if torch.cuda.is_available() else "cpu",
    tropical_proj=True,
    tropical_norm=False,
    symmetric=True,
)

x = torch.randn(2, 32, 128)     # [batch, seq, d_model]
y, scores = attn(x)
```

---

### Training a model from scratch
```bash
python experiment.py --job_file jobs_to_do_train --job_id 0 
```
`--job_id` selects the row in `jobs_to_do_train.csv`.
The script logs training progress to outputs/<timestamp>/.

---

### Citation

If you use this repository or Tropical Attention in your research, please cite:

```bibtex
@article{hashemi2025tropical,
  title={Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms},
  author={Hashemi, Baran and Pasque, Kurt and Teska, Chris and Yoshida, Ruriko},
  journal={arXiv e-prints},
  pages={arXiv--2505},
  year={2025}
}
```





