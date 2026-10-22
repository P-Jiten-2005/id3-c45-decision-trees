# ID3 and C4.5 — how they work

This document explains the two algorithms implemented in [`dtree/`](../dtree) and the
choices made in the implementation. Code references point to the functions that compute
each quantity.

## 1. Decision-tree induction

Both algorithms build the tree top-down and greedily:

```
BuildTree(S, attributes):
    create node N with the class distribution of S
    if S is pure, attributes is empty, or a stopping rule fires:
        return N as a leaf labelled with the majority class
    A ← the attribute that best separates the classes in S
    for each outcome v of the test on A:
        N.child[v] ← BuildTree(S_v, attributes')
    return N
```

They differ in **how "best" is measured**, **which tests are allowed**, and **what happens
after the tree is grown**.

## 2. Entropy and information gain (ID3)

For a set *S* with class proportions *p₁ … p_k*:

$$H(S) = -\sum_{i=1}^{k} p_i \log_2 p_i$$

*H* is 0 for a pure set and log₂ *k* when every class is equally likely
([`criteria.entropy`](../dtree/criteria.py)).

Splitting *S* on attribute *A* with values *v* gives subsets *S_v*. The information gain is
the expected reduction in entropy
([`criteria.information_gain`](../dtree/criteria.py)):

$$\text{Gain}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

### Worked example — Play Tennis

The 14-day dataset has 9 *Yes* and 5 *No*:

$$H(S) = -\tfrac{9}{14}\log_2\tfrac{9}{14} - \tfrac{5}{14}\log_2\tfrac{5}{14} = 0.940$$

| Outlook | Yes | No | H(S_v) |
|---|---|---|---|
| Sunny | 2 | 3 | 0.971 |
| Overcast | 4 | 0 | 0.000 |
| Rain | 3 | 2 | 0.971 |

$$\text{Gain}(S, \text{Outlook}) = 0.940 - \tfrac{5}{14}(0.971) - \tfrac{4}{14}(0) - \tfrac{5}{14}(0.971) = 0.247$$

The gains for Humidity (0.152), Wind (0.048) and Temperature (0.029) are lower, so ID3
places Outlook at the root. Repeating the procedure gives the tree in the README; the test
`test_id3_reproduces_classic_tennis_tree` checks it.

## 3. ID3 specifics

* **Multi-way splits**: one branch per attribute value seen at the node.
* **Each attribute is used at most once** on a root-to-leaf path.
* **Categorical only**. The preprocessing pipeline discretises numeric attributes into
  *k* bins (equal-frequency by default) before ID3 sees them.
* **No missing values**. They are imputed (mode / median from the training split).
* **Stops** when a node is pure, no attributes remain, the best gain is zero, `max_depth`
  is reached, or fewer than `min_samples_split` samples remain.
* **Unseen values at prediction time** (a category not present at that node during
  training) fall back to the node's majority class.
* **Weakness — bias towards many-valued attributes**: an ID column splits the data into
  pure singletons, so its gain equals *H(S)*, the maximum possible. ID3 picks it and
  memorises the training set. `test_c45_prefers_gain_ratio_over_id_column` demonstrates
  this.

## 4. C4.5 improvements

### 4.1 Gain ratio

C4.5 divides gain by the **split information**, the entropy of the partition itself
([`criteria.split_information`](../dtree/criteria.py)):

$$\text{SplitInfo}(S, A) = -\sum_{v} \frac{|S_v|}{|S|}\log_2\frac{|S_v|}{|S|}
\qquad
\text{GainRatio}(S, A) = \frac{\text{Gain}(S, A)}{\text{SplitInfo}(S, A)}$$

For Outlook: SplitInfo = 1.577, so GainRatio = 0.247 / 1.577 = 0.156.

An attribute with almost no split information (nearly all samples share one value) can
get an inflated ratio from a tiny gain. Following Quinlan, the implementation only
considers attributes whose gain is **at least the average gain** of all candidates, then
picks the highest gain ratio among them (`C45Classifier._build`).

C4.5 also requires that **at least two branches receive `min_samples_leaf` samples**
(default 2, Quinlan's `MINOBJS`), which rules out ID-like splits.

### 4.2 Continuous attributes

For a numeric attribute the samples are sorted by value. Every midpoint between two
consecutive distinct values is a candidate threshold *t*, giving the binary test
*A ≤ t* versus *A > t*. Cumulative class counts make evaluating all thresholds a single
O(n log n) pass (`C45Classifier._eval_continuous`).

* The threshold with the highest information gain is chosen.
* As in C4.5 Release 8 (Quinlan, 1996), the gain is reduced by
  log₂(*number of candidate thresholds*) / |*S*|. Without this MDL-style correction,
  numeric attributes would be favoured simply because they offer many thresholds.
* Unlike categorical attributes, a numeric attribute **can be tested again** lower in the
  tree with a different threshold.

### 4.3 Missing values

C4.5 never discards an incomplete example. Each sample carries a weight (initially 1).

* **Choosing a test**: gain is computed on samples whose value is known and multiplied by
  the known fraction *F*: Gain(S, A) = *F* × (H(S_known) − Σ …). Split information treats
  "missing" as one more outcome.
* **Splitting**: a sample with a known value goes to its branch unchanged. A sample with a
  missing value goes to **every** branch, with its weight multiplied by that branch's share
  of the known weight. All class distributions in the tree are therefore weighted (they can
  show fractional counts).
* **Classifying**: if the tested value is missing, the sample is sent down all branches
  and the resulting class distributions are combined using the same training shares
  (`C45Classifier._distribute`).

Choose **"C4.5 native handling"** in the app to use this; otherwise missing values are
imputed for both algorithms so they compete on identical data.

### 4.4 Pessimistic error pruning

A fully grown tree overfits. C4.5 prunes it bottom-up using only the training data.

For a node covering *N* (weighted) samples with *E* misclassified, the observed error rate
*E/N* is optimistic. C4.5 treats the errors as binomial and uses the **upper limit of a
confidence interval** with confidence factor *CF* (default 25 %):

$$U_{CF}(E, N) = \frac{E + 0.5 + \frac{z^2}{2} + z\sqrt{(E + 0.5)\left(1 - \frac{E + 0.5}{N}\right) + \frac{z^2}{4}}}{N + z^2}$$

where *z* is the standard-normal quantile for 1 − *CF*. The estimated number of errors is
*N · U_CF(E, N)*. Small cases (*E* < 1) use the exact binomial bound, as in Quinlan's
`AddErrs` routine, which [`criteria.pessimistic_errors`](../dtree/criteria.py) ports.
Quinlan's own example: a leaf with 6 cases and no errors gets *U* = 0.206, i.e. 1.24
predicted errors.

**Subtree replacement**: after its children have been pruned, a node is turned into a
leaf if its own estimated errors are no greater than the sum of its children's estimated
errors (plus 0.1, as in C4.5). A lower *CF* is more pessimistic and prunes more.

Not implemented: subtree *raising* (C4.5 can also replace a node with its most populated
child), windowing, and rule-set generation (C4.5rules). Pruning only by replacement is a
common teaching simplification and still captures the main effect.

## 5. Summary

| | ID3 | C4.5 |
|---|---|---|
| Split criterion | Information gain | Gain ratio (among above-average-gain attributes) |
| Categorical attributes | Multi-way split, used once per path | Same |
| Continuous attributes | Discretised during preprocessing | Binary threshold split, reusable |
| Missing values | Must be imputed | Fractional weights in training and prediction |
| Minimum branch size | — | ≥ 2 branches with `min_samples_leaf` samples |
| Pruning | None | Pessimistic error pruning (subtree replacement) |
| Typical behaviour | Large tree, near-perfect training accuracy | Smaller tree, better generalisation |

## 6. Evaluation metrics

With TP, FP, FN for a given class *c*:

* **Precision** = TP / (TP + FP) — of the samples predicted as *c*, how many really are *c*.
* **Recall** = TP / (TP + FN) — of the real *c* samples, how many were found.
* **F1-score** = 2 · Precision · Recall / (Precision + Recall).
* **Accuracy** = correct predictions / all predictions.

Per-class values are combined by **macro** averaging (every class counts equally),
**weighted** averaging (by class support — the default) or **micro** averaging (global
counts, equal to accuracy for single-label problems). The **confusion matrix** has actual
classes as rows and predicted classes as columns; the diagonal holds correct predictions.

## References

* J. R. Quinlan, "Induction of decision trees," *Machine Learning*, 1(1):81–106, 1986.
* J. R. Quinlan, *C4.5: Programs for Machine Learning*, Morgan Kaufmann, 1993.
* J. R. Quinlan, "Improved use of continuous attributes in C4.5," *Journal of Artificial
  Intelligence Research*, 4:77–90, 1996.
* T. M. Mitchell, *Machine Learning*, McGraw-Hill, 1997, chapter 3.
