# Method summary

This page is a one-page formal summary of the TRACE framework. See the paper
for the full derivation and empirical evaluation.

## Problem setup

Let $\mathcal{D}=\{d_i\}_{i=1}^{|\mathcal{D}|}$ be the audit corpus where each
evidence snippet is a triple $d_i=(x_i, t_i, \kappa_i)$ with text $x_i$,
fiscal year $t_i \in \mathbb{Z} \cup \{\bot\}$, and statistical calibre
$\kappa_i \in \mathcal{K}$. For a query $q$ the system produces a
role-structured response

$$y \in \mathcal{Y}_{\mathrm{audit}} = \mathcal{Y}_{\mathrm{OCI}} \cup \{y_{\mathrm{abstain}}\},$$

where $\mathcal{Y}_{\mathrm{OCI}}$ groups atomic claims into an Observation,
Cause, and Impact component. A quantitative claim is admitted only if it is
supported by a *single* evidence snippet under the attribute equivalence
$(v, u_s, t, \kappa)$.

## Stage 1 — Risk-aware dual-space retrieval

Given token-level contextual embeddings $\mathbf{H}_x = f_\phi(x)$, TRACE
computes a softmax risk attention $a_i(x)$ against a fixed risk prototype
$\mathbf{w}_r$ and represents each text as a discrete risk measure

$$\mu(x) = \sum_i a_i(x)\,\delta_{\mathbf{h}_i}.$$

For a query–document pair $(q, d)$ the alignment cost is the entropy-regularised
optimal-transport cost

$$\mathrm{OT}_{\varepsilon}(q, d) = \min_{T \in \Pi(\mathbf{a}, \mathbf{b})}\sum_{ij} T_{ij} C_{ij} + \varepsilon \sum_{ij} T_{ij} \log T_{ij},$$

with $C_{ij} = \|\mathbf{h}_i^q - \mathbf{h}_j^d\|_2^2$. The retrieval score
combines four signals,

$$\mathrm{Score}(q, d) = s_{\mathrm{sem}} + \lambda_{\mathrm{ot}} s_{\mathrm{ot}} + \lambda_{\mathrm{risk}} s_{\mathrm{risk}} + \lambda_{\mathrm{num}} s_{\mathrm{num}},$$

each normalised within the rerank candidate set. Retrieval is executed as a
cascade: dense semantic top-$K$ followed by per-candidate fusion and rerank.

## Stage 2 — Set-level OCI evidence-chain construction

For each role $r \in \{\mathrm{O}, \mathrm{C}, \mathrm{I}\}$ a role-conditioned
query $q_r = [r] \oplus q$ produces an independent candidate pool
$\mathcal{C}_r$. The joint chain $\mathcal{E} = (e_\mathrm{O}, e_\mathrm{C}, e_\mathrm{I})$
is scored by

$$J(\mathcal{E}) = \mathrm{Cov}(q, \mathcal{E}) + \eta \Big( e^{-\mathrm{OT}(\mu(e_\mathrm{O}), \mu(e_\mathrm{C}))} + e^{-\mathrm{OT}(\mu(e_\mathrm{C}), \mu(e_\mathrm{I}))} \Big),$$

where $\mathrm{Cov}(q, \mathcal{E}) = \exp(-\mathrm{OT}(\mu(q), \nu(\mathcal{E})))$
and $\nu(\mathcal{E})$ is the role-weighted mixture of per-snippet risk
measures. A beam search approximates the argmax. If
$J(\mathcal{E}^*) < \delta_c$, the system falls back to $y_{\mathrm{abstain}}$.

## Stage 3 — Attribute-consistent generation

For each atomic claim $u$ and evidence $e \in \mathcal{E}^*$ the support score
mixes semantic similarity and attribute consistency:

$$\mathrm{Align}(u, e) = \lambda_{\mathrm{attr}} \mathrm{Cons}(u, e) + (1 - \lambda_{\mathrm{attr}}) s_{\mathrm{sup}}(u, e),$$

with $\mathrm{Cons}(u, e)$ defined as the product of value ($D_v$), unit
($M_u$), year ($M_t$), and calibre ($\mathrm{Sim}_\kappa$) compatibility.
Unsupported risk claims (with $\mathrm{Align}(u, c(u)) < \delta_a$) are
suppressed locally; if no legal risk claim remains the pipeline globally
abstains.

## Evaluation metrics

- **Retrieval:** `Recall@K`, `nDCG@K`, `RR@K` (role-recall).
- **Generation:** `ROUGE-L`, character-level `F1`.
- **Audit:** `AC` (audit consistency), `QAC` (quantitative-attribute
  consistency), `OCR` (OCI completeness rate).
- **Stress test:** `VAR` (valid abstention rate), `PHR` (pseudo-risk
  hallucination rate).

The hyper-parameters used for the main experiments are the constants at the
top of [`experiments/run_main.py`](../experiments/run_main.py).
