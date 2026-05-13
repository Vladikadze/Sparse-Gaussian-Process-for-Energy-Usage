# Sparse Gaussian Process for Energy Usage

Predicting steel plant energy consumption using sparse GPs — built to understand how these models behave in practice, not to ship a production system.

## What this does

The dataset is 15-minute interval energy readings from a steel plant, split by load type (Light, Medium, Maximum). The model predicts energy usage (kWh) from time of day.

The notebook runs two inference approaches side by side:

**FITC** — a classical sparse GP approximation. Fast, no gradient-based training, hyperparameters tuned by grid search.

**SVGP (Stochastic Variational GP)** — a variational formulation that optimises a lower bound on the marginal likelihood (ELBO). Inducing point locations are learned jointly with kernel parameters rather than fixed in advance.

Both are wrapped in a mixture-of-experts setup: one GP per load type, combined using either distance-based or logistic-regression gating.

## Results

| Model | RMSE | NLL | 95% coverage |
|---|---|---|---|
| SVGP — distance-gated | 0.634 | 0.854 | 90.8% |
| SVGP — learned-gating | 0.640 | 0.938 | **94.8%** |
| FITC — learned-gating | 0.666 | 1.637 | 83.1% |
| FITC — distance-gated | 0.698 | 92.19 | 34.0% |

RMSE differences are modest. The bigger story is in NLL and coverage: FITC with distance gating produces wildly overconfident uncertainty estimates (NLL 92, 34% coverage on a nominal 95% interval). SVGP with learned gating hits 94.8% coverage — essentially exact calibration.

The takeaway: the inference method matters more for uncertainty quality than for point prediction accuracy.

## Pipeline

1. Load and sort by date
2. Filter to weekdays only
3. Temporal train/test split (no data leakage)
4. Normalise: time → [0, 1], target → standard score
5. Tune one FITC GP per load type via grid search
6. Train one SVGP per load type via ELBO optimisation (500 iterations, Adam)
7. Combine experts using two gating methods
8. Evaluate on held-out test set: RMSE, NLL, coverage

## What's interesting

**Learned inducing points.** The SVGP moves its inducing points during training rather than keeping them on a uniform grid. They end up clustered in the high-density, high-variance parts of the input space — the model is deciding where to spend its representational budget.

**ELBO convergence.** All three experts converge within ~100 iterations. Maximum_Load starts at a much lower ELBO than the others, reflecting its greater heterogeneity.

**Gating matters for uncertainty.** Distance-based gating produces badly calibrated intervals under FITC. Learned logistic-regression gating fixes most of this. Under SVGP both gating methods produce reasonable uncertainty — the variational posterior is better regularised.

## What's not here

- Hyperparameter optimisation via marginal likelihood (FITC uses grid search only)
- Features beyond time of day (no lag features, no categorical encoding of load type)
- Kernel comparisons (only RBF used)
- Calibration plots

## How to run

```
pip install numpy pandas matplotlib scikit-learn torch gpytorch
```

Update the data path at the top of the notebook:

```python
DATA_PATH = "path/to/Steel_industry_data.csv"
```

Then run top to bottom. The SVGP section installs gpytorch automatically if run in Colab.

## Notes on AI usage

README: written with AI assistance. Minor syntax fixes: AI-assisted. Everything else (modelling decisions, experiments, analysis) was done manually.
