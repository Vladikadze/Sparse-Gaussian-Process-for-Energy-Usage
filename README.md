# Sparse Gaussian Process for Energy Usage

## Motivation

This project started out of personal curiosity:
how far can you push **sparse Gaussian Processes** on a real-world time-series dataset without overengineering everything?

The goal wasn’t to build a production system, but to understand:

* how FITC-style sparse GPs behave in practice
* how sensitive they are to hyperparameters
* where they break or underperform

---

## What this does

* Uses a **steel industry energy consumption dataset**
* Focuses on **weekday data only**
* Predicts **energy usage (kWh)** based on **time of day**
* Implements a **Sparse Gaussian Process (FITC approximation)**
* Compares behavior across:

  * lengthscale
  * noise
  * signal variance
* Includes some basic exploratory analysis (variance, trends, etc.)

---

## Pipeline (short version)

1. Load and sort data by date
2. Filter weekdays
3. Train/test split (temporal, not random)
4. Normalize:

   * time → [0, 1]
   * target → standard score
5. Fit sparse GP with inducing points
6. Evaluate predictions (MSE + visual checks)

---

## Results

What worked:

* The model **captures the general daily pattern** pretty well
* Sparse approximation keeps things **computationally manageable**
* Reasonable performance with a small number of inducing points

What didn’t:

* Performance is **very sensitive to hyperparameters**
* Some runs clearly **underfit the data**
* The model struggles with:

  * sharper peaks
  * irregular daily behavior

---

## Underperforming Areas (Important)

These are the weak spots right now:

* **Hyperparameter tuning**

  * Currently grid-based and shallow
  * No real optimization (no marginal likelihood maximization)

* **Inducing point selection**

  * Fixed / naive choice
  * No optimization or adaptive placement

* **Feature space**

  * Only uses time of day
  * Ignores potentially important signals:

    * load type
    * lag features
    * seasonality beyond daily cycle

* **Model flexibility**

  * Single RBF kernel → too smooth
  * Can’t capture abrupt changes well

* **Uncertainty quality**

  * Variance estimates not deeply evaluated
  * No calibration checks

---

## What should be added next

If this were to be pushed further:

* Proper **hyperparameter optimization**

  * log marginal likelihood
  * Bayesian optimization

* Better **inducing point strategy**

  * K-means initialization
  * learned locations

* Richer features:

  * categorical encoding of `Load_Type`
  * time features (hour, cyclic encoding)
  * lagged usage values

* Try different kernels:

  * periodic kernel (for daily cycles)
  * composite kernels (RBF + periodic)

* Compare against baselines:

  * linear regression
  * tree-based models
  * full GP (on smaller subset)

* Add proper evaluation:

  * MAE / RMSE comparison
  * uncertainty calibration

---

## How to run

Right now it’s a single notebook.

You’ll need:

* numpy
* pandas
* matplotlib
* scikit-learn

Update the dataset path:

```python
DATA_PATH = "path/to/Steel_industry_data.csv"
```

Then just run the notebook top to bottom.

---

## Notes on AI usage

* README file: generated with AI assistance
* Minor syntax fixes: AI-assisted

Everything else (modeling, structure, experimentation) was done manually.

---

## Final note

This is not a polished implementation.
It’s a working exploration.

If anything, the main takeaway is:
**sparse GPs are powerful, but fragile if you don’t treat hyperparameters and structure seriously.**
