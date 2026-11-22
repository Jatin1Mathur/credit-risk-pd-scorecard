# Study Guide: PD Scorecard

Read this before an interview. Every section maps to code in `src/pdmodel/`.

## 1. The problem

A **probability of default (PD)** is the chance a borrower fails to pay within a horizon (usually 12
months). Together with **LGD** (loss given default, the share lost) and **EAD** (exposure at default)
it gives **expected loss: EL = PD × LGD × EAD**. Banks under Basel IRB and rating agencies both need PDs
that (1) rank borrowers well and (2) are at the right level.

Those two needs are tested separately:
- **Discrimination** (ranking): Gini, AUC, KS.
- **Calibration** (level): Brier, Hosmer–Lemeshow, binomial test per grade.

A model can rank perfectly and still have every PD wrong by a factor of 5. That is why step 6 exists.

## 2. Weight of Evidence (binning.py)

For each bin: `WoE = ln(%bad in bin / %good in bin)`.
- WoE > 0: the bin is riskier than average. WoE < 0: safer.
- Binning turns any feature (numeric or text) into one number on the log-odds scale, handles
  outliers (they fall into the end bins) and non-linear effects.
- **Monotonic bins**: the bad rate must only rise or only fall across bins of a numeric feature, so the
  model is explainable ("longer duration = riskier"), which reviewers and regulators expect.
- **Minimum 5 % per bin**: avoids WoE values built on a handful of loans.
- **Laplace smoothing (+0.5)**: a bin with zero bads would otherwise give WoE = −∞.

**Information Value**: `IV = Σ (%bad − %good) × WoE`. Rules of thumb: < 0.02 useless, 0.02–0.1 weak,
0.1–0.3 medium, > 0.3 strong. Checking-account status has IV 0.63, the strongest predictor.

## 3. Feature selection (model.py)

1. IV ≥ 0.02.
2. Correlation filter: if two WoE features correlate above 0.7, keep the higher-IV one (avoids
   unstable coefficients from multicollinearity).
3. Backward elimination in the logistic regression: drop the feature with p > 0.05, and drop any
   feature whose coefficient is **negative**. With WoE = ln(%bad/%good), every coefficient should be
   positive; a negative one means the feature's effect flipped because of correlation with others.

Result: 18 candidates → 9 features. Credit amount, property and age dropped on p-value.

**Why sex and nationality are excluded**: anti-discrimination law (e.g. EU, US ECOA) and responsible-AI
practice. They are removed before binning so they cannot influence anything.

## 4. Logistic regression

`log(PD / (1 − PD)) = β0 + Σ βi · WoEi`

statsmodels is used instead of scikit-learn because it reports **standard errors and p-values**, which
model documentation requires. No regularisation: the variables are already few and screened.

## 5. Scorecard points

Points are a linear rescaling of log-odds: **600 points at good:bad odds of 50:1, 20 points to double
the odds (PDO)**.
`factor = PDO / ln 2`, `offset = 600 − factor · ln 50`, `score = offset + factor · ln(good odds)`.
The points table assigns points per bin, so a loan officer can add up a score by hand. A test checks
that the per-bin points add up to the model score.

## 6. Calibration (calibration.py)

German Credit has 30 % defaults because bad loans were over-sampled. A real portfolio's long-run
default rate (the **central tendency**) is much lower; here 5 % is assumed.

The fix is an **intercept shift**: add a constant to the log-odds of every borrower so the mean PD
equals 5 %. It is found with Brent's root-finding method. Because every borrower moves by the same
amount in log-odds, the **ranking and Gini do not change**; only the level does. A test checks both.

This is the core of "model calibration" in rating methodology: the ranking model and the PD level
are separate decisions.

## 7. Master scale and back-testing

Calibrated PDs map to 7 grades (R1 ≤ 0.5 %, R2 ≤ 1 %, … R7 > 16 %), like an agency rating scale.
Checks:
- **Monotonicity**: observed default rate rises from R1 to R7. ✔
- **Binomial test** per grade: if the PD is right, the number of defaults in a grade follows a
  Binomial(n, PD). A one-sided p < 5 % means the PD is too low. No grade fails.

The test uses PDs at the **sample** default level (30 %), because the observed defaults in this
dataset are at that level. Grades are assigned with the long-run PDs.

## 8. Validation metrics (metrics.py)

| Metric | Meaning | Here |
|---|---|---|
| AUC | P(random bad scores riskier than random good) | 0.774 |
| Gini | 2·AUC − 1, also called accuracy ratio | 0.547 |
| KS | Largest gap between the score CDFs of bads and goods | 0.457 |
| Brier | Mean squared error of PD vs. outcome | 0.168 |
| Hosmer–Lemeshow | χ² test of predicted vs. observed defaults by decile; high p = fine | p = 0.61 |
| PSI | Shift of the score distribution between samples; < 0.1 stable | 0.013 |

**Leakage in cross-validation**: WoE uses the target. If bins are fitted on all data and then
cross-validated, the validation folds have seen their own labels. `cross_validate` refits binning,
selection and regression inside every fold.

**Bootstrap CI**: with only 90 defaults in the test set, Gini has a wide 95 % interval (0.43–0.65).
Always quote uncertainty on small samples.

## 9. Challenger model

Gradient boosting (HistGradientBoosting) reaches Gini 0.60 vs. 0.55. Arguments for keeping the
scorecard: explainability (points per bin), fewer inputs, stable coefficients, regulatory acceptance,
and the gap is inside the scorecard's confidence interval. Challenger models are used to check that
the champion is not leaving a lot of performance on the table.

## 10. Likely interview questions

1. Why WoE instead of one-hot encoding? *Monotonic, one coefficient per feature, handles outliers and
   missing values, directly interpretable on the log-odds scale.*
2. What does a negative coefficient on a WoE variable tell you? *Multicollinearity or overfitting; the
   variable's sign flipped. Drop or re-bin it.*
3. Your Gini is 0.55, is that good? *For application scorecards 0.4–0.6 is typical; German Credit
   benchmarks are 0.45–0.60. Quote the CI.*
4. How is calibration different from discrimination? *See section 1 and 6.*
5. How would you validate this on real data? *Out-of-time sample (latest vintage), PSI monitoring,
   annual back-testing of grades, override analysis.*
6. How could AI help model calibration? *Automating binning search and monotonicity checks, flagging
   drift, generating documentation drafts; with human review and the same validation gates as in
   my agent-evaluation project.*
7. What is the difference between PIT and TTC PDs? *Point-in-time PDs move with the economic cycle;
   through-the-cycle PDs (agency ratings) target a long-run average, which is what the central-tendency
   calibration approximates.*
8. How does PD relate to Structured Finance? *Loan-level PDs (and LGDs) of the pool feed cash-flow
   models that determine the expected loss on each tranche; rating methodologies then map tranche EL
   to ratings.*
