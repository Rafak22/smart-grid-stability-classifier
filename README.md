# Smart Grid Stability Classifier

## Predicting Electrical Grid Stability Using Machine Learning

**Data Source:** UCI Machine Learning Repository — Electrical Grid Stability Simulated Data  
**Dataset URL:** [Data_for_UCI_named.csv](https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data)  
**Author:** Rafa Alshareef 
**Aligned with:** Saudi Vision 2030 — Renewable Energy and Smart Grid Modernization

---

## State the Goal

### Current Non-ML Solution

Grid operators today rely on two primary non-ML approaches to manage electrical grid stability:

**1. SCADA Systems (Supervisory Control and Data Acquisition)**  
SCADA is the dominant technology used by electrical utilities worldwide, including the Saudi Electricity Company (SEC). It continuously collects sensor readings from across the grid — voltage levels, frequency, current flow, and load — and transmits them to a centralized control room. When a reading crosses a pre-set threshold, an alarm is triggered and a human operator manually intervenes by rerouting power, switching circuit breakers, or dispatching field engineers.

The fundamental limitation of SCADA is that it is entirely reactive. It detects a problem only after it has already begun. By the time an alarm fires, a human operator notices, and a corrective command is issued, the grid may have already entered an unstable state. In complex multi-node grids with renewable energy sources introducing additional variability, this reaction gap can be the difference between a minor adjustment and a cascading regional blackout.

**2. Physics-Based Simulation Tools (PSS/E, PowerWorld)**  
Power systems engineers use specialized simulation software to model grid behavior under different load and generation scenarios. These simulations are run periodically — typically monthly or quarterly — to identify vulnerabilities in the grid configuration and to update protection relay settings. While these tools are highly accurate within their scenarios, they are static planning instruments. They cannot assess the stability of the grid as it exists right now, under real-time conditions.

**Limitations of Current Approaches:**
- SCADA reacts to instability after it begins, not before
- Fixed alarm thresholds cannot account for complex multi-parameter interactions across nodes
- A grid can approach instability through a combination of moderate readings that individually appear safe but collectively are dangerous — SCADA cannot detect this
- Physics-based simulations are periodic, not real-time, and require specialized engineering expertise to interpret
- Neither approach provides a probability score that quantifies how close the grid is to instability at any given moment

---

### Application, Goal and Description

**Application:** A real-time grid stability prediction system for smart grid operators, energy engineers, and infrastructure planners at electrical utilities.

**Goal:** Given the current state of a 4-node decentralized smart grid — described by the reaction times and power consumption or production rates of each node — predict whether the grid configuration is stable or unstable.

**Description:** This model is trained on 10,000 simulated observations of a 4-node star topology smart grid, where one central electricity producer is connected to three consumer nodes. Each observation captures the reaction time (tau) and power coefficient (p) of all four nodes, along with the price elasticity coefficient (g) of each node. The dataset was generated using physics-based simulation methodology aligned with the Decentral Smart Grid Control (DSGC) concept. The model learns which combinations of these 12 input parameters lead to stable versus unstable grid states, enabling proactive identification of dangerous configurations before physical consequences occur.

---

### ML Task

- **Task Type:** Binary Classification
- **Target Variable:** stabf — stable or unstable
- **Input Features:** 12 numerical grid parameters (tau1–tau4, p1–p4, g1–g4)
- **Output:** Predicted stability label (stable or unstable) with associated probability score

---

## 1. Clear Use Case for ML

| Dimension | Without ML (SCADA) | With ML |
|---|---|---|
| Difference | Detects instability after it starts based on fixed thresholds | Predicts instability before it occurs based on the combined state of all 12 parameters simultaneously |
| Cost | Requires 24/7 staffed control rooms, senior operators, and expensive periodic engineering studies | One-time model training cost and near-zero inference cost per prediction once deployed |
| Maintenance | Threshold settings must be manually reviewed and updated by engineers as grid topology changes | Model is retrained when new simulation data is generated reflecting updated grid configurations |
| Expertise | Requires experienced power systems engineers to interpret alarms and simulation outputs | Any grid operator can receive a plain stable or unstable prediction with confidence score through an API or dashboard |

### Why This Problem Specifically Requires Machine Learning

The 12 input parameters interact in non-linear ways. A reaction time of tau1 that is safe under one combination of p and g values may be dangerous under another. Fixed threshold systems and simple rule-based logic cannot capture these interactions. Machine learning identifies complex boundaries in the 12-dimensional input space that separate stable from unstable configurations — boundaries that no human engineer could practically define manually.

Additionally, as smart grids integrate more renewable energy sources, the variability and interdependence of node parameters increases. The pattern recognition capability of ML becomes more valuable, not less, as grid complexity grows.

---

## 2. Does ART Apply to the Data?

### Available

All 12 input features required at prediction time are available in real-time in any modern smart grid deployment. Reaction times (tau1–tau4) and power coefficients (p1–p4) are continuously measured by smart meters and PMUs (Phasor Measurement Units). Price elasticity coefficients (g1–g4) are system-level parameters known from grid configuration. No feature requires future information or privileged access that would be unavailable during live inference. The model can receive these 12 values and return a prediction in milliseconds.

### Representative

The dataset contains 10,000 simulated observations covering a wide and diverse range of parameter combinations across all four nodes. The simulation methodology is based on published research in the European Physical Journal Special Topics and reflects the behavior of the Decentral Smart Grid Control (DSGC) concept, a real and actively researched approach to smart grid management. The target variable reflects genuine physical stability outcomes derived from the underlying differential equations of the grid system, not arbitrary labels. The class distribution is 63.8% unstable and 36.2% stable, which reflects the reality that many grid configurations are inherently dangerous — especially those with asymmetric reaction times or imbalanced power flows.

**Limitation:** The data is simulated rather than collected from a live operational grid. While the simulation is physics-grounded, real-world grids introduce additional noise, measurement error, and environmental variables not captured in simulation. A production deployment would require validation against real operational data.

### Trusted

The dataset was donated to the UCI Machine Learning Repository by Vadim Arzamasov of the Karlsruhe Institute of Technology and is published under a Creative Commons Attribution 4.0 International license. The simulation methodology is documented and peer-reviewed, referenced in a published journal article. It is not user-generated data, not scraped from informal sources, and not the output of another ML system. The physics-based generation process means the labels are ground truth outcomes derived from simulation equations rather than human annotation, making them highly reliable.

---

## 3. What is the Quantity and Quality of the Data?

**Quantity:**
- Total observations: 10,000 rows
- Total features: 14 columns (12 input features, 1 continuous stability score, 1 binary target)
- Usable for classification: 10,000 rows with the binary target stabf
- No missing values in any column
- Train / test split: 80% training (8,000 rows), 20% test (2,000 rows)

**Quality:**
- All features are continuous numerical values with no categorical encoding required
- No missing values anywhere in the dataset
- No duplicate rows
- All feature values fall within physically meaningful ranges
- The continuous stability score (stab) is available as an additional column but is excluded from model input to prevent data leakage — it is the numerical form of the target variable itself
- Class imbalance is moderate: 63.8% unstable vs 36.2% stable. This is addressed using class_weight='balanced' in model training

**Feature Ranges:**
- tau1 to tau4 (reaction times): 0.5 to 10.0 seconds
- p1 to p4 (power coefficients): -2.0 to 2.0 (negative indicates production, positive indicates consumption)
- g1 to g4 (price elasticity): 0.05 to 1.0

---

## 4. What Features Have Been Engineered?

The original dataset contains 12 raw physical measurements across 4 nodes. 
The following 6 features are engineered to capture higher-level patterns 
that the raw values alone cannot express:

**1. total_reaction_time**  
Sum of all four node reaction times (tau1 + tau2 + tau3 + tau4). Even if no 
single tau value looks alarming individually, a high total reveals that the 
entire system is sluggish as a whole. A slow system cannot self-correct before 
instability occurs.

**2. reaction_time_variance**  
Variance across the four tau values. High variance means nodes are responding 
at very different speeds — the producer may be reacting 9 times slower than 
the consumers. This asynchronous behavior prevents coordination and 
destabilizes the grid even when the average reaction time appears acceptable.

**3. producer_consumer_ratio**  
Ratio of the producer node reaction time (tau1) to the average consumer 
reaction time ((tau2 + tau3 + tau4) / 3). The producer is the center of the 
star topology — if it reacts significantly slower than the consumers it 
supplies, the entire grid suffers. A ratio above 3.0 is a strong instability 
signal.

**4. avg_price_elasticity**  
Mean of all four price elasticity coefficients (g1 + g2 + g3 + g4) / 4. 
Measures the overall responsiveness of the grid to price signals. When this 
average is low, the self-regulating mechanism of the Decentral Smart Grid 
Control system breaks down — price changes but no node adjusts its behavior, 
making balance impossible to restore.

**5. net_power_balance**  
Sum of all four power coefficients (p1 + p2 + p3 + p4). In a perfectly 
balanced grid this value is zero because the producer supplies exactly what 
the consumers demand combined. Large deviations from zero indicate that 
supply and demand are mismatched, placing stress on the system that grows 
over time if not corrected.

**6. tau1_x_g1**  
Interaction term between the producer reaction time and producer price 
elasticity, calculated as tau1 * (1 - g1). This is the single most powerful 
engineered feature. It captures the combined risk of a producer that is both 
slow to react AND unresponsive to price signals simultaneously. A producer 
with high tau1 and low g1 is the most dangerous configuration in the dataset 
because it is the node all three consumers depend on.

Example:
- tau1 = 9.5, g1 = 0.06 → tau1_x_g1 = 9.5 * 0.94 = 8.93 (extreme danger)
- tau1 = 1.5, g1 = 0.95 → tau1_x_g1 = 1.5 * 0.05 = 0.075 (very safe)

---

## 5. Which Features Have the Most Predictive Power?

Based on feature importance analysis from the trained Random Forest model:

**Highest importance:**
- **tau1** — The producer node's reaction time is the single most predictive feature. When the central producer responds slowly to demand changes, the grid is significantly more likely to become unstable.
- **reaction_time_variance** — Asynchronous response across nodes is a strong instability signal.
- **net_power_balance** — Large deviations from zero power balance strongly predict instability.

**Medium importance:**
- **tau2, tau3, tau4** — Individual consumer reaction times each contribute meaningful signal.
- **total_reaction_time** — Aggregate system sluggishness is predictive beyond individual node values.
- **p1** — The producer power coefficient interacts strongly with tau1.

**Lower importance:**
- **g1 to g4** — Price elasticity coefficients add marginal value individually, but avg_price_elasticity captures the pattern more efficiently.
- **max_consumer_load** — Informative but partially redundant with net_power_balance.

---

## 6. What is the Prediction of the Model and How is the Decision Based on It?

### What the Model Predicts

Given the 12 grid parameters (plus engineered features) for a current grid configuration, the model outputs:
1. A binary label: **stable** or **unstable**
2. A probability score between 0 and 1 representing confidence in the unstable prediction

### How Decisions Are Based on the Output

The model output is used by grid operators as follows:

- **Unstable prediction with probability above 0.80:** Immediate automated protective action is triggered — load shedding, rerouting, or islanding of affected nodes. The operator is alerted with a high-priority alarm.

- **Unstable prediction with probability between 0.60 and 0.80:** A warning is issued to the operator. Preventive measures are prepared but not yet executed. The configuration is monitored at increased frequency.

- **Stable prediction with probability above 0.75:** Normal grid operation continues. No intervention required.

- **Stable prediction with probability below 0.75:** The configuration is flagged for monitoring. The model's uncertainty indicates a borderline configuration that warrants attention.

This probability-based decision structure allows operators to calibrate their response to the degree of risk rather than treating every prediction as an equal binary alert, which is a significant improvement over fixed-threshold SCADA alarms.

---

## 7. What are the Model's Metrics?

The following metrics are used to evaluate model performance:

| Metric | Description | Reason for Inclusion |
|---|---|---|
| Accuracy | Percentage of correct predictions | Overall performance baseline |
| F1 Score (Weighted) | Harmonic mean of precision and recall, weighted by class frequency | Primary metric — accounts for class imbalance between stable and unstable |
| Precision | Of all predicted unstable, how many were actually unstable | Measures false alarm rate — important for operator trust |
| Recall (Unstable class) | Of all actually unstable configurations, how many did the model catch | Critical — a missed unstable prediction is more dangerous than a false alarm |
| ROC-AUC | Area under the ROC curve | Measures discrimination ability across all probability thresholds |
| Cross-Validation F1 (5-fold) | Stable estimate of generalization across data splits | Essential for validating model performance on a finite dataset |

### Why Recall on the Unstable Class is Critical

In power grid applications, a false negative (predicting stable when the grid is actually unstable) is far more dangerous than a false positive (predicting unstable when the grid is actually stable). A false negative means the grid reaches an unstable state with no warning. A false positive means an unnecessary but harmless intervention. The model is therefore optimized to prioritize high recall on the unstable class, even at the cost of some precision.

---

## 8. What are the Success and Failure Criteria?

| Criterion | Success Threshold | Failure Threshold | Rationale |
|---|---|---|---|
| CV F1 Weighted (5-fold) | >= 0.90 | < 0.75 | High bar justified by 10,000 clean rows and clear class separation in physics-based data |
| Test Accuracy | >= 0.88 | < 0.75 | Model must significantly outperform majority-class baseline of 63.8% |
| Test F1 Weighted | >= 0.88 | < 0.75 | Primary metric accounting for class imbalance |
| Unstable Class Recall | >= 0.90 | < 0.80 | Missing an unstable configuration is unacceptable in grid safety contexts |
| ROC-AUC | >= 0.92 | < 0.80 | Strong discrimination across all decision thresholds required |
| Best model beats baseline | Must beat naive majority-class classifier (63.8% accuracy) by at least 20 percentage points | Fails if improvement is marginal | Confirms ML adds real safety value over doing nothing |

### Baseline Reference

A naive classifier that always predicts the majority class (unstable) would achieve 63.8% accuracy but would provide zero useful information — it could never predict a stable configuration. The model must substantially exceed this baseline across all metrics to justify deployment.

### Corrective Actions if Criteria Are Not Met

If the success criteria are not achieved, the following steps will be taken in order:

1. Apply SMOTE oversampling to the training set to address class imbalance more aggressively
2. Perform hyperparameter tuning using GridSearchCV with cross-validation
3. Engineer additional interaction features between tau and p values for each node
4. Evaluate ensemble stacking of Random Forest and XGBoost predictions
5. Re-examine the stab continuous variable for potential feature engineering as a proxy signal

---

## Usage

See the notebook `saudi_grid_stability_classifier.ipynb` for the complete implementation including data loading, feature engineering, model training, evaluation, and dashboard visualization.

### Setup

```bash
pip install -r requirements.txt
jupyter notebook saudi_grid_stability_classifier.ipynb
```

### Running the API

```bash
uvicorn api:app --reload
```

### Running the Gradio App

```bash
python app.py
```
