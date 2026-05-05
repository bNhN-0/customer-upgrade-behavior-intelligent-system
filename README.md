<h1 align="center">Customer Upgrade Behavior Intelligent System</h1>

<p align="center">
  <strong>ABM-based CRM decision-support system for customer upgrade behavior, churn risk, and retention automation.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red" />
  <img src="https://img.shields.io/badge/Model-Agent--Based%20Modeling-green" />
  <img src="https://img.shields.io/badge/Storage-Firestore-orange" />
</p>

Customer Upgrade Behavior Intelligent System is a Streamlit-based decision-support system that analyzes customer device-upgrade behavior and supports CRM automation.

The system uses customer and device signals such as device age, battery condition, trade-in value, engagement, past upgrades, social influence, and price sensitivity to estimate upgrade readiness, delay behavior, and churn risk.

It also assigns each customer a behavioral persona and recommends CRM follow-up actions, helping turn customer behavior signals into automated retention and upgrade decisions.

---

## Overview

- Streamlit dashboard for analyzing customer device-upgrade behavior.
- Uses an Agent-Based Modeling (ABM) structure where each customer is treated as an individual agent.
- Estimates upgrade readiness, delay behavior, and churn risk.
- Assigns customers into four personas: `Loyalist`, `Fan`, `Switcher`, and `Drifter`.
- Recommends CRM actions based on persona and predicted outcome.
- Supports CSV upload, customer exploration, persona analysis, and optional Firestore storage.

---

## Dataset

Default dataset file:

- `customer_upgrade_behavior.csv`

Required input columns:

| Column | Meaning |
|---|---|
| `id` | Customer identifier |
| `DA` | Device Age |
| `BH` | Battery Health / Battery Degradation Impact |
| `TI` | Trade-In Value |
| `ENG` | Engagement |
| `PU` | Past Upgrades |
| `SI` | Social Influence |
| `PS` | Price Sensitivity |

Each input is scaled between `0` and `1`.

The included dataset is synthetic and used for demonstration and testing.

---

## Model Workflow

```text
1. Load customer records from CSV or Firestore
2. Compute behavioral states
   ├── Need
   ├── Bonding
   └── Hesitation

3. Compute persona states
   ├── Loyalist
   ├── Fan
   ├── Switcher
   └── Drifter

4. Derive system behavior
   ├── Commitment
   ├── Volatility
   └── ShortTermIntent

5. Update long-term upgrade readiness
6. Classify customer outcome
   ├── Upgrade Soon
   ├── Delay Upgrade
   └── Churn Risk

7. Recommend CRM actions
```

---

## Model Architecture

The model is built using an Agent-Based Modeling (ABM) structure. Each customer record represents an agent, and each agent contains internal behavioral states that are updated through layered equations.

  ![Model Architecture](/model.png)


---

## Model Equations

### Behavioral State Layer

```text
N = (1 - PS) × [αN(DA + BH) + (1 - αN){βN(TI + PU + SI) + (1 - βN)λN ENG}]
```

```text
B = (1 - PS) × [αB ENG + (1 - αB){βB(PU + SI) + (1 - βB)λB TI}]
```

```text
H = PS × [1 - {αH TI + (1 - αH)[βH(DA + BH + PU) + (1 - βH)λH(ENG + SI)]}]
```

### Persona Layer

```text
Y1 = [1 - H] × [α1B + (1 - α1)N]
```

```text
Y2 = [1 - H] × [α2B + (1 - α2)N]
```

```text
Y3 = [α3H + (1 - α3)N] × (1 - B)
```

```text
Y4 = H × [α4(1 - B) + (1 - α4)N]
```

| Symbol | Persona |
|---|---|
| `Y1` | Loyalist |
| `Y2` | Fan |
| `Y3` | Switcher |
| `Y4` | Drifter |

### System Behavior Layer

```text
C = [1 - {δC Y4 + (1 - δC)Y3}] × [αC Y1 + (1 - αC)Y2]
```

```text
V = [αV Y3 + (1 - αV)Y4] × [1 - {δV Y1 + (1 - δV)Y2}]
```

### Output Layer

```text
S = C × (1 - V)
```

```text
LUI(t + Δt) = LUI(t) + η[S(t) - LUI(t)]Δt
```

---

## Prediction Outputs

Each customer receives:

| Output | Description |
|---|---|
| Dominant persona | Main behavioral persona assigned to the customer |
| Persona score weights | Relative score distribution across personas |
| Behavioral state values | Need, Bonding, and Hesitation |
| System behavior values | Commitment and Volatility |
| Short-term upgrade intent | Immediate upgrade-readiness signal |
| Long-term upgrade readiness | Dynamic readiness score over time |
| Predicted outcome | `Upgrade Soon`, `Delay Upgrade`, or `Churn Risk` |

---

## CRM Action Routing

The dashboard maps persona and predicted outcome to CRM actions.

Examples:

| Segment Type | Example CRM Action |
|---|---|
| Loyal high-intent customers | Upgrade offers |
| Interested customers | Bundle or feature messaging |
| Switcher-type customers | Comparison or trade-in messaging |
| Churn-risk customers | Win-back messaging |
| Low-readiness segments | Lower marketing spend |

---

## Dashboard Features

- Segment overview metrics
- Long-term upgrade readiness distribution
- Prediction breakdown charts
- Persona distribution
- Persona-level averages
- Behavioral profile comparison
- CRM action summary
- CSV export
- Customer Explorer
- Persona radar chart
- CSV upload
- Optional Firestore storage

---

## Tech Stack

| Area | Tools |
|---|---|
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Storage | Firebase Firestore |
| Language | Python |

---

## How To Run

1. Create and activate a virtual environment.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

4. Open the local Streamlit URL shown in the terminal.

---

## Firestore Support

Firestore is optional.

If `.streamlit/secrets.toml` contains Firebase credentials, the app can save and reload computed prediction records from the `customer_upgrade_predictions` collection.

Expected structure:

```text
project-folder/
├── app.py
├── customer_upgrade_behavior.csv
├── requirements.txt
└── .streamlit/
    └── secrets.toml
```

---

## Disclaimer

This project uses synthetic customer data for a fictional mobile-device company scenario. It does not use real customer records, proprietary company data, or any official brand-owned dataset.
