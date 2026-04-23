# Gas Station Investment Analysis — Full Project Finance Model

## Overview
This repository contains a Python-based project finance model for a greenfield gas station in Saint Petersburg.

The project started with a simple unlevered valuation model and was upgraded to a fuller credit-style model with:
- operating projection for 10 years,
- debt schedule with annuity repayment,
- taxes,
- maintenance capex,
- working capital,
- salvage value,
- project IRR / NPV,
- equity IRR / NPV,
- DSCR analysis,
- downside / base / upside scenarios,
- debt-capacity sensitivity by throughput.

## Folder structure
```text
project_finance_gas_station/
├── data/
├── notebooks/
├── outputs/
├── src/
│   ├── model.py
│   └── model_full.py
├── README.md
└── requirements.txt
```

## Market-backed inputs used in the model
These inputs are tied to current public data and then translated into model assumptions:
- Bank of Russia key rate: 15.0%
- Saint Petersburg AI-95 retail price used as a reference anchor: 69.88 RUB/liter

## Core model assumptions
### Operating
- CAPEX: 18.0m RUB
- CAPEX contingency: 5%
- Daily throughput: 3,000 liters
- Gross margin: 8.0 RUB/liter
- Annual OPEX (year 1): 6.0m RUB
- Revenue growth: 3%
- Inflation: 6%
- Project life: 10 years

### Financing
- Debt share: 70%
- Equity share: 30%
- Debt rate: 20%
- Debt tenor: 5 years
- Discount rate: 18%

### Additional full-model assumptions
- Tax rate: 20%
- Maintenance capex: 0.5% of revenue
- Working capital: 7 revenue days
- Salvage value: 10% of CAPEX in the terminal year

## Outputs
Running `python src/model_full.py` creates:
- `data/projection_full_model.csv`
- `data/debt_schedule.csv`
- `data/scenario_summary.csv`
- `data/debt_sensitivity_volume.csv`
- charts in the `outputs/` folder

## How to run
```bash
python -m pip install -r requirements.txt
python src/model_full.py
```

## What this project demonstrates
- project finance logic,
- financial modeling in Python,
- debt-service analysis,
- scenario analysis,
- business interpretation of model outputs.

## Suggested interview story
“I built a project finance model for a gas station investment. I started from a simple NPV / IRR model and then expanded it into a fuller financing case with debt, taxes, maintenance capex, working capital, DSCR, and scenario analysis. The model showed that under base assumptions the project does not clear the target return and struggles to support debt comfortably, which led to a discussion of the key operational levers needed to make the project bankable.”
