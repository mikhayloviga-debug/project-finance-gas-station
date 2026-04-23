
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import numpy_financial as npf
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class Assumptions:
    # Market-backed reference assumptions
    key_rate: float = 0.15
    retail_price_rub_per_liter: float = 69.88

    # Model assumptions
    discount_rate: float = 0.18
    capex_rub: float = 18_000_000
    daily_volume_liters: int = 3000
    margin_rub_per_liter: float = 8.0
    annual_opex_rub: float = 6_000_000
    project_years: int = 10

    # Debt assumptions
    debt_share: float = 0.70
    debt_rate: float = 0.20
    debt_years: int = 5

    # Full-model assumptions
    tax_rate: float = 0.20
    inflation_rate: float = 0.06
    revenue_growth_rate: float = 0.03
    capex_contingency_share: float = 0.05
    maintenance_capex_share_of_revenue: float = 0.005
    working_capital_days_of_revenue: int = 7
    salvage_value_share_of_capex: float = 0.10


def annuity_payment(principal: float, rate: float, periods: int) -> float:
    return float(npf.pmt(rate, periods, -principal))


def build_operating_projection(a: Assumptions) -> pd.DataFrame:
    rows = []
    base_annual_liters = a.daily_volume_liters * 365
    opening_wc = (base_annual_liters * a.retail_price_rub_per_liter / 365) * a.working_capital_days_of_revenue

    rows.append({
        "year": 0,
        "annual_volume_liters": 0.0,
        "retail_price_rub_per_liter": 0.0,
        "revenue_rub": 0.0,
        "gross_profit_rub": 0.0,
        "opex_rub": 0.0,
        "ebitda_rub": 0.0,
        "maintenance_capex_rub": 0.0,
        "change_in_working_capital_rub": opening_wc,
        "tax_rub": 0.0,
        "unlevered_free_cash_flow_rub": -(a.capex_rub * (1 + a.capex_contingency_share) + opening_wc),
    })

    prev_revenue = None
    for year in range(1, a.project_years + 1):
        volume = base_annual_liters * ((1 + a.revenue_growth_rate) ** (year - 1))
        retail_price = a.retail_price_rub_per_liter * ((1 + a.inflation_rate) ** (year - 1))
        margin = a.margin_rub_per_liter * ((1 + a.inflation_rate) ** (year - 1))
        revenue = volume * retail_price
        gross_profit = volume * margin
        opex = a.annual_opex_rub * ((1 + a.inflation_rate) ** (year - 1))
        ebitda = gross_profit - opex
        maintenance_capex = revenue * a.maintenance_capex_share_of_revenue
        ebit = ebitda - maintenance_capex
        tax = max(ebit, 0) * a.tax_rate

        wc_required = revenue / 365 * a.working_capital_days_of_revenue
        if prev_revenue is None:
            delta_wc = 0.0
        else:
            prev_wc = prev_revenue / 365 * a.working_capital_days_of_revenue
            delta_wc = wc_required - prev_wc

        salvage = a.capex_rub * a.salvage_value_share_of_capex if year == a.project_years else 0.0
        wc_release = wc_required if year == a.project_years else 0.0

        ufcf = ebitda - maintenance_capex - tax - delta_wc + salvage + wc_release

        rows.append({
            "year": year,
            "annual_volume_liters": volume,
            "retail_price_rub_per_liter": retail_price,
            "revenue_rub": revenue,
            "gross_profit_rub": gross_profit,
            "opex_rub": opex,
            "ebitda_rub": ebitda,
            "maintenance_capex_rub": maintenance_capex,
            "change_in_working_capital_rub": delta_wc,
            "tax_rub": tax,
            "salvage_value_rub": salvage,
            "working_capital_release_rub": wc_release,
            "unlevered_free_cash_flow_rub": ufcf,
        })
        prev_revenue = revenue

    projection = pd.DataFrame(rows)
    projection["discount_factor"] = 1 / ((1 + a.discount_rate) ** projection["year"])
    projection["discounted_ufcf_rub"] = projection["unlevered_free_cash_flow_rub"] * projection["discount_factor"]
    projection["cumulative_ufcf_rub"] = projection["unlevered_free_cash_flow_rub"].cumsum()
    return projection


def build_debt_schedule(a: Assumptions) -> pd.DataFrame:
    principal = a.capex_rub * (1 + a.capex_contingency_share) * a.debt_share
    payment = annuity_payment(principal, a.debt_rate, a.debt_years)
    balance = principal
    rows = []

    for year in range(1, a.project_years + 1):
        if year <= a.debt_years:
            opening = balance
            interest = opening * a.debt_rate
            principal_repayment = payment - interest
            closing = max(opening - principal_repayment, 0.0)
            debt_service = payment
            balance = closing
        else:
            opening = balance
            interest = 0.0
            principal_repayment = 0.0
            debt_service = 0.0
            closing = balance

        rows.append({
            "year": year,
            "opening_balance_rub": opening,
            "interest_rub": interest,
            "principal_repayment_rub": principal_repayment,
            "debt_service_rub": debt_service,
            "closing_balance_rub": closing,
        })
    return pd.DataFrame(rows)


def build_full_model(a: Assumptions) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    op = build_operating_projection(a)
    debt = build_debt_schedule(a)

    op1 = op.merge(debt, on="year", how="left")
    debt_cols = ["opening_balance_rub", "interest_rub", "principal_repayment_rub", "debt_service_rub", "closing_balance_rub"]
    for c in debt_cols:
        op1[c] = op1[c].fillna(0.0)

    equity_investment = a.capex_rub * (1 + a.capex_contingency_share) * (1 - a.debt_share) + op.loc[0, "change_in_working_capital_rub"]
    op1.loc[0, "equity_cash_flow_rub"] = -equity_investment
    for idx in op1.index[1:]:
        op1.loc[idx, "equity_cash_flow_rub"] = op1.loc[idx, "unlevered_free_cash_flow_rub"] - op1.loc[idx, "debt_service_rub"]

    op1["discounted_equity_cf_rub"] = op1["equity_cash_flow_rub"] * op1["discount_factor"]
    op1["dscr"] = np.where(op1["debt_service_rub"] > 0, op1["ebitda_rub"] / op1["debt_service_rub"], np.nan)

    ufcf = op1["unlevered_free_cash_flow_rub"].tolist()
    eqcf = op1["equity_cash_flow_rub"].tolist()

    min_dscr = float(np.nanmin(op1["dscr"])) if op1["dscr"].notna().any() else np.nan
    avg_dscr = float(np.nanmean(op1["dscr"])) if op1["dscr"].notna().any() else np.nan
    debt_years_mask = op1["debt_service_rub"] > 0

    metrics = {
        "annual_liters_year_1": float(op1.loc[1, "annual_volume_liters"]),
        "annual_revenue_year_1_rub": float(op1.loc[1, "revenue_rub"]),
        "ebitda_year_1_rub": float(op1.loc[1, "ebitda_rub"]),
        "project_npv_rub": float(npf.npv(a.discount_rate, ufcf)),
        "project_irr": float(npf.irr(ufcf)),
        "equity_npv_rub": float(npf.npv(a.discount_rate, eqcf)),
        "equity_irr": float(npf.irr(eqcf)),
        "avg_dscr": avg_dscr,
        "min_dscr": min_dscr,
        "loan_amount_rub": float(a.capex_rub * (1 + a.capex_contingency_share) * a.debt_share),
        "equity_contribution_rub": float(equity_investment),
        "avg_annual_debt_service_rub": float(op1.loc[debt_years_mask, "debt_service_rub"].mean()) if debt_years_mask.any() else 0.0,
    }
    return op1, debt, metrics


def scenario_table(a: Assumptions) -> pd.DataFrame:
    scenarios = {
        "downside": {"daily_volume_liters": int(a.daily_volume_liters * 0.85), "margin_rub_per_liter": a.margin_rub_per_liter - 0.75},
        "base": {"daily_volume_liters": a.daily_volume_liters, "margin_rub_per_liter": a.margin_rub_per_liter},
        "upside": {"daily_volume_liters": int(a.daily_volume_liters * 1.15), "margin_rub_per_liter": a.margin_rub_per_liter + 0.75},
    }

    rows = []
    for name, overrides in scenarios.items():
        s = Assumptions(**{**asdict(a), **overrides})
        model, debt, metrics = build_full_model(s)
        rows.append({
            "scenario": name,
            "daily_volume_liters": s.daily_volume_liters,
            "margin_rub_per_liter": s.margin_rub_per_liter,
            "project_npv_rub": metrics["project_npv_rub"],
            "project_irr": metrics["project_irr"],
            "equity_irr": metrics["equity_irr"],
            "min_dscr": metrics["min_dscr"],
        })
    return pd.DataFrame(rows)


def debt_capacity_sensitivity(a: Assumptions) -> pd.DataFrame:
    rows = []
    for volume in range(2200, 4801, 200):
        s = Assumptions(**{**asdict(a), "daily_volume_liters": volume})
        model, debt, metrics = build_full_model(s)
        rows.append({
            "daily_volume_liters": volume,
            "project_npv_rub": metrics["project_npv_rub"],
            "equity_irr": metrics["equity_irr"],
            "min_dscr": metrics["min_dscr"],
        })
    return pd.DataFrame(rows)


def save_outputs(model: pd.DataFrame, debt: pd.DataFrame, scenarios: pd.DataFrame, sens: pd.DataFrame) -> None:
    model.to_csv(DATA_DIR / "projection_full_model.csv", index=False)
    debt.to_csv(DATA_DIR / "debt_schedule.csv", index=False)
    scenarios.to_csv(DATA_DIR / "scenario_summary.csv", index=False)
    sens.to_csv(DATA_DIR / "debt_sensitivity_volume.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(model["year"], model["cumulative_ufcf_rub"] / 1e6, marker="o")
    plt.axhline(0, linewidth=1)
    plt.title("Cumulative unlevered free cash flow")
    plt.xlabel("Year")
    plt.ylabel("RUB million")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "full_model_cumulative_ufcf.png", dpi=160)
    plt.close()

    debt_only = model[model["year"] >= 1]
    plt.figure(figsize=(10, 5))
    plt.bar(debt_only["year"], debt_only["debt_service_rub"] / 1e6)
    plt.title("Annual debt service")
    plt.xlabel("Year")
    plt.ylabel("RUB million")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "debt_service.png", dpi=160)
    plt.close()

    dscr = debt_only.dropna(subset=["dscr"])
    plt.figure(figsize=(10, 5))
    plt.plot(dscr["year"], dscr["dscr"], marker="o")
    plt.axhline(1.2, linestyle="--", linewidth=1)
    plt.title("DSCR by year")
    plt.xlabel("Year")
    plt.ylabel("DSCR")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dscr_by_year.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(scenarios["scenario"], scenarios["project_npv_rub"] / 1e6)
    plt.axhline(0, linewidth=1)
    plt.title("Scenario NPV comparison")
    plt.xlabel("Scenario")
    plt.ylabel("NPV, RUB million")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scenario_npv.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(sens["daily_volume_liters"], sens["min_dscr"], marker="o")
    plt.axhline(1.2, linestyle="--", linewidth=1)
    plt.title("Minimum DSCR sensitivity to daily volume")
    plt.xlabel("Daily volume, liters")
    plt.ylabel("Minimum DSCR")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "min_dscr_sensitivity_volume.png", dpi=160)
    plt.close()


def main() -> None:
    a = Assumptions()
    assumptions_df = pd.DataFrame([asdict(a)])
    assumptions_df.to_csv(DATA_DIR / "full_model_assumptions.csv", index=False)

    model, debt, metrics = build_full_model(a)
    scenarios = scenario_table(a)
    sens = debt_capacity_sensitivity(a)
    save_outputs(model, debt, scenarios, sens)

    print("=== Full model metrics ===")
    for k, v in metrics.items():
        if "irr" in k:
            print(f"{k}: {v:.2%}")
        else:
            print(f"{k}: {v:,.2f}")


if __name__ == "__main__":
    main()
