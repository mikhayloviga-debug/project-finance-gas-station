from pathlib import Path
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

KEY_RATE = 0.15
DISCOUNT_RATE = 0.18
CAPEX = 18_000_000
DAILY_VOLUME = 3000
RETAIL_PRICE = 69.88
MARGIN_PER_LITER = 8.0
ANNUAL_OPEX = 6_000_000
YEARS = 10


def build_projection():
    annual_liters = DAILY_VOLUME * 365
    annual_revenue = annual_liters * RETAIL_PRICE
    annual_gross_profit = annual_liters * MARGIN_PER_LITER
    annual_ebitda = annual_gross_profit - ANNUAL_OPEX
    cash_flows = [-CAPEX] + [annual_ebitda] * YEARS
    discounted = [cf / ((1 + DISCOUNT_RATE) ** i) for i, cf in enumerate(cash_flows)]

    projection = pd.DataFrame({
        "year": list(range(0, YEARS + 1)),
        "daily_volume_liters": [np.nan] + [DAILY_VOLUME] * YEARS,
        "annual_volume_liters": [np.nan] + [annual_liters] * YEARS,
        "retail_price_rub_per_liter": [np.nan] + [RETAIL_PRICE] * YEARS,
        "revenue_rub": [0] + [annual_revenue] * YEARS,
        "gross_profit_rub": [0] + [annual_gross_profit] * YEARS,
        "opex_rub": [0] + [ANNUAL_OPEX] * YEARS,
        "ebitda_rub": [-CAPEX] + [annual_ebitda] * YEARS,
        "cash_flow_rub": cash_flows,
        "discounted_cash_flow_rub": discounted,
    })
    projection["cumulative_cash_flow_rub"] = projection["cash_flow_rub"].cumsum()
    projection["cumulative_discounted_cash_flow_rub"] = projection["discounted_cash_flow_rub"].cumsum()

    metrics = {
        "annual_liters": annual_liters,
        "annual_revenue_rub": annual_revenue,
        "annual_gross_profit_rub": annual_gross_profit,
        "annual_ebitda_rub": annual_ebitda,
        "npv_rub": npf.npv(DISCOUNT_RATE, cash_flows),
        "irr": npf.irr(cash_flows),
        "simple_payback_years": CAPEX / annual_ebitda if annual_ebitda > 0 else np.nan,
    }
    return projection, metrics


def sensitivity_by_volume():
    rows = []
    for volume in range(1500, 5001, 250):
        annual_ebitda = volume * 365 * MARGIN_PER_LITER - ANNUAL_OPEX
        cash_flows = [-CAPEX] + [annual_ebitda] * YEARS
        rows.append({
            "daily_volume_liters": volume,
            "annual_ebitda_rub": annual_ebitda,
            "npv_rub": npf.npv(DISCOUNT_RATE, cash_flows),
            "irr": npf.irr(cash_flows),
        })
    return pd.DataFrame(rows)


def sensitivity_by_margin():
    rows = []
    for margin in np.arange(5, 11.5, 0.5):
        annual_ebitda = DAILY_VOLUME * 365 * margin - ANNUAL_OPEX
        cash_flows = [-CAPEX] + [annual_ebitda] * YEARS
        rows.append({
            "margin_rub_per_liter": margin,
            "annual_ebitda_rub": annual_ebitda,
            "npv_rub": npf.npv(DISCOUNT_RATE, cash_flows),
            "irr": npf.irr(cash_flows),
        })
    return pd.DataFrame(rows)


def save_charts(projection, volume_sensitivity, margin_sensitivity):
    plt.figure(figsize=(10, 5))
    plt.plot(projection["year"], projection["cumulative_cash_flow_rub"] / 1e6, marker="o", label="Cumulative cash flow")
    plt.plot(projection["year"], projection["cumulative_discounted_cash_flow_rub"] / 1e6, marker="o", label="Cumulative discounted cash flow")
    plt.axhline(0, linewidth=1)
    plt.title("Cumulative project cash flow")
    plt.xlabel("Year")
    plt.ylabel("RUB million")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cash_flow_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(volume_sensitivity["daily_volume_liters"], volume_sensitivity["npv_rub"] / 1e6, marker="o")
    plt.axhline(0, linewidth=1)
    plt.title("NPV sensitivity to daily throughput")
    plt.xlabel("Daily volume, liters")
    plt.ylabel("NPV, RUB million")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sensitivity_volume.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(margin_sensitivity["margin_rub_per_liter"], margin_sensitivity["npv_rub"] / 1e6, marker="o")
    plt.axhline(0, linewidth=1)
    plt.title("NPV sensitivity to gross margin per liter")
    plt.xlabel("Margin, RUB per liter")
    plt.ylabel("NPV, RUB million")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sensitivity_margin.png", dpi=160)
    plt.close()


def main():
    projection, metrics = build_projection()
    volume_sensitivity = sensitivity_by_volume()
    margin_sensitivity = sensitivity_by_margin()

    projection.to_csv(DATA_DIR / "projection_base.csv", index=False)
    volume_sensitivity.to_csv(DATA_DIR / "sensitivity_volume.csv", index=False)
    margin_sensitivity.to_csv(DATA_DIR / "sensitivity_margin.csv", index=False)
    save_charts(projection, volume_sensitivity, margin_sensitivity)

    print("=== Base case metrics ===")
    for key, value in metrics.items():
        if "irr" in key:
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value:,.2f}")


if __name__ == "__main__":
    main()
