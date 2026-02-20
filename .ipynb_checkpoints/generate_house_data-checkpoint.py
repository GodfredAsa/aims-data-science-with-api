"""
Generate synthetic house price dataset (~1000 records) with null values.
Saves to data/house_prices.csv. Run: python generate_house_data.py
Uses only standard library so no numpy/pandas required for generation.
"""
import csv
import os
import random

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "house_prices.csv")
N_ROWS = 1000
SEED = 42

# Location premiums (name -> add to price)
LOCATIONS = ["Downtown", "Suburb", "Rural", "Midtown", "Waterfront"]
LOCATION_EFFECT = [0, 30000, 50000, 20000, 80000]


def main():
    random.seed(SEED)
    rows = []
    for i in range(N_ROWS):
        area_sqm = round(random.uniform(45, 350), 1)
        bedrooms = random.randint(1, 6)
        bathrooms = min(random.randint(1, 4), bedrooms + 1)
        age_years = random.randint(0, 79)
        loc_idx = random.randint(0, len(LOCATIONS) - 1)
        location = LOCATIONS[loc_idx]
        has_garage = random.randint(0, 1)
        has_garden = random.randint(0, 1)
        near_school = random.randint(0, 1)
        price_base = (
            2000 * area_sqm
            + 15000 * bedrooms
            + 10000 * bathrooms
            - 500 * age_years
            + LOCATION_EFFECT[loc_idx]
            + 25000 * has_garage
            + 15000 * has_garden
            + 10000 * near_school
        )
        price = int(max(80_000, min(1_200_000, price_base + random.gauss(0, 30000))))
        rows.append({
            "area_sqm": area_sqm,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age_years,
            "location": location,
            "has_garage": has_garage,
            "has_garden": has_garden,
            "near_school": near_school,
            "price": price,
        })

    # Inject nulls
    null_plan = [
        ("area_sqm", 0.04),
        ("bedrooms", 0.03),
        ("bathrooms", 0.05),
        ("age_years", 0.06),
        ("location", 0.02),
        ("price", 0.01),
    ]
    indices = list(range(N_ROWS))
    for col, pct in null_plan:
        k = int(N_ROWS * pct)
        for idx in random.sample(indices, k):
            rows[idx][col] = ""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fieldnames = ["area_sqm", "bedrooms", "bathrooms", "age_years", "location",
                  "has_garage", "has_garden", "near_school", "price"]
    with open(OUTPUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {N_ROWS} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
