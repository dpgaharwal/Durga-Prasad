"""
Track A/D - Deployment subsample generator.

The full IEEE-CIS dataset (~600MB) can't be committed to git or deployed to
Railway's free tier. This script (run LOCALLY, where you have the real data)
produces a small stratified subsample -- same 51-feature schema, same ~3.5%
fraud rate -- small enough to commit directly to the repo (a few MB).

The deployed backend checks for this file first, before the full ieee_data/
folder or the synthetic fallback -- see _load_data() in web/backend/main.py.

Run once, locally:
    cd generate/track_a_transactions
    python make_deploy_subsample.py
"""

import pandas as pd
from real_data_loader import load_real_data

N_LEGIT = 8000
N_FRAUD = 300   # roughly preserves the real ~3.5% fraud rate at this scale


def main():
    print("Loading full real IEEE-CIS dataset (this reads the large local files)...")
    df = load_real_data("ieee_data")

    legit = df[df.isFraud == 0].sample(n=N_LEGIT, random_state=7)
    fraud = df[df.isFraud == 1].sample(n=min(N_FRAUD, (df.isFraud == 1).sum()), random_state=7)

    subsample = pd.concat([legit, fraud], ignore_index=True).sample(frac=1.0, random_state=7)
    subsample.to_csv("deploy_subsample.csv", index=False)

    print(f"Wrote deploy_subsample.csv: {len(subsample)} rows "
          f"({len(fraud)} fraud, {len(fraud)/len(subsample):.2%})")
    print("Commit this file to the repo -- it's small enough for git, "
          "unlike the raw ieee_data/ folder which stays gitignored.")


if __name__ == "__main__":
    main()
