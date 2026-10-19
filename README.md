# NARUTO | MONTE CARLO LAB

1. Create a **private** GitHub repository (if private deployment is unsupported by your chosen hosting tier, use a public repository containing **only this generic code**, no account or trade data).
2. Upload `app.py` and `requirements.txt` to repository root.
3. Go to https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app** → **Deploy a public app from GitHub** → select your repository, branch `main`, main file `app.py`.
5. Deploy. Open the app, change parameters, and press **Run Monte Carlo**.
6. The app displays UP TO DATE only after a successful run using the current parameters; changed inputs show OUTDATED until rerun.

Local test (optional): `pip install -r requirements.txt` then `streamlit run app.py`.

Security: Do not upload API keys, passwords, personal account records, or trading journal exports to a public GitHub repository.

Model limitations: independent Bernoulli outcomes, fixed RR and risk, no fees/slippage/liquidation; not a backtest.
