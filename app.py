import math
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Naruto | Monte Carlo Lab', page_icon='📈', layout='wide')
st.markdown('''<style>
[data-testid="stMetric"] {
  color: #171717 !important;
}
[data-testid="stMetricLabel"] {
  color: #555047 !important;
}
[data-testid="stMetricValue"] {
  color: #171717 !important;
}
.stButton button {background:#171717;color:white;border-radius:8px;border:0}
</style>''', unsafe_allow_html=True)
st.title('NARUTO | MONTE CARLO LAB')
st.caption('Trading Risk Simulator • Independent random-path engine • Assumptions are not verified trading performance')

with st.sidebar:
    st.header('Simulation Settings')
    capital = st.number_input('Initial Capital (USD)', min_value=1.0, value=500.0, step=100.0)
    risk_pct = st.number_input('Risk per Trade (%)', min_value=0.01, max_value=99.0, value=5.0, step=0.5)
    rr = st.number_input('Risk Reward (1:R)', min_value=0.01, max_value=100.0, value=5.0, step=0.5)
    win_pct = st.number_input('Assumed Win Rate (%)', min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    trades = st.number_input('Trades per Path', min_value=1, max_value=2000, value=200, step=10)
    paths = st.selectbox('Simulation Paths', [1000, 5000, 10000, 50000, 100000], index=2)
    target = st.number_input('Target Equity (USD)', min_value=1.0, value=3000.0, step=500.0)
    ruin_pct = st.number_input('Ruin Threshold (% of initial capital)', min_value=0.1, max_value=99.0, value=20.0, step=5.0)
    seed = st.number_input('Random Seed (reproducible)', min_value=0, max_value=2147483647, value=20261020)
    run = st.button('▶ Run Monte Carlo', type='primary', use_container_width=True)

risk = risk_pct / 100
p = win_pct / 100
expectancy = p * rr - (1-p)
breakeven = 100/(1+rr)
cols = st.columns(4)
cols[0].metric('Expectancy (R)', f'{expectancy:+.2f}R')
cols[1].metric('Break-even Win Rate', f'{breakeven:.2f}%')
cols[2].metric('10-loss Drawdown', f'{(1-(1-risk)**10)*100:.2f}%')
cols[3].metric('10-loss Balance', f'${capital*(1-risk)**10:,.2f}')

@st.cache_data(show_spinner=False, max_entries=3)
def simulate(capital, risk, rr, p, trades, paths, target, ruin_pct, seed):
    rng = np.random.default_rng(seed)
    final = np.empty(paths, dtype=np.float64)
    maxdd = np.empty(paths, dtype=np.float64)
    streaks = np.empty(paths, dtype=np.int32)
    hit = np.empty(paths, dtype=np.bool_)
    hit_trade = np.zeros(paths, dtype=np.int32)
    ruined = np.empty(paths, dtype=np.bool_)
    # Store just ten-ish candidate trajectories, avoiding full paths x trades memory.
    candidate_count = min(paths, 1000)
    candidate_idx = np.unique(np.linspace(0, paths-1, candidate_count, dtype=int))
    candidates = {}
    threshold = capital * ruin_pct/100
    for start in range(0, paths, 2000):
        end = min(start+2000, paths)
        n = end-start
        equity = np.full(n, capital, dtype=np.float64)
        peak = equity.copy()
        dd = np.zeros(n, dtype=np.float64)
        streak = np.zeros(n, dtype=np.int32)
        longest = streak.copy()
        ever_hit = equity >= target
        first_hit = np.zeros(n, dtype=np.int32)
        ever_ruined = equity <= threshold
        selected = candidate_idx[(candidate_idx>=start)&(candidate_idx<end)]
        local = selected-start
        sample = np.empty((trades+1, len(local)), dtype=np.float64)
        if len(local): sample[0] = equity[local]
        for t in range(1, trades+1):
            wins = rng.random(n) < p
            equity *= np.where(wins, 1+risk*rr, 1-risk)
            peak = np.maximum(peak, equity)
            dd = np.maximum(dd, 1-equity/peak)
            streak = np.where(wins, 0, streak+1)
            longest = np.maximum(longest, streak)
            new_hit = (~ever_hit) & (equity>=target)
            first_hit[new_hit] = t
            ever_hit |= equity>=target
            ever_ruined |= equity<=threshold
            if len(local): sample[t] = equity[local]
        final[start:end] = equity
        maxdd[start:end] = dd
        streaks[start:end] = longest
        hit[start:end] = ever_hit
        hit_trade[start:end] = first_hit
        ruined[start:end] = ever_ruined
        for j, index in enumerate(selected): candidates[int(index)] = sample[:,j].copy()
    # Choose representative paths by FINAL equity quantile rank, not random first ten.
    order = np.argsort(final)
    quantiles = np.array([0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.95])
    # Candidates have independently simulated valid paths; find nearest final to global quantile.
    ci = np.array(list(candidates.keys()), dtype=int)
    cfinal = final[ci]
    selected_paths = {}
    for q in quantiles:
        desired = np.quantile(final, q)
        k = int(np.argmin(abs(cfinal-desired)))
        selected_paths[f'P{int(q*100):02d}'] = candidates[int(ci[k])]
    chart = pd.DataFrame(selected_paths)
    chart.index.name = 'Trade #'
    return {
        'final':final,'maxdd':maxdd,'streaks':streaks,'hit':hit,'hit_trade':hit_trade,
        'ruined':ruined,'chart':chart,'run_at':datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

if run:
    with st.spinner(f'Running {paths:,} paths × {trades:,} trades...'):
        st.session_state['result'] = simulate(capital,risk,rr,p,int(trades),int(paths),target,ruin_pct,int(seed))
        st.session_state['settings'] = (capital,risk_pct,rr,win_pct,int(trades),int(paths),target,ruin_pct,int(seed))

if 'result' in st.session_state:
    settings = (capital,risk_pct,rr,win_pct,int(trades),int(paths),target,ruin_pct,int(seed))
    saved_settings = st.session_state['settings']
    if settings != saved_settings:
        st.warning('OUTDATED — Settings changed. Click Run Monte Carlo to recalculate.')
    else:
        r = st.session_state['result']
        st.success(f'UP TO DATE — {saved_settings[5]:,} paths × {saved_settings[4]:,} trades | Completed {r["run_at"]}')
        f = r['final']; dd=r['maxdd']; streak=r['streaks']
        st.subheader('Monte Carlo Risk Metrics')
        for row in [
            [('Median Final Equity',f'${np.median(f):,.2f}'),('P5 Final Equity',f'${np.quantile(f,.05):,.2f}'),('P95 Final Equity',f'${np.quantile(f,.95):,.2f}'),('Mean Final Equity',f'${np.mean(f):,.2f}')],
            [('P95 Max Drawdown',f'{np.quantile(dd,.95)*100:.2f}%'),('Mean Max Drawdown',f'{np.mean(dd)*100:.2f}%'),('P95 Loss Streak',f'{np.quantile(streak,.95):.0f}'),('DD ≥ 40%',f'{np.mean(dd>=.4)*100:.2f}%')],
            [('Goal Hit Rate',f'{np.mean(r["hit"])*100:.2f}%'),('Final Goal Rate',f'{np.mean(f>=target)*100:.2f}%'),('Risk of Ruin',f'{np.mean(r["ruined"])*100:.2f}%'),('End Below Initial',f'{np.mean(f<capital)*100:.2f}%')]
        ]:
            cs=st.columns(4)
            for c,(label,value) in zip(cs,row):c.metric(label,value)
        hits=r['hit_trade'][r['hit_trade']>0]
        if len(hits): st.caption(f'Median first goal-hit trade (among paths that hit): {np.median(hits):.0f}')
        st.subheader('Representative Equity Paths')
        st.caption('Ten simulated paths selected to approximate final-equity percentiles. They are illustrative paths, not confidence bounds at each trade.')
        st.line_chart(r['chart'])
        st.subheader('Final Equity Distribution')
        st.caption('Logarithmic x-bins used because compounded equity may be highly skewed.')
        vals = np.log10(np.maximum(f,1e-12))
        counts, edges = np.histogram(vals,bins=40)
        st.bar_chart(pd.DataFrame({'Paths':counts},index=np.round((edges[:-1]+edges[1:])/2,2)))
        summary=pd.DataFrame({'Metric':['Median Final Equity','P5 Final Equity','P95 Final Equity','Mean Final Equity','P95 Max Drawdown','Mean Max Drawdown','P95 Longest Loss Streak','Goal Hit Rate','Final Goal Rate','Risk of Ruin','DD >= 40%','End Below Initial'], 'Value':[np.median(f),np.quantile(f,.05),np.quantile(f,.95),np.mean(f),np.quantile(dd,.95),np.mean(dd),np.quantile(streak,.95),np.mean(r['hit']),np.mean(f>=target),np.mean(r['ruined']),np.mean(dd>=.4),np.mean(f<capital)]})
        st.download_button('Download Simulation Metrics (CSV)',summary.to_csv(index=False).encode('utf-8'),'monte_carlo_metrics.csv','text/csv')
else:
    st.info('Set your assumptions in the sidebar, then click **Run Monte Carlo**. No saved snapshot is presented as a fresh simulation.')

st.caption('Model: independent win/loss per trade; full equity fraction risked each trade; winning return +risk×RR, losing return −risk. No fees, slippage, funding, leverage liquidation, regime shifts, or win-rate uncertainty. This is NOT a backtest or investment forecast.')
