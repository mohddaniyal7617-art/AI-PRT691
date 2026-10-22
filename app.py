"""
Service Haus - Service Outreach Priority Queue (v2, rules-first).

Ranking is driven by transparent, owner-adjustable rules - NOT by the machine-learning experiments, because
those did not beat simple baselines under vehicle-grouped validation (see evaluation_results.json).
Decision support only: staff review every row before any customer is contacted.
"""
import json, re
from datetime import date
import numpy as np, pandas as pd, streamlit as st

st.set_page_config(page_title="Service Haus - Outreach Priority", layout="wide")
st.title("Service Haus: Service Outreach Priority Queue")
st.caption("Decision support only. A person reviews every row before any customer is contacted.")


@st.cache_data
def load_data():
    df = pd.read_csv("workshop_service_data.csv", parse_dates=["issue_date"])
    df["client_type"] = df["client_type"].astype(str).str.strip().str.title()
    df["vehicle"] = df["vehicle_token"].astype(str).str.upper().apply(
        lambda x: re.sub(r"_(SMOKE|EXTRA|VIC|QLD|SUPERVISOR).*", "", x))
    return df


@st.cache_data
def load_eval():
    try:
        return json.load(open("evaluation_results.json"))
    except FileNotFoundError:
        return None


df, ev = load_data(), load_eval()

# ---- owner-adjustable rules (defaults: retail = owner's ~6-month guidance; fleet = ASSUMPTION to confirm) ----
st.sidebar.header("Service interval rules")
retail_days = st.sidebar.number_input("Retail: expected days between services", 30, 365, 180, 5)
fleet_days = st.sidebar.number_input("Fleet: expected days between services", 30, 365, 90, 5,
                                     help="Assumption - confirm with the owner.")
warn_mult = st.sidebar.slider("Weight for an unresolved mechanic warning", 1.0, 2.0, 1.5, 0.1)
snapshot = pd.Timestamp(st.sidebar.date_input("Calculate as at", date.today()))

# ---- latest record per vehicle, scored by rules ----
latest = df.sort_values("issue_date").groupby("vehicle").last().reset_index()
latest["visits_on_record"] = latest["vehicle"].map(df.groupby("vehicle").size())
latest["days_since"] = (snapshot - latest["issue_date"]).dt.days.clip(lower=0)
latest["expected"] = np.where(latest["client_type"] == "Fleet", fleet_days, retail_days)
latest["score"] = latest["days_since"] / latest["expected"] * np.where(latest["has_warning"] == 1, warn_mult, 1.0)
latest["tier"] = pd.cut(latest["score"], [-np.inf, 0.85, 1.15, np.inf],
                        labels=["Routine / Recent", "Upcoming Due", "Critical / Overdue"])
subs = {"cat_brakes": "Brakes", "cat_cooling": "Cooling", "cat_drivetrain": "Drivetrain", "cat_electrical": "Electrical"}
latest["last_work_flags"] = latest.apply(
    lambda r: ", ".join(v for k, v in subs.items() if r[k] == 1) or "Routine only", axis=1)
latest["history"] = np.where(latest["visits_on_record"] >= 2, "Repeat visitor", "Single visit on record")

st.sidebar.header("Filters")
tiers = st.sidebar.multiselect("Tier", ["Critical / Overdue", "Upcoming Due", "Routine / Recent"],
                               default=["Critical / Overdue", "Upcoming Due"])
segs = st.sidebar.multiselect("Segment", ["Fleet", "Retail"], default=["Fleet", "Retail"])
view = latest[latest["tier"].isin(tiers) & latest["client_type"].isin(segs)].sort_values("score", ascending=False)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Vehicles on record", len(latest))
c2.metric("In current queue", len(view))
c3.metric("Critical / Overdue", int((latest["tier"] == "Critical / Overdue").sum()))
c4.metric("Open mechanic warnings in queue", int(view["has_warning"].sum()))

st.subheader("Priority outreach list")
show = view[["vehicle_token", "client_type", "issue_date", "days_since", "expected", "last_work_flags",
             "has_warning", "history", "score", "tier"]].rename(columns={
    "vehicle_token": "Vehicle token", "client_type": "Segment", "issue_date": "Last serviced",
    "days_since": "Days since", "expected": "Expected interval (days)", "last_work_flags": "Work on last visit",
    "has_warning": "Open warning", "history": "History", "score": "Priority score", "tier": "Tier"})
st.dataframe(show.style.format({"Priority score": "{:.2f}"}), use_container_width=True)
st.download_button("Download this list (CSV)", show.to_csv(index=False), "outreach_queue.csv")

st.warning("**How to read this list.** Rows are ranked by time since the last visit against the expected service "
           "interval, increased when a technician noted an open warning. A low-priority row does **not** mean the "
           "vehicle is safe, and a high-priority row is a prompt to check - not a diagnosis. Confirm the vehicle, "
           "the customer's contact preferences and any open disputes before calling. Invoice amounts are "
           "deliberately not shown.")

with st.expander("What the prediction experiments showed (why ML is not driving this list)"):
    if ev:
        st.write(f"Evidence window: {ev['observation_window'][0]} to {ev['observation_window'][1]}; "
                 f"{ev['n_training_pairs']} usable repeat-visit pairs from {ev['vehicles_with_pairs']} vehicles. "
                 "Errors are leave-one-vehicle-out (no vehicle in both training and test).")
        t = pd.DataFrame([{"Method": k, "MAE (days)": v["lovo_mae_days"], "95% CI": f"{v['ci95'][0]}-{v['ci95'][1]}"}
                          for k, v in ev["regression_lovo"].items()])
        st.table(t)
        st.write("No learned model beat the simple baselines, so intervals come from the rules above.")
    else:
        st.write("Run evaluate_models.py to generate evaluation_results.json.")
