"""
evaluate_models.py  -  honest, vehicle-grouped evaluation of the Service Haus prediction experiments.

Replaces the in-sample scoring in train_models.py. Every score is Leave-One-Vehicle-Out (LOVO):
all of a vehicle's visit-pairs are held out together, so no vehicle appears in both train and test.
Results are written to evaluation_results.json, which the dashboard (app.py) reads - nothing is hardcoded.
"""
import json, re, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support
warnings.filterwarnings("ignore")

SUBSYSTEMS = ["cat_brakes", "cat_cooling", "cat_drivetrain", "cat_electrical"]
FEATURES = ["client_type", "odometer_km", "total_aud", "has_warning", "oil_grade"] + SUBSYSTEMS
CAT_COLS = ["client_type", "oil_grade"]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]


def load(path="workshop_service_data.csv"):
    df = pd.read_csv(path)
    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    df = df.dropna(subset=["issue_date"])
    for c in ["vehicle_token", "client_type", "oil_grade"]:
        df[c] = df[c].astype(str).str.strip().str.upper()
    df["vehicle"] = df["vehicle_token"].apply(lambda x: re.sub(r"_(SMOKE|EXTRA|VIC|QLD|SUPERVISOR).*", "", x))
    return df.sort_values(["vehicle", "issue_date"])


def transition_pairs(df):
    rows = []
    for v, g in df.groupby("vehicle"):
        r = g.to_dict("records")
        for i in range(len(r) - 1):
            gap = (r[i + 1]["issue_date"] - r[i]["issue_date"]).days
            if gap <= 0:           # same-day duplicate invoices are not a return visit
                continue
            row = {"vehicle": v, "gap_days": gap}
            row.update({c: r[i][c] for c in FEATURES})
            row.update({"next_" + c: r[i + 1][c] for c in SUBSYSTEMS})
            rows.append(row)
    return pd.DataFrame(rows)


def pipe(model, scale=False):
    num = Pipeline([("i", SimpleImputer(strategy="median"))] + ([("s", StandardScaler())] if scale else []))
    pre = ColumnTransformer([("n", num, NUM_COLS),
                             ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS)])
    return Pipeline([("p", pre), ("m", model)])


class Constant:
    """Transparent baseline: predict the training mean/median, optionally per segment."""
    def __init__(self, stat="mean", by=None): self.stat, self.by = stat, by
    def fit(self, X, y):
        self.g = getattr(np, self.stat)(y)
        if self.by:
            d = pd.DataFrame({"k": X[self.by].values, "y": y})
            self.m = d.groupby("k").y.agg(self.stat).to_dict()
        return self
    def predict(self, X):
        return np.array([self.m.get(k, self.g) for k in X[self.by]]) if self.by else np.full(len(X), self.g)


def lovo_predict(P, y, make):
    pred = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(P, y, P["vehicle"].values):
        m = make(); m.fit(P.iloc[tr][FEATURES], y[tr]); pred[te] = m.predict(P.iloc[te][FEATURES])
    return pred


def vehicle_bootstrap_ci(y, pred, groups, n=500, seed=0):
    rng = np.random.default_rng(seed); vs = np.unique(groups); out = []
    for _ in range(n):
        idx = np.concatenate([np.where(groups == v)[0] for v in rng.choice(vs, len(vs))])
        out.append(mean_absolute_error(y[idx], pred[idx]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    df = load(); P = transition_pairs(df); y = P["gap_days"].values; groups = P["vehicle"].values
    makers = {
        "Global mean interval (baseline)": lambda: Constant("mean"),
        "Global median interval (baseline)": lambda: Constant("median"),
        "Median interval by client type (baseline)": lambda: Constant("median", "client_type"),
        "Random forest (original config)": lambda: pipe(RandomForestRegressor(100, random_state=42)),
        "Random forest (regularised, min_leaf=5)": lambda: pipe(RandomForestRegressor(200, min_samples_leaf=5, random_state=42)),
        "Ridge regression": lambda: pipe(Ridge(alpha=10), scale=True),
    }
    reg = {}
    for name, mk in makers.items():
        p = lovo_predict(P, y, mk)
        reg[name] = {"lovo_mae_days": round(float(mean_absolute_error(y, p)), 2),
                     "ci95": [round(v, 1) for v in vehicle_bootstrap_ci(y, p, groups)]}
    rf = makers["Random forest (original config)"](); rf.fit(P[FEATURES], y)
    in_sample = round(float(mean_absolute_error(y, rf.predict(P[FEATURES]))), 2)

    clf = {}
    for c in SUBSYSTEMS:
        t = P["next_" + c].values; pr = np.zeros(len(t))
        for tr, te in LeaveOneGroupOut().split(P, t, groups):
            if t[tr].sum() == 0: continue
            m = pipe(RandomForestClassifier(200, min_samples_leaf=2, class_weight="balanced", random_state=42))
            m.fit(P.iloc[tr][FEATURES], t[tr]); pr[te] = m.predict(P.iloc[te][FEATURES])
        f = lambda yp: [round(float(v), 2) for v in precision_recall_fscore_support(t, yp, average="binary", zero_division=0)[:3]]
        clf[c.replace("cat_", "")] = {"positives": int(t.sum()), "n": int(len(t)),
                                       "rf_precision_recall_f1": f(pr),
                                       "always_positive_f1": f(np.ones(len(t)))[2],
                                       "persistence_f1": f(P[c].values)[2]}
    out = {"n_invoices": int(len(df)), "n_vehicles": int(df["vehicle"].nunique()),
           "vehicles_with_pairs": int(P["vehicle"].nunique()), "n_training_pairs": int(len(P)),
           "observation_window": [str(df[df.issue_date >= "2026-01-01"].issue_date.min().date()), str(df.issue_date.max().date())],
           "gap_days_median": float(np.median(y)), "gap_days_max": int(y.max()),
           "share_gaps_14d_or_less": round(float((y <= 14).mean()), 2),
           "rf_in_sample_mae_days_DO_NOT_REPORT": in_sample,
           "regression_lovo": reg, "subsystem_classifiers_lovo": clf}
    json.dump(out, open("evaluation_results.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
