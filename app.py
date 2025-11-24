import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import firebase_admin
from firebase_admin import credentials, firestore

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Apple Upgrade Prediction Dashboard",
    page_icon="",
    layout="wide"
)

# ---------------- FIRESTORE SETUP ----------------
@st.cache_resource
def get_db():
    """Initialize Firebase Admin only once."""
    if not firebase_admin._apps:
        firebase_creds = dict(st.secrets["firebase"])
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = get_db()
TARGET_COLLECTION = "apple_upgrade_predictions"

# ---------------- MODEL LOGIC (FIXED v3 — STRONGER SEPARATION) ----------------
# Goal of this fix:
# - Make personas more distinct (Loyalist ≠ Fan, Switcher ≠ Drifter)
# - Make Upgrade/Delay/Churn align better with personas
# - Keep professor-style flow: Inputs → Behaviorals → Personas → C/V → X,Y,S → Forcing Term

import numpy as np

def compute_behaviorals(DA, BH, TI, ENG, PU, SI, PS):
    """
    Layer 1: Behavioral Extraction
    7 inputs -> (Need, Bonding, Hesitation)
    """
    N = (DA + TI + ENG + PU + SI) / 5.0
    B = (ENG + PU + SI) / 3.0
    H = (
        (1 - DA) + BH + (1 - TI) + (1 - ENG) +
        (1 - PU) + (1 - SI) + PS * (1 - TI)
    ) / 7.0
    return N, B, H


def _softmax(x, beta=4.0):
    """
    Stable softmax for persona normalization.
    beta↑ makes persona weights sharper (more distinct clusters).
    """
    x = np.array(x, dtype=float) * beta
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


def compute_persona(DA, BH, TI, ENG, PU, SI, PS):
    """
    Layer 2: Persona Mapping (STRONGER weights)
    (N,B,H) -> persona scores -> persona weights -> (C,V)

    Returns:
      dominant persona (string)
      raw persona scores (dict)
      persona weights (dict)
      Commitment C
      Volatility V
    """
    N, B, H = compute_behaviorals(DA, BH, TI, ENG, PU, SI, PS)

    # ---- FIXED Persona Equations (more gap between types) ----
    H1_loyalist = 0.8 * N + 0.4 * B - 0.6 * H
    H2_fan      = 0.6 * B + 0.3 * N - 0.4 * H
    H3_switcher = 0.8 * H - 0.2 * B + 0.05 * N
    H4_drifter  = 1.1 * H - 0.6 * N - 0.2 * B

    scores = {
        "Loyalist": H1_loyalist,
        "Fan": H2_fan,
        "Switcher": H3_switcher,
        "Drifter": H4_drifter
    }

    order = ["Loyalist", "Fan", "Switcher", "Drifter"]
    vec = [scores[k] for k in order]

    # sharper normalization
    w_vec = _softmax(vec, beta=4.0)
    weights = dict(zip(order, w_vec))

    # ---- FIXED Autonomy Signals (extremes dominate) ----
    C_raw = 1.4 * weights["Loyalist"] + 1.0 * weights["Fan"]
    V_raw = 1.3 * weights["Switcher"] + 1.5 * weights["Drifter"]

    # normalize C and V back into [0,1] and to sum ~ 1
    s = C_raw + V_raw + 1e-9
    C = float(C_raw / s)
    V = float(V_raw / s)

    dominant = max(scores, key=scores.get)
    return dominant, scores, weights, C, V


def compute_forcing_term(DA, BH, TI, ENG, PU, SI, PS):
    """
    Layer 3: Forcing Term Dynamics (persona-driven)
    Uses ONLY (C,V) from Layer 2.
    No scaling of alpha/omega/eta by C.
    """

    dt = 0.01
    t = 800

    # fixed parameters (as you wanted)
    alpha = 0.9
    omega = 0.7
    eta   = 0.9

    # Layer 2 output
    _, _, _, C, V = compute_persona(DA, BH, TI, ENG, PU, SI, PS)

    forcing = np.zeros(t)

    # ---- FIXED Initial Pressure (stronger separation) ----
    # baseline 0.15
    # commitment boosts more
    # volatility suppresses more
    forcing[0] = np.clip(0.15 + 0.5*C - 0.35*V, 0, 1)

    for k in range(1, t):
        # ---- FIXED Short-term signals (clean professor-style but sharper) ----
        # Upgrade Pressure
        X = (alpha * C) + (1 - alpha) * V
        # Hesitation Impact
        Y = omega * V
        # Effective Upgrade Signal
        S = X * (1 - Y)

        forcing[k] = forcing[k-1] + eta * (S - forcing[k-1]) * dt

    return float(forcing[-1])


def classify_forcing_term(value: float) -> str:
    value = round(value, 2)
    if value >= 0.60:
        return "Upgrade Soon"
    elif value >= 0.10:
        return "Delay Upgrade"
    else:
        return "Churn Risk"


# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data_from_firestore():
    docs = list(db.collection(TARGET_COLLECTION).stream())
    rows = []
    for doc in docs:
        d = doc.to_dict()
        rows.append({
            "id": d.get("source_id", doc.id),
            "DA": d.get("DA"),
            "BH": d.get("BH"),
            "TI": d.get("TI"),
            "ENG": d.get("ENG"),
            "PU": d.get("PU"),
            "SI": d.get("SI"),
            "PS": d.get("PS"),
            "forcing_term": d.get("forcing_term"),
            "decision": d.get("decision"),
            "created_at": d.get("created_at"),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["forcing_term"] = pd.to_numeric(df["forcing_term"], errors="coerce")
    df = df.dropna(subset=["forcing_term"])

    personas, score_list, Ns, Bs, Hs = [], [], [], [], []
    for _, r in df.iterrows():
        # FIX: compute_persona returns 5 values now
        p, scores, _, _, _ = compute_persona(
            r.DA, r.BH, r.TI, r.ENG, r.PU, r.SI, r.PS
        )
        personas.append(p)
        score_list.append(scores)

        N, B, H = compute_behaviorals(
            r.DA, r.BH, r.TI, r.ENG, r.PU, r.SI, r.PS
        )
        Ns.append(N); Bs.append(B); Hs.append(H)

    df["persona"] = personas
    df["persona_scores"] = score_list
    df["Need"] = Ns
    df["Bonding"] = Bs
    df["Hesitation"] = Hs

    return df


# ---------------- UI STYLE ----------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        h3 { margin-top: 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- MAIN APP ----------------
st.title(" Apple Upgrade Prediction Dashboard")
st.caption("Firestore → Persona & Forcing Term Insights")

with st.sidebar:
    st.markdown("### Data controls")
    if st.button("Refresh Firestore"):
        load_data_from_firestore.clear()
        st.rerun()

df = load_data_from_firestore()

tabs = st.tabs(["📊 Overview", "🧠 Persona Insights", "👤 User Explorer", "⬆️ Data Loader"])
tab_overview, tab_persona, tab_user, tab_loader = tabs


# ===================== TAB 4: DATA LOADER (ALWAYS VISIBLE) =====================
with tab_loader:
    st.subheader("CSV → Compute → Save to Firestore")

    st.markdown(
        """
        Upload your raw CSV and we will:
        - compute forcing_term
        - classify decision
        - compute persona + persona_scores
        - save into Firestore

        Required columns:
        `id, DA, BH, TI, ENG, PU, SI, PS`
        """
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        raw_df = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(raw_df.head())

        required_cols = ["id", "DA", "BH", "TI", "ENG", "PU", "SI", "PS"]
        missing = [c for c in required_cols if c not in raw_df.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            if st.button("Compute & Save"):
                ok = 0
                with st.spinner("Computing + saving..."):
                    for _, r in raw_df.iterrows():
                        try:
                            user_id = str(r["id"])
                            DA, BH, TI, ENG, PU, SI, PS = map(
                                float, [r.DA, r.BH, r.TI, r.ENG, r.PU, r.SI, r.PS]
                            )

                            ft_raw = compute_forcing_term(DA, BH, TI, ENG, PU, SI, PS)
                            ft = round(ft_raw, 3)
                            decision = classify_forcing_term(ft)

                            # FIX: compute_persona returns 5 values now
                            dominant, scores, _, _, _ = compute_persona(
                                DA, BH, TI, ENG, PU, SI, PS
                            )

                            out_doc = {
                                "DA": DA, "BH": BH, "TI": TI, "ENG": ENG,
                                "PU": PU, "SI": SI, "PS": PS,
                                "forcing_term": ft,
                                "decision": decision,
                                "persona": dominant,
                                "persona_scores": scores,
                                "source_id": user_id,
                                "created_at": firestore.SERVER_TIMESTAMP,
                            }

                            db.collection(TARGET_COLLECTION).document(user_id).set(out_doc)
                            ok += 1
                        except Exception as e:
                            st.warning(f"Skipped row {r.get('id','?')} due to {e}")

                load_data_from_firestore.clear()
                st.success(f"Saved {ok} users to Firestore.")
                st.info("Go to Overview / Persona tabs to explore.")
    else:
        st.info("Upload a CSV to compute and push results.")


# ===================== IF NO DATA YET: SHOW MESSAGES, DON'T STOP APP =====================
if df.empty:
    with tab_overview:
        st.warning("No computed documents found yet. Use Data Loader tab to upload CSV.")
    with tab_persona:
        st.info("Persona insights will appear after CSV upload.")
    with tab_user:
        st.info("User explorer will appear after CSV upload.")
    st.stop()


# ---------------- FILTERS ----------------
st.sidebar.markdown("### Filters")

decision_options = ["Upgrade Soon", "Delay Upgrade", "Churn Risk"]
selected_decisions = st.sidebar.multiselect(
    "Decision segment",
    decision_options,
    default=decision_options
)

persona_options = ["Loyalist", "Fan", "Switcher", "Drifter"]
selected_personas = st.sidebar.multiselect(
    "Persona type",
    persona_options,
    default=persona_options
)

forcing_min_val = float(df["forcing_term"].min())
forcing_max_val = float(df["forcing_term"].max())

forcing_min, forcing_max = st.sidebar.slider(
    "Forcing term range",
    forcing_min_val,
    forcing_max_val,
    (forcing_min_val, forcing_max_val),
    step=0.05
)

filtered_df = df[
    df["decision"].isin(selected_decisions)
    & df["persona"].isin(selected_personas)
    & (df["forcing_term"] >= forcing_min)
    & (df["forcing_term"] <= forcing_max)
].copy()


# ---------------- KPIs ----------------
total_users = len(filtered_df)
avg_forcing = filtered_df["forcing_term"].mean() if total_users else 0

upgrade_count = int((filtered_df["decision"] == "Upgrade Soon").sum())
delay_count   = int((filtered_df["decision"] == "Delay Upgrade").sum())
churn_count   = int((filtered_df["decision"] == "Churn Risk").sum())

upgrade_rate = upgrade_count / total_users * 100 if total_users else 0
delay_rate   = delay_count / total_users * 100 if total_users else 0
churn_rate   = churn_count / total_users * 100 if total_users else 0

k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Users (filtered)", total_users)
with k2: st.metric("Avg forcing term", f"{avg_forcing:.3f}")
with k3: st.metric("Upgrade Soon", f"{upgrade_rate:.1f}%")
with k4: st.metric("Delay Upgrade", f"{delay_rate:.1f}%")
st.write(f"**Churn Risk:** {churn_count} users ({churn_rate:.1f}%)")
st.markdown("---")


# ===================== TAB 1: OVERVIEW =====================
with tab_overview:
    st.subheader("Forcing term overview")
    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown("**Forcing term by user (sorted)**")
        line_df = filtered_df.sort_values("forcing_term").set_index("id")[["forcing_term"]]
        st.line_chart(line_df)

    with c2:
        st.markdown("**Decision breakdown**")
        decision_counts = filtered_df["decision"].value_counts().reindex(decision_options, fill_value=0)
        fig, ax = plt.subplots()
        ax.pie(decision_counts.values, labels=decision_counts.index, autopct="%1.0f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

    st.markdown("**Forcing term distribution**")
    arr = filtered_df["forcing_term"].to_numpy()
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(arr, bins=10, edgecolor="black")
    ax_hist.set_xlabel("Forcing term")
    ax_hist.set_ylabel("Frequency")
    st.pyplot(fig_hist)


# ===================== TAB 2: PERSONA INSIGHTS =====================
with tab_persona:
    st.subheader("Persona Insights")

    p_counts = filtered_df["persona"].value_counts().reindex(persona_options, fill_value=0)
    c1, c2 = st.columns([1.2, 2])

    with c1:
        st.markdown("**Persona distribution**")
        figp, axp = plt.subplots()
        axp.pie(p_counts.values, labels=p_counts.index, autopct="%1.0f%%", startangle=90)
        axp.axis("equal")
        st.pyplot(figp)

    with c2:
        st.markdown("**Average behaviorals per persona**")
        persona_means = filtered_df.groupby("persona")[["Need","Bonding","Hesitation","forcing_term"]].mean()
        persona_means = persona_means.reindex(persona_options)

        st.bar_chart(persona_means[["Need","Bonding","Hesitation"]], use_container_width=True)
        st.caption("Need ↑ and Bonding ↑ push toward Upgrade. Hesitation ↑ pushes toward Delay/Churn.")

    st.markdown("---")
    st.markdown("**Mean forcing term by persona**")
    st.bar_chart(persona_means[["forcing_term"]], use_container_width=True)


# ===================== TAB 3: USER EXPLORER =====================
def radar_chart(scores_dict, title="Persona Radar"):
    labels = list(scores_dict.keys())
    values = list(scores_dict.values())

    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(title, y=1.1)
    ax.grid(True)
    return fig

with tab_user:
    st.subheader("User Explorer")

    selected_user_id = st.selectbox(
        "Select user ID",
        options=filtered_df["id"].tolist()
    )

    user_row = filtered_df[filtered_df["id"] == selected_user_id].iloc[0]
    persona = user_row["persona"]
    scores  = user_row["persona_scores"]

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown(f"**Decision:** {user_row['decision']}")
        st.markdown(f"**Forcing term:** `{user_row['forcing_term']:.3f}`")
        st.markdown(f"**Persona:** **{persona}**")

        beh_df = pd.DataFrame({
            "Factor": ["Need", "Bonding", "Hesitation"],
            "Value": [user_row["Need"], user_row["Bonding"], user_row["Hesitation"]]
        })
        st.bar_chart(beh_df, x="Factor", y="Value", use_container_width=True)

    with right:
        st.markdown("**Persona radar scores (H1–H4)**")
        fig = radar_chart(scores, title=f"{persona} Profile")
        st.pyplot(fig)
