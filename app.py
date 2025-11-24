import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import firebase_admin
from firebase_admin import credentials, firestore

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Apple Upgrade Prediction Dashboard",
    page_icon="🍏",
    layout="wide"
)

# ---------------- FIRESTORE SETUP ----------------
@st.cache_resource
def get_db():
    """Initialize Firebase Admin only once."""
    if not firebase_admin._apps:
        # Use Streamlit secrets for safest deployment
        firebase_creds = dict(st.secrets["firebase"])
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = get_db()
TARGET_COLLECTION = "apple_upgrade_predictions"

# ---------------- MODEL LOGIC ----------------
def compute_behaviorals(DA, BH, TI, ENG, PU, SI, PS):
    """Compute behavioral signals exactly like your code."""
    # Need
    N = (DA + TI + ENG + PU + SI) / 5.0
    # Bonding
    B = (ENG + PU + SI) / 3.0
    # Hesitation
    H = (
        (1 - DA) + BH + (1 - TI) + (1 - ENG) +
        (1 - PU) + (1 - SI) + PS * (1 - TI)
    ) / 7.0
    return N, B, H


def compute_persona(DA, BH, TI, ENG, PU, SI, PS):
    """
    Persona weights based on your project logic:
    Loyalist needs + bonding, low hesitation
    Fan strong bonding, low hesitation
    Switcher high hesitation dominates
    Drifter hesitation dominates, low need/bonding
    """
    N, B, H = compute_behaviorals(DA, BH, TI, ENG, PU, SI, PS)

    H1_loyalist = 0.6 * N + 0.3 * B - 0.5 * H
    H2_fan      = 0.7 * B + 0.2 * N - 0.3 * H
    H3_switcher = 0.6 * H - 0.3 * B + 0.1 * N
    H4_drifter  = 1.0 * H - 0.5 * N - 0.2 * B

    scores = {
        "Loyalist": H1_loyalist,
        "Fan": H2_fan,
        "Switcher": H3_switcher,
        "Drifter": H4_drifter
    }
    dominant = max(scores, key=scores.get)
    return dominant, scores


def compute_forcing_term(DA, BH, TI, ENG, PU, SI, PS):
    """Your corrected forcing-term model."""
    dt = 0.01
    eta = 0.9
    alpha = 0.7
    omega = 0.5
    t = 800

    X = np.zeros(t)
    Y = np.zeros(t)
    S = np.zeros(t)
    forcing_term = np.zeros(t)

    X[0] = alpha * (1 - BH) + (1 - alpha) * DA
    Y[0] = (omega * DA + (1 - BH) * omega) * PS
    S[0] = X[0] * (1 - Y[0])
    forcing_term[0] = 0.1

    for k in range(1, t):
        N, B, H = compute_behaviorals(DA, BH, TI, ENG, PU, SI, PS)

        X[k] = alpha * B + (1 - alpha) * N - (alpha * H)
        Y[k] = (omega * N + omega * B) * H
        S[k] = X[k] * (1 - Y[k])

        forcing_term[k] = forcing_term[k - 1] + eta * (S[k - 1] - forcing_term[k - 1]) * dt

    return float(forcing_term[-1])


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

        row = {
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
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["forcing_term"] = pd.to_numeric(df["forcing_term"], errors="coerce")
    df = df.dropna(subset=["forcing_term"])

    # persona columns computed fresh for dashboard
    personas = []
    score_list = []
    Ns, Bs, Hs = [], [], []

    for _, r in df.iterrows():
        p, scores = compute_persona(r.DA, r.BH, r.TI, r.ENG, r.PU, r.SI, r.PS)
        personas.append(p)
        score_list.append(scores)

        N, B, H = compute_behaviorals(r.DA, r.BH, r.TI, r.ENG, r.PU, r.SI, r.PS)
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
st.title("🍏 Apple Upgrade Prediction Dashboard")
st.caption("Firestore → Persona & Forcing Term Insights")

# Sidebar refresh
with st.sidebar:
    st.markdown("### Data controls")
    if st.button("Refresh Firestore"):
        load_data_from_firestore.clear()
        st.rerun()

df = load_data_from_firestore()

tabs = st.tabs(["📊 Overview", "🧠 Persona Insights", "👤 User Explorer", "⬆️ Data Loader"])
tab_overview, tab_persona, tab_user, tab_loader = tabs


# ---------------- EMPTY STATE ----------------
if df.empty:
    with tab_overview:
        st.warning("No computed documents found yet. Upload + compute using Data Loader.")
    with tab_persona:
        st.info("Persona insights appear once data exists.")
    with tab_user:
        st.info("User explorer appears once data exists.")
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
        if filtered_df.empty:
            st.info("No users match current filters.")
        else:
            line_df = filtered_df.sort_values("forcing_term").set_index("id")[["forcing_term"]]
            st.line_chart(line_df)

    with c2:
        st.markdown("**Decision breakdown**")
        if not filtered_df.empty:
            decision_counts = filtered_df["decision"].value_counts().reindex(decision_options, fill_value=0)
            fig, ax = plt.subplots()
            ax.pie(decision_counts.values, labels=decision_counts.index, autopct="%1.0f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)
        else:
            st.info("No data.")

    st.markdown("**Forcing term distribution**")
    if not filtered_df.empty:
        arr = filtered_df["forcing_term"].to_numpy()
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(arr, bins=10, edgecolor="black")
        ax_hist.set_xlabel("Forcing term")
        ax_hist.set_ylabel("Frequency")
        st.pyplot(fig_hist)


# ===================== TAB 2: PERSONA INSIGHTS =====================
with tab_persona:
    st.subheader("Persona Insights")

    if filtered_df.empty:
        st.info("No users match current filters.")
    else:
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

            # Clean bar view instead of table
            st.bar_chart(persona_means[["Need","Bonding","Hesitation"]], use_container_width=True)

            st.caption("Need ↑ and Bonding ↑ push toward Upgrade. Hesitation ↑ pushes toward Delay/Churn.")

        st.markdown("---")
        st.markdown("**Mean forcing term by persona**")
        st.bar_chart(persona_means[["forcing_term"]], use_container_width=True)


# ===================== TAB 3: USER EXPLORER =====================
def radar_chart(scores_dict, title="Persona Radar"):
    labels = list(scores_dict.keys())
    values = list(scores_dict.values())

    # close loop
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

    if filtered_df.empty:
        st.info("No users match current filters.")
    else:
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

            # quick behavioral bars
            beh_df = pd.DataFrame({
                "Factor": ["Need", "Bonding", "Hesitation"],
                "Value": [user_row["Need"], user_row["Bonding"], user_row["Hesitation"]]
            })
            st.bar_chart(beh_df, x="Factor", y="Value", use_container_width=True)

        with right:
            st.markdown("**Persona radar scores (H1–H4)**")
            fig = radar_chart(scores, title=f"{persona} Profile")
            st.pyplot(fig)


# ===================== TAB 4: DATA LOADER =====================
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

        required_cols = ["id","DA","BH","TI","ENG","PU","SI","PS"]
        missing = [c for c in required_cols if c not in raw_df.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            if st.button("Compute & Save"):
                ok = 0
                with st.spinner("Computing + saving..."):
                    for _, r in raw_df.iterrows():
                        user_id = str(r["id"])
                        DA,BH,TI,ENG,PU,SI,PS = map(float, [r.DA,r.BH,r.TI,r.ENG,r.PU,r.SI,r.PS])

                        ft_raw = compute_forcing_term(DA,BH,TI,ENG,PU,SI,PS)
                        ft = round(ft_raw,3)
                        decision = classify_forcing_term(ft)

                        dominant, scores = compute_persona(DA,BH,TI,ENG,PU,SI,PS)

                        out_doc = {
                            "DA": DA,"BH": BH,"TI": TI,"ENG": ENG,
                            "PU": PU,"SI": SI,"PS": PS,
                            "forcing_term": ft,
                            "decision": decision,
                            "persona": dominant,
                            "persona_scores": scores,
                            "source_id": user_id,
                            "created_at": firestore.SERVER_TIMESTAMP,
                        }

                        db.collection(TARGET_COLLECTION).document(user_id).set(out_doc)
                        ok += 1

                load_data_from_firestore.clear()
                st.success(f"Saved {ok} users to Firestore.")
                st.info("Go to Overview / Persona tabs to explore.")
