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

# ---------------- MODEL LOGIC  ----------------

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


def _softmax(x, beta=3.5):
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
    Layer 2: Persona Mapping
    (N,B,H) -> persona scores -> persona weights -> (C,V)

    Returns:
      dominant persona (string)
      raw persona scores (dict)
      persona weights (dict)
      Commitment C
      Volatility V
    """
    N, B, H = compute_behaviorals(DA, BH, TI, ENG, PU, SI, PS)

    H1_loyalist = 0.90 * N + 0.25 * B - 0.70 * H
    H2_fan      = 0.95 * B + 0.15 * N - 0.45 * H
    H3_switcher = 0.80 * H - 0.20 * B + 0.05 * N
    H4_drifter  = 1.10 * H - 0.60 * N - 0.20 * B

    scores = {
        "Loyalist": H1_loyalist,
        "Fan": H2_fan,
        "Switcher": H3_switcher,
        "Drifter": H4_drifter
    }

    order = ["Loyalist", "Fan", "Switcher", "Drifter"]
    vec = [scores[k] for k in order]

    w_vec = _softmax(vec, beta=3.5)
    weights = dict(zip(order, w_vec))

    # Commitment from positive personas
    C_raw = 1.4 * weights["Loyalist"] + 1.0 * weights["Fan"]
    # Volatility from unstable personas
    V_raw = 1.3 * weights["Switcher"] + 1.5 * weights["Drifter"]

    s = C_raw + V_raw + 1e-9
    C = float(C_raw / s)
    V = float(V_raw / s)

    dominant = max(weights, key=weights.get)

    return dominant, scores, weights, C, V


def compute_forcing_term(DA, BH, TI, ENG, PU, SI, PS):
    """
    Layer 3: Forcing Term Dynamics (persona-driven)
    Uses ONLY (C,V) from Layer 2.
    """
    dt = 0.01
    t = 800

    # parameters
    alpha = 0.9
    omega = 0.7
    eta   = 0.9

    _, _, _, C, V = compute_persona(DA, BH, TI, ENG, PU, SI, PS)

    forcing = np.zeros(t)

    # initial pressure: baseline + commitment boost - volatility penalty
    forcing[0] = np.clip(0.15 + 0.5*C - 0.35*V, 0, 1)

    for k in range(1, t):
        X = (alpha * C) + (1 - alpha) * V      # Upgrade Pressure
        Y = omega * V                          # Hesitation Impact
        S = X * (1 - Y)                        # Effective short-term signal

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


# ---------------- ACTION LAYER (CRM / BUSINESS RULES) ----------------

def recommend_actions(persona: str, decision: str):
    """
    Map (persona, decision) -> list of recommended CRM / business actions.
    """
    persona = persona or ""
    decision = decision or ""

    persona = persona.strip()
    decision = decision.strip()

    actions = []

    if persona == "Loyalist":
        if decision == "Upgrade Soon":
            actions = [
                "Send VIP early-upgrade invitation.",
                "Offer exclusive preorder slot or color.",
                "Apply small extra trade-in bonus (+5%)."
            ]
        elif decision == "Delay Upgrade":
            actions = [
                "Send soft reminder (no heavy discount).",
                "Offer accessory promo (case / cable / MagSafe).",
                "Promote AppleCare or extended warranty."
            ]
        else:  # Churn Risk but Loyalist (rare)
            actions = [
                "Trigger retention check: ask feedback on why they hesitate.",
                "Offer limited-time loyalty thank-you coupon.",
            ]

    elif persona == "Fan":
        if decision == "Upgrade Soon":
            actions = [
                "Highlight bundle deal (iPhone + AirPods / iCloud storage).",
                "Promote 0% installment or student pricing.",
                "Show benefits: better camera, battery, performance."
            ]
        elif decision == "Delay Upgrade":
            actions = [
                "Send value-focused email: 'What you gain by upgrading.'",
                "Small discount or trade-in top-up to unlock decision.",
            ]
        else:  # Churn Risk
            actions = [
                "Check if price is main barrier; offer budget / SE model.",
                "Do not overspend on incentives; keep light reminders only.",
            ]

    elif persona == "Switcher":
        if decision == "Upgrade Soon":
            actions = [
                "Emphasize Apple ecosystem lock-in (Continuity, iCloud, Handoff).",
                "Give competitive trade-in value even from Android.",
                "Provide migration assistance (data transfer, training tips)."
            ]
        elif decision == "Delay Upgrade":
            actions = [
                "Retarget with comparison ads vs competitors.",
                "Offer limited-time trade-in bonus to reduce indecision.",
            ]
        else:  # Churn Risk
            actions = [
                "Trigger win-back campaign with strong but one-time offer.",
                "Highlight long-term value and resale price of Apple devices."
            ]

    elif persona == "Drifter":
        if decision == "Upgrade Soon":
            actions = [
                "Suggest mid-tier or refurbished models.",
                "Keep communication simple and low-cost.",
            ]
        elif decision == "Delay Upgrade":
            actions = [
                "Send occasional generic promo (no heavy personalization).",
                "Suggest budget-friendly alternatives or older models.",
            ]
        else:  # Churn Risk
            actions = [
                "Send one final 'thank you + small discount' offer.",
                "If no engagement → reduce marketing spend for this user."
            ]
    else:
        # Unknown persona fallback
        actions = [
            "Monitor behavior for more data.",
            "Keep generic but not aggressive communication."
        ]

    return actions


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
            # may or may not exist yet:
            "crm_actions": d.get("crm_actions"),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["forcing_term"] = pd.to_numeric(df["forcing_term"], errors="coerce")
    df = df.dropna(subset=["forcing_term"])

    personas, score_list, Ns, Bs, Hs, actions_col = [], [], [], [], [], []
    for _, r in df.iterrows():
        p, scores, _, _, _ = compute_persona(
            r.DA, r.BH, r.TI, r.ENG, r.PU, r.SI, r.PS
        )
        personas.append(p)
        score_list.append(scores)

        N, B, H = compute_behaviorals(
            r.DA, r.BH, r.TI, r.ENG, r.PU, r.SI, r.PS
        )
        Ns.append(N)
        Bs.append(B)
        Hs.append(H)

        # if Firestore already has crm_actions, use them; else compute now
        stored_actions = r.get("crm_actions") if isinstance(r, pd.Series) else None
        if stored_actions:
            actions_col.append(stored_actions)
        else:
            actions_col.append(recommend_actions(p, r.decision))

    df["persona"] = personas
    df["persona_scores"] = score_list
    df["Need"] = Ns
    df["Bonding"] = Bs
    df["Hesitation"] = Hs
    df["crm_actions"] = actions_col

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
st.caption("Firestore → Persona → Forcing Term → CRM Action Recommendations")

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
        - derive CRM actions
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

                            dominant, scores, _, _, _ = compute_persona(
                                DA, BH, TI, ENG, PU, SI, PS
                            )

                            crm_actions = recommend_actions(dominant, decision)

                            out_doc = {
                                "DA": DA, "BH": BH, "TI": TI, "ENG": ENG,
                                "PU": PU, "SI": SI, "PS": PS,
                                "forcing_term": ft,
                                "decision": decision,
                                "persona": dominant,
                                "persona_scores": scores,
                                "crm_actions": crm_actions,
                                "source_id": user_id,
                                "created_at": firestore.SERVER_TIMESTAMP,
                            }

                            db.collection(TARGET_COLLECTION).document(user_id).set(out_doc)
                            ok += 1
                        except Exception as e:
                            st.warning(f"Skipped row {r.get('id','?')} due to {e}")

                load_data_from_firestore.clear()
                st.success(f"Saved {ok} users to Firestore.")
                st.info("Go to Overview / Persona / User Explorer tabs to explore.")
    else:
        st.info("Upload a CSV to compute and push results.")


# ===================== IF NO DATA YET =====================
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

    st.markdown("---")
    st.markdown("### Action Playbook (Persona → Typical Strategy)")
    st.markdown(
        """
        - **Loyalist** → Reward & retain (VIP upgrades, loyalty perks).  
        - **Fan** → Convince & support (installments, value-focused messaging).  
        - **Switcher** → Stabilize (ecosystem benefits, trade-in competitiveness).  
        - **Drifter** → Low-cost touch (generic promos, budget models, limited spend).
        """
    )


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
    decision = user_row["decision"]

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown(f"**User ID:** `{selected_user_id}`")
        st.markdown(f"**Decision:** {decision}")
        st.markdown(f"**Forcing term:** `{user_row['forcing_term']:.3f}`")
        st.markdown(f"**Persona:** **{persona}**")

        beh_df = pd.DataFrame({
            "Factor": ["Need", "Bonding", "Hesitation"],
            "Value": [user_row["Need"], user_row["Bonding"], user_row["Hesitation"]]
        })
        st.bar_chart(beh_df, x="Factor", y="Value", use_container_width=True)

        st.markdown("---")
        st.markdown("### Recommended CRM Actions")

        stored_actions = user_row.get("crm_actions")
        if stored_actions:
            action_list = stored_actions
        else:
            action_list = recommend_actions(persona, decision)

        for a in action_list:
            st.markdown(f"- {a}")

    with right:
        st.markdown("**Persona radar scores (H1–H4)**")
        fig = radar_chart(scores, title=f"{persona} Profile")
        st.pyplot(fig)
