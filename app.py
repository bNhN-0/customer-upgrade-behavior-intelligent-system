import streamlit as st
import numpy as np
import pandas as pd

import firebase_admin
from firebase_admin import credentials, firestore

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

# ---------------- MODEL LOGIC ----------------
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
    if value >= 0.70:
        return "Upgrade Soon"
    elif value >= 0.30:
        return "Delay Upgrade"
    else:
        return "Churn Risk"


# ---------------- ACTION LAYER (CRM / BUSINESS RULES) ----------------
def recommend_actions(persona: str, decision: str):
    """
    Simple rule-based action trigger.
    Persona + Decision -> Recommended CRM actions.
    """
    persona = (persona or "").strip()
    decision = (decision or "").strip()

    actions = ["Send general follow-up message."]

    if persona == "Loyalist":
        if decision == "Upgrade Soon":
            actions = [
                "Send VIP upgrade offer.",
                "Give small trade-in bonus."
            ]
        elif decision == "Delay Upgrade":
            actions = [
                "Send gentle reminder.",
                "Offer small accessory promo."
            ]
        else:
            actions = [
                "Ask for feedback.",
                "Send loyalty thank-you coupon."
            ]

    elif persona == "Fan":
        if decision == "Upgrade Soon":
            actions = [
                "Promote bundle deal.",
                "Highlight camera/battery benefits."
            ]
        elif decision == "Delay Upgrade":
            actions = [
                "Send value explanation.",
                "Offer small trade-in top-up."
            ]
        else:
            actions = [
                "Suggest cheaper model.",
                "Keep light reminders only."
            ]

    elif persona == "Switcher":
        if decision == "Upgrade Soon":
            actions = [
                "Highlight Apple ecosystem features.",
                "Give competitive trade-in value."
            ]
        elif decision == "Delay Upgrade":
            actions = [
                "Retarget with comparison ads.",
                "Give limited-time trade-in bonus."
            ]
        else:
            actions = [
                "Send win-back offer.",
                "Highlight long-term resale value."
            ]

    elif persona == "Drifter":
        if decision == "Upgrade Soon":
            actions = [
                "Suggest mid-tier or refurbished models.",
                "Keep communication simple."
            ]
        elif decision == "Delay Upgrade":
            actions = [
                "Send occasional generic promo.",
                "Suggest older/cheaper models."
            ]
        else:
            actions = [
                "Send final small discount.",
                "Reduce marketing cost for this user."
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
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        h3 { margin-top: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- MAIN APP ----------------
st.title("Apple Upgrade Prediction Dashboard")
st.caption("From behavioral inputs → Personas → Forcing Term → CRM Actions")

with st.sidebar:
    st.markdown("### Data controls")
    if st.button("Refresh Firestore"):
        load_data_from_firestore.clear()
        st.rerun()

df = load_data_from_firestore()

tabs = st.tabs([
    "Overview",
    "Persona Insights",
    "CRM Planner",
    "User Explorer",
    "Data Loader"
])
tab_overview, tab_persona, tab_crm, tab_user, tab_loader = tabs


# ===================== TAB 5: DATA LOADER =====================
with tab_loader:
    st.subheader("CSV → Compute → Save to Firestore")

    st.markdown(
        """
        1. Upload your raw CSV  
        2. We compute: forcing_term, decision, persona, CRM actions  
        3. Everything is saved into Firestore  

        **Required columns:** `id, DA, BH, TI, ENG, PU, SI, PS`
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
                st.info("Go to the other tabs to explore.")
    else:
        st.info("Upload a CSV to compute and push results.")


# ===================== IF NO DATA YET =====================
if df.empty:
    with tab_overview:
        st.warning("No computed documents found yet. Use the Data Loader tab to upload CSV.")
    with tab_persona:
        st.info("Persona insights will appear after CSV upload.")
    with tab_user:
        st.info("User explorer will appear after CSV upload.")
    with tab_crm:
        st.info("CRM planner needs data from Firestore.")
    st.stop()


# ---------------- FILTERS (GLOBAL) ----------------
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


# ---------------- KPIs (top of page) ----------------
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
    st.subheader("Segment Forcing Term & Decisions")

    if not filtered_df.empty:
        line_df = (
            filtered_df
            .sort_values("forcing_term")
            .set_index("id")[["forcing_term"]]
        )
        st.markdown("**Forcing term by user (sorted)**")
        st.line_chart(line_df, use_container_width=True)

        st.markdown("**Decision breakdown (counts)**")
        decision_counts = (
            filtered_df["decision"]
            .value_counts()
            .reindex(decision_options, fill_value=0)
        )
        st.dataframe(decision_counts.rename("Count").to_frame())
    else:
        st.info("No users match the current filter.")


# ===================== TAB 2: PERSONA INSIGHTS =====================
with tab_persona:
    st.subheader("Persona Insights")

    if not filtered_df.empty:
        p_counts = (
            filtered_df["persona"]
            .value_counts()
            .reindex(persona_options, fill_value=0)
        )
        st.markdown("**Persona distribution (counts)**")
        st.dataframe(p_counts.rename("Count").to_frame(), use_container_width=True)

        st.markdown("**Mean forcing term by persona**")
        persona_means = filtered_df.groupby("persona")[["forcing_term"]].mean()
        persona_means = persona_means.reindex(persona_options)
        st.bar_chart(persona_means, use_container_width=True)
    else:
        st.info("No persona data for the current filter.")


# ===================== TAB 3: CRM PLANNER =====================
with tab_crm:
    st.subheader("CRM Planner – Recommended Actions")

    if "crm_actions" in filtered_df.columns and not filtered_df.empty:
        all_actions = []
        for actions in filtered_df["crm_actions"]:
            if isinstance(actions, (list, tuple)):
                all_actions.extend(actions)
            elif isinstance(actions, str):
                all_actions.append(actions)

        if all_actions:
            action_series = pd.Series(all_actions)
            action_counts = action_series.value_counts().sort_values(ascending=False)

            st.markdown("### Top CRM Actions Across Selected Segment")
            st.dataframe(
                action_counts.rename("Count").to_frame(),
                use_container_width=True,
            )

            st.markdown("### Download This Segment")
            export_df = filtered_df[["id", "persona", "decision", "crm_actions"]].copy()
            csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CRM Segment as CSV",
                data=csv,
                file_name="crm_segment_export.csv",
                mime="text/csv",
            )
        else:
            st.info("This filtered segment has no CRM actions.")
    else:
        st.info("No CRM action data available. Upload and compute data first.")


# ===================== TAB 4: USER EXPLORER =====================
with tab_user:
    st.subheader("User Explorer")

    if filtered_df.empty:
        st.info("No users in current filter.")
    else:
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
            st.markdown("### Persona raw scores (H1–H4)")
            st.json(scores)
