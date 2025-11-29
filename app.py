import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

import firebase_admin
from firebase_admin import credentials, firestore

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Apple Upgrade Prediction Dashboard",
    page_icon="📱",
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

    C_raw = 1.4 * weights["Loyalist"] + 1.0 * weights["Fan"]
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

    alpha = 0.9
    omega = 0.7
    eta   = 0.9

    _, _, _, C, V = compute_persona(DA, BH, TI, ENG, PU, SI, PS)

    forcing = np.zeros(t)

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
                "Give small trade-in bonus.",
            ]
        elif decision == "Delay Upgrade":
            actions = [
                "Send gentle reminder.",
                "Offer small accessory promo.",
            ]
        else:
            actions = [
                "Ask for feedback.",
                "Send loyalty thank-you coupon.",
            ]

    elif persona == "Fan":
        if decision == "Upgrade Soon":
            actions = [
                "Promote bundle deal.",
                "Highlight camera/battery benefits.",
            ]
        elif decision == "Delay Upgrade":
            actions = [
                "Send value explanation.",
                "Offer small trade-in top-up.",
            ]
        else:
            actions = [
                "Suggest cheaper model.",
                "Keep light reminders only.",
            ]

    elif persona == "Switcher":
        if decision == "Upgrade Soon":
            actions = [
                "Highlight Apple ecosystem features.",
                "Give competitive trade-in value.",
            ]
        elif decision == "Delay Upgrade":
            actions = [
                "Retarget with comparison ads.",
                "Give limited-time trade-in bonus.",
            ]
        else:
            actions = [
                "Send win-back offer.",
                "Highlight long-term resale value.",
            ]

    elif persona == "Drifter":
        if decision == "Upgrade Soon":
            actions = [
                "Suggest mid-tier or refurbished models.",
                "Keep communication simple.",
            ]
        elif decision == "Delay Upgrade":
            actions = [
                "Send occasional generic promo.",
                "Suggest older/cheaper models.",
            ]
        else:
            actions = [
                "Send final small discount.",
                "Reduce marketing cost for this user.",
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


# ---------------- UI STYLE (Apple-ish clean look) ----------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.0rem;
            padding-bottom: 2.0rem;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }
        h1, h2, h3 {
            font-family: -apple-system, system-ui, BlinkMacSystemFont, "SF Pro Text", sans-serif;
        }
        .metric-card {
            border-radius: 14px;
            padding: 0.9rem 1.1rem;
            background: #111827;
            color: #F9FAFB;
            box-shadow: 0 14px 30px rgba(15,23,42,0.38);
        }
        .metric-title {
            font-size: 0.75rem;
            opacity: 0.65;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 600;
        }
        .metric-sub {
            font-size: 0.8rem;
            opacity: 0.8;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def metric_card(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------- MAIN APP ----------------
st.title("📱 Apple Upgrade Prediction Dashboard")
st.caption("From behavioral inputs → personas → forcing term → CRM actions")

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    if st.button("🔁 Refresh Firestore"):
        load_data_from_firestore.clear()
        st.rerun()

df = load_data_from_firestore()

tabs = st.tabs([
    "📊 Overview",
    "🧠 Personas",
    "📧 CRM Planner",
    "👤 User Explorer",
    "⬆️ Data Loader",
])
tab_overview, tab_persona, tab_crm, tab_user, tab_loader = tabs


# ===================== TAB 5: DATA LOADER =====================
with tab_loader:
    st.subheader("⬆️ Upload & Compute")

    st.markdown(
        """
        1. Upload CSV of users  
        2. We compute forcing term, decision, persona, CRM actions  
        3. Results are saved into Firestore  

        **Required columns**: `id, DA, BH, TI, ENG, PU, SI, PS`
        """
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        raw_df = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(raw_df.head(), use_container_width=True)

        required_cols = ["id", "DA", "BH", "TI", "ENG", "PU", "SI", "PS"]
        missing = [c for c in required_cols if c not in raw_df.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            if st.button("🚀 Compute & Save to Firestore"):
                ok = 0
                with st.spinner("Running model and writing to Firestore..."):
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
                            st.warning(f"Skipped row {r.get('id', '?')} due to: {e}")

                load_data_from_firestore.clear()
                st.success(f"Saved {ok} users to Firestore.")
                st.info("Switch to the other tabs to explore the results.")
    else:
        st.info("Upload a CSV file to start.")


# ===================== IF NO DATA YET =====================
if df.empty:
    with tab_overview:
        st.warning("No documents found yet. Use the **Data Loader** tab to upload CSV.")
    with tab_persona:
        st.info("Persona insights will appear after you upload and compute data.")
    with tab_user:
        st.info("User explorer needs at least one record.")
    with tab_crm:
        st.info("CRM planner will be populated after data is available.")
    st.stop()


# ---------------- GLOBAL FILTERS ----------------
st.sidebar.markdown("### 🔍 Segment Filter")

decision_options = ["Upgrade Soon", "Delay Upgrade", "Churn Risk"]
selected_decisions = st.sidebar.multiselect(
    "Decision segment",
    decision_options,
    default=decision_options,
)

persona_options = ["Loyalist", "Fan", "Switcher", "Drifter"]
selected_personas = st.sidebar.multiselect(
    "Persona type",
    persona_options,
    default=persona_options,
)

forcing_min_val = float(df["forcing_term"].min())
forcing_max_val = float(df["forcing_term"].max())

forcing_min, forcing_max = st.sidebar.slider(
    "Forcing term range",
    forcing_min_val,
    forcing_max_val,
    (forcing_min_val, forcing_max_val),
    step=0.05,
)

filtered_df = df[
    df["decision"].isin(selected_decisions)
    & df["persona"].isin(selected_personas)
    & (df["forcing_term"] >= forcing_min)
    & (df["forcing_term"] <= forcing_max)
].copy()


# ---------------- KPI CARDS ----------------
total_users = len(filtered_df)
avg_forcing = filtered_df["forcing_term"].mean() if total_users else 0

upgrade_count = int((filtered_df["decision"] == "Upgrade Soon").sum())
delay_count   = int((filtered_df["decision"] == "Delay Upgrade").sum())
churn_count   = int((filtered_df["decision"] == "Churn Risk").sum())

upgrade_rate = upgrade_count / total_users * 100 if total_users else 0
delay_rate   = delay_count / total_users * 100 if total_users else 0
churn_rate   = churn_count / total_users * 100 if total_users else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Users (filtered)", f"{total_users}", "")
with c2:
    metric_card("Avg forcing term", f"{avg_forcing:.3f}", "")
with c3:
    metric_card("Upgrade Soon", f"{upgrade_rate:.1f}%", f"{upgrade_count} users")
with c4:
    metric_card("Churn Risk", f"{churn_rate:.1f}%", f"{churn_count} users")

st.markdown("---")


# ===================== TAB 1: OVERVIEW =====================
with tab_overview:
    st.subheader("Segment Overview")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("**Forcing term by user (sorted)**")
        if not filtered_df.empty:
            line_df = (
                filtered_df
                .sort_values("forcing_term")
                .set_index("id")[["forcing_term"]]
            )
            st.line_chart(line_df, use_container_width=True)
        else:
            st.info("No users match the current filter.")

    with col_right:
        st.markdown("**Decision mix**")
        if not filtered_df.empty:
            decision_counts = (
                filtered_df["decision"]
                .value_counts()
                .reindex(decision_options, fill_value=0)
            )
            fig, ax = plt.subplots()
            ax.pie(
                decision_counts.values,
                labels=decision_counts.index,
                autopct="%1.0f%%",
                startangle=90,
            )
            ax.axis("equal")
            st.pyplot(fig)
        else:
            st.info("No decision data for the current filter.")

    with st.expander("Show forcing term distribution"):
        if not filtered_df.empty:
            arr = filtered_df["forcing_term"].to_numpy()
            fig_hist, ax_hist = plt.subplots()
            ax_hist.hist(arr, bins=10, edgecolor="black")
            ax_hist.set_xlabel("Forcing term")
            ax_hist.set_ylabel("Frequency")
            st.pyplot(fig_hist)
        else:
            st.info("No forcing term values for the current filter.")


# ===================== TAB 2: PERSONA INSIGHTS =====================
with tab_persona:
    st.subheader("Persona Insights")

    p_counts = (
        filtered_df["persona"]
        .value_counts()
        .reindex(persona_options, fill_value=0)
    )

    row1_col1, row1_col2 = st.columns([1.2, 2])

    with row1_col1:
        st.markdown("**Persona distribution**")
        figp, axp = plt.subplots()
        axp.pie(p_counts.values, labels=p_counts.index, autopct="%1.0f%%", startangle=90)
        axp.axis("equal")
        st.pyplot(figp)

    with row1_col2:
        st.markdown("**Mean forcing term by persona**")
        if not filtered_df.empty:
            persona_means = filtered_df.groupby("persona")[["forcing_term"]].mean()
            persona_means = persona_means.reindex(persona_options)
            st.bar_chart(persona_means, use_container_width=True)
        else:
            st.info("No data for this segment.")

    with st.expander("Behavioral profile per persona (Need / Bonding / Hesitation)"):
        if not filtered_df.empty:
            beh_means = filtered_df.groupby("persona")[["Need", "Bonding", "Hesitation"]].mean()
            beh_means = beh_means.reindex(persona_options)
            st.bar_chart(beh_means, use_container_width=True)
            st.caption("Need↑ + Bonding↑ push toward Upgrade. Hesitation↑ pushes toward Delay/Churn.")
        else:
            st.info("No data to show behavioral profiles.")

    st.markdown("---")
    st.markdown("### Persona Playbook (high level)")
    st.markdown(
        """
        - **Loyalist** → Reward & retain (VIP upgrades, loyalty perks).  
        - **Fan** → Convince & support (installments, value messaging).  
        - **Switcher** → Stabilize (ecosystem benefits, competitive trade-in).  
        - **Drifter** → Low-cost touch (generic promos, budget models).
        """
    )


# ===================== TAB 3: CRM PLANNER =====================
with tab_crm:
    st.subheader("📧 CRM Planner – Actions for Current Segment")

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

            st.markdown("#### Top CRM actions (for this filtered segment)")

            top_n = min(8, len(action_counts))
            subset = action_counts.head(top_n)[::-1]  # reverse for nicer layout

            wrapped_labels = [textwrap.fill(lbl, width=32) for lbl in subset.index]

            fig_act, ax_act = plt.subplots(figsize=(9, 5))
            ax_act.barh(range(len(subset)), subset.values)
            ax_act.set_yticks(range(len(subset)))
            ax_act.set_yticklabels(wrapped_labels)
            ax_act.set_xlabel("Frequency")
            ax_act.set_title("Top CRM Actions")
            ax_act.grid(axis="x", linestyle="--", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_act)

            with st.expander("See raw list of actions with counts"):
                st.dataframe(action_counts.rename("Count").to_frame(), use_container_width=True)

            st.markdown("#### 📥 Export this segment")
            export_df = filtered_df[["id", "persona", "decision", "crm_actions"]].copy()
            csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CRM Segment as CSV",
                data=csv,
                file_name="crm_segment_export.csv",
                mime="text/csv",
            )
        else:
            st.info("This segment has no CRM actions.")
    else:
        st.info("No CRM action data available yet. Upload and compute data first.")


# ===================== TAB 4: USER EXPLORER =====================
def radar_chart(scores_dict, title="Persona Radar"):
    labels = list(scores_dict.keys())
    values = list(scores_dict.values())

    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(title, y=1.1)
    ax.grid(True)
    return fig


with tab_user:
    st.subheader("👤 User Explorer")

    selected_user_id = st.selectbox(
        "Select user ID",
        options=filtered_df["id"].tolist(),
    )

    user_row = filtered_df[filtered_df["id"] == selected_user_id].iloc[0]
    persona = user_row["persona"]
    scores  = user_row["persona_scores"]
    decision = user_row["decision"]

    left, right = st.columns([1, 1.1])

    with left:
        st.markdown(f"**User ID:** `{selected_user_id}`")
        st.markdown(f"**Decision:** {decision}")
        st.markdown(f"**Forcing term:** `{user_row['forcing_term']:.3f}`")
        st.markdown(f"**Persona:** **{persona}**")

        beh_df = pd.DataFrame({
            "Factor": ["Need", "Bonding", "Hesitation"],
            "Value": [user_row["Need"], user_row["Bonding"], user_row["Hesitation"]],
        })
        st.bar_chart(beh_df, x="Factor", y="Value", use_container_width=True)

        st.markdown("---")
        st.markdown("### Recommended CRM actions")
        stored_actions = user_row.get("crm_actions")
        action_list = stored_actions if stored_actions else recommend_actions(persona, decision)

        for a in action_list:
            st.markdown(f"- {a}")

    with right:
        st.markdown("**Persona radar (H1–H4)**")
        fig = radar_chart(scores, title=f"{persona} profile")
        st.pyplot(fig)
