import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import firebase_admin
from firebase_admin import credentials, firestore

st.set_page_config(
    page_title="Apple Upgrade Prediction Dashboard",
    page_icon="",
    layout="wide"
)

# FIREBASE ADMIN SDK (API)
@st.cache_resource
def get_db():
    if not firebase_admin._apps:
        firebase_creds = dict(st.secrets["firebase"])
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = get_db()
TARGET_COLLECTION = "apple_upgrade_predictions"

# Model Logic

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
    """Map forcing term → upgrade intention segment."""
    value = round(value, 2)
    if value >= 0.70:
        return "Upgrade Soon"
    elif value >= 0.30:
        return "Delay Upgrade"
    else:
        return "Churn Risk"


# ACTION LAYER (CRM / BUSINESS RULES)
def recommend_actions(persona: str, intention: str):
    """
    Simple rule-based action trigger.
    Persona + Intention -> Recommended CRM actions.
    """
    persona = (persona or "").strip()
    intention = (intention or "").strip()

    # Default fallback
    actions = ["Send general follow-up message."]

    if persona == "Loyalist":
        if intention == "Upgrade Soon":
            actions = [
                "Send VIP upgrade offer.",
                "Give small trade-in bonus."
            ]
        elif intention == "Delay Upgrade":
            actions = [
                "Send reminder.",
                "Offer a small accessory promotion."
            ]
        else:  # Churn Risk
            actions = [
                "Request feedback.",
                "Send a loyalty thank-you coupon."
            ]

    elif persona == "Fan":
        if intention == "Upgrade Soon":
            actions = [
                "Promote a device bundle.",
                "Highlight key feature improvements."
            ]
        elif intention == "Delay Upgrade":
            actions = [
                "Explain value of upgrading.",
                "Offer a modest trade-in top-up."
            ]
        else:  # Churn Risk
            actions = [
                "Suggest a lower-priced model.",
                "Reduce communication frequency."
            ]

    elif persona == "Switcher":
        if intention == "Upgrade Soon":
            actions = [
                "Highlight ecosystem benefits.",
                "Provide competitive trade-in value."
            ]
        elif intention == "Delay Upgrade":
            actions = [
                "Send comparison messaging.",
                "Offer a limited-time trade-in promotion."
            ]
        else:  # Churn Risk
            actions = [
                "Send a win-back offer.",
                "Highlight long-term device value."
            ]

    elif persona == "Drifter":
        if intention == "Upgrade Soon":
            actions = [
                "Suggest mid-range or refurbished devices.",
                "Keep communication simple."
            ]
        elif intention == "Delay Upgrade":
            actions = [
                "Send occasional generic promotion.",
                "Suggest older or budget models."
            ]
        else:  # Churn Risk
            actions = [
                "Send a final discount offer.",
                "Limit further marketing spend for this user."
            ]

    return actions

# DATA LOADING
@st.cache_data
def load_data_from_firestore():
    
    # API call → reads all documents in the collection

    docs = list(db.collection(TARGET_COLLECTION).stream())
    rows = []
    for doc in docs:
        d = doc.to_dict()
        intention_val = d.get("intention", d.get("decision"))
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
            "intention": intention_val,
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

        # if Firestore already has crm_actions, use them; else compute now
        stored_actions = r.get("crm_actions") if isinstance(r, pd.Series) else None
        if stored_actions:
            actions_col.append(stored_actions)
        else:
            actions_col.append(recommend_actions(p, r.intention))

    df["persona"] = personas
    df["persona_scores"] = score_list
    df["Need"] = Ns
    df["Bonding"] = Bs
    df["Hesitation"] = Hs
    df["crm_actions"] = actions_col

    return df


#  UI STYLE
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

# MAIN APP
st.title("Apple Upgrade Prediction Dashboard")

with st.sidebar:
    st.markdown("### Data controls")
    if st.button("Refresh data from Firestore"):
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


# TAB 5: DATA LOADER 
with tab_loader:
    st.subheader("Import CSV and Save to Firestore")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        raw_df = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(raw_df.head())

        required_cols = ["id", "DA", "BH", "TI", "ENG", "PU", "SI", "PS"]
        missing = [c for c in required_cols if c not in raw_df.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            if st.button("Compute and save"):
                ok = 0
                with st.spinner("Computing and saving documents..."):
                    for _, r in raw_df.iterrows():
                        try:
                            user_id = str(r["id"])
                            DA, BH, TI, ENG, PU, SI, PS = map(
                                float, [r.DA, r.BH, r.TI, r.ENG, r.PU, r.SI, r.PS]
                            )

                            ft_raw = compute_forcing_term(DA, BH, TI, ENG, PU, SI, PS)
                            ft = round(ft_raw, 3)
                            intention = classify_forcing_term(ft)

                            dominant, scores, _, _, _ = compute_persona(
                                DA, BH, TI, ENG, PU, SI, PS
                            )

                            crm_actions = recommend_actions(dominant, intention)

                            out_doc = {
                                "DA": DA, "BH": BH, "TI": TI, "ENG": ENG,
                                "PU": PU, "SI": SI, "PS": PS,
                                "forcing_term": ft,
                                "intention": intention,
                                "decision": intention,  # legacy alias (optional)
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
                st.info("Use the other tabs to review the results.")
    else:
        st.info("Upload a CSV file to compute and save results.")


if df.empty:
    with tab_overview:
        st.warning("No computed documents found. Use the Data Loader tab to upload and process a CSV file.")
    with tab_persona:
        st.info("Persona insights will be available after data is loaded.")
    with tab_user:
        st.info("User Explorer will be available after data is loaded.")
    with tab_crm:
        st.info("CRM Planner will be available after data is loaded.")
    st.stop()


st.sidebar.markdown("### Filters")

intention_options = ["Upgrade Soon", "Delay Upgrade", "Churn Risk"]
selected_intentions = st.sidebar.multiselect(
    "Output Prediction",
    intention_options,
    default=intention_options
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
    df["intention"].isin(selected_intentions)
    & df["persona"].isin(selected_personas)
    & (df["forcing_term"] >= forcing_min)
    & (df["forcing_term"] <= forcing_max)
].copy()


total_users = len(filtered_df)
avg_forcing = filtered_df["forcing_term"].mean() if total_users else 0

upgrade_count = int((filtered_df["intention"] == "Upgrade Soon").sum())
delay_count   = int((filtered_df["intention"] == "Delay Upgrade").sum())
churn_count   = int((filtered_df["intention"] == "Churn Risk").sum())

upgrade_rate = upgrade_count / total_users * 100 if total_users else 0
delay_rate   = delay_count / total_users * 100 if total_users else 0
churn_rate   = churn_count / total_users * 100 if total_users else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Users (filtered)", f"{total_users}")
with c2:
    st.metric("Upgrade Soon", f"{upgrade_count} ({upgrade_rate:.1f}%)")
with c3:
    st.metric("Delay Upgrade", f"{delay_count} ({delay_rate:.1f}%)")
with c4:
    st.metric("Churn Risk", f"{churn_count} ({churn_rate:.1f}%)")

st.markdown(f"**Average forcing term:** `{avg_forcing:.3f}`")
st.markdown("---")



# ===================== TAB 1: OVERVIEW =====================
with tab_overview:
    st.subheader("Forcing Term and Prediction Overview")

    c1, c2 = st.columns([2, 1])

    # -------- Left: Forcing term distribution (histogram) --------
    with c1:
        st.markdown("**Forcing term distribution**")
        if not filtered_df.empty:
            arr = filtered_df["forcing_term"].to_numpy()
            fig_hist, ax_hist = plt.subplots()
            ax_hist.hist(arr, bins=10, edgecolor="black")
            ax_hist.set_xlabel("Forcing term")
            ax_hist.set_ylabel("Number of users")
            st.pyplot(fig_hist)
        else:
            st.info("No forcing term values for the current filter.")

    # -------- Right: Intention breakdown pie --------
    with c2:
        st.markdown("**Prediction breakdown**")
        if not filtered_df.empty:
            intention_counts = (
                filtered_df["intention"]
                .value_counts()
                .reindex(intention_options, fill_value=0)
            )
            fig, ax = plt.subplots()
            ax.pie(
                intention_counts.values,
                labels=intention_counts.index,
                autopct="%1.0f%%",
                startangle=90,
            )
            ax.axis("equal")
            st.pyplot(fig)
        else:
            st.info("No intention data for the current filter.")


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
        st.markdown("**Average forcing term by persona**")
        persona_means = filtered_df.groupby("persona")[["forcing_term"]].mean()
        persona_means = persona_means.reindex(persona_options)
        st.bar_chart(persona_means, use_container_width=True)

    with st.expander("Behavioral profile by persona (Need / Bonding / Hesitation)"):
        if not filtered_df.empty:
            beh_means = filtered_df.groupby("persona")[["Need", "Bonding", "Hesitation"]].mean()
            beh_means = beh_means.reindex(persona_options)
            st.bar_chart(beh_means, use_container_width=True)
            st.caption("Higher Need and Bonding are linked to upgrades; higher Hesitation is linked to delay or churn.")
        else:
            st.info("No data available to show behavioral profiles.")


# TAB 3: CRM PLANNER ==
with tab_crm:
    st.subheader("CRM Planner")

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

            st.markdown("### CRM actions in this segment")

            st.dataframe(
                action_counts.rename("Count").to_frame(),
                use_container_width=True,
            )

            #  Horizontal Bar Chart 
            import textwrap

            top_n = min(8, len(action_counts))  
            subset = action_counts.head(top_n)[::-1]  

            wrapped_labels = [textwrap.fill(lbl, width=30) for lbl in subset.index]

            fig_act, ax_act = plt.subplots(figsize=(10, 6))
            ax_act.barh(range(len(subset)), subset.values)
            ax_act.set_yticks(range(len(subset)))
            ax_act.set_yticklabels(wrapped_labels)
            ax_act.set_xlabel("Count")
            ax_act.set_title("Top CRM actions")
            ax_act.grid(axis="x", linestyle="--", alpha=0.4)

            plt.tight_layout()
            st.pyplot(fig_act)

            st.markdown("### Export segment")
            export_df = filtered_df[["id", "persona", "intention", "crm_actions"]].copy()
            csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="crm_segment_export.csv",
                mime="text/csv",
            )

        else:
            st.info("This segment currently has no CRM actions.")
    else:
        st.info("No CRM action data available. Load and compute data first.")


# ===================== TAB 4: USER EXPLORER =====================
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
    intention = user_row["intention"]

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown("**User summary**")
        st.markdown(f"- User ID: `{selected_user_id}`")
        st.markdown(f"- Intention: {intention}")
        st.markdown(f"- Forcing term: `{user_row['forcing_term']:.3f}`")
        st.markdown(f"- Persona: **{persona}**")

        beh_df = pd.DataFrame({
            "Factor": ["Need", "Bonding", "Hesitation"],
            "Value": [user_row["Need"], user_row["Bonding"], user_row["Hesitation"]]
        })
        st.markdown("**Behavioral factors**")
        st.bar_chart(beh_df, x="Factor", y="Value", use_container_width=True)

        st.markdown("---")
        st.markdown("**Recommended CRM actions**")

        stored_actions = user_row.get("crm_actions")
        if stored_actions:
            action_list = stored_actions
        else:
            action_list = recommend_actions(persona, intention)

        for a in action_list:
            st.markdown(f"- {a}")

    with right:
        st.markdown("**Persona radar (H1–H4)**")
        fig = radar_chart(scores, title=f"{persona} profile")
        st.pyplot(fig)
