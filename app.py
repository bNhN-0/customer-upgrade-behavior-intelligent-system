import textwrap

import firebase_admin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from firebase_admin import credentials, firestore

st.set_page_config(
    page_title="Customer Upgrade Behavior Intelligent System",
    page_icon="",
    layout="wide",
)

DATASET_PATH = "customer_upgrade_behavior.csv"
TARGET_COLLECTION = "customer_upgrade_predictions"
INPUT_COLS = ["DA", "BH", "TI", "ENG", "PU", "SI", "PS"]
INTENTION_OPTIONS = ["Upgrade Soon", "Delay Upgrade", "Churn Risk"]
PERSONA_OPTIONS = ["Loyalist", "Fan", "Switcher", "Drifter"]


@st.cache_resource
def get_db():
    if "firebase" not in st.secrets:
        return None

    try:
        if not firebase_admin._apps:
            firebase_creds = dict(st.secrets["firebase"])
            cred = credentials.Certificate(firebase_creds)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception:
        return None


db = get_db()


def _clip01(x):
    return float(np.clip(x, 0.05, 0.95))


def compute_behaviorals(DA, BH, TI, ENG, PU, SI, PS):
    alpha_N = 0.46
    beta_N = 0.86
    lambda_N = 1.00

    N_support = (
        alpha_N * (DA + BH)
        + (1 - alpha_N)
        * (beta_N * (TI + PU + SI) + (1 - beta_N) * lambda_N * ENG)
    )
    N = _clip01((1 - PS) * N_support)

    alpha_B = 0.38
    beta_B = 0.80
    lambda_B = 1.00

    B_support = (
        alpha_B * ENG
        + (1 - alpha_B)
        * (beta_B * (PU + SI) + (1 - beta_B) * lambda_B * TI)
    )
    B = _clip01((1 - PS) * B_support)

    alpha_H = 0.27
    beta_H = 0.75
    lambda_H = 0.50

    H_reducer = (
        alpha_H * TI
        + (1 - alpha_H)
        * (beta_H * (DA + BH + PU) + (1 - beta_H) * lambda_H * (ENG + SI))
    )
    H_reducer = _clip01(H_reducer)
    H = _clip01(PS * (1 - H_reducer))

    return N, B, H


def compute_persona(DA, BH, TI, ENG, PU, SI, PS):
    N, B, H = compute_behaviorals(DA, BH, TI, ENG, PU, SI, PS)

    alpha_1 = 0.60
    alpha_2 = 0.75
    alpha_3 = 0.60
    alpha_4 = 0.60

    Y1 = _clip01((1 - H) * (alpha_1 * B + (1 - alpha_1) * N))

    Y2 = _clip01((1 - H) * (alpha_2 * B + (1 - alpha_2) * N))

    Y3 = _clip01((alpha_3 * H + (1 - alpha_3) * N) * (1 - B))

    Y4 = _clip01(H * (alpha_4 * (1 - B) + (1 - alpha_4) * N))

    scores = {
        "Loyalist": Y1,
        "Fan": Y2,
        "Switcher": Y3,
        "Drifter": Y4,
    }

    total_score = sum(scores.values()) + 1e-9
    weights = {k: float(v / total_score) for k, v in scores.items()}
    dominant = max(scores, key=scores.get)

    delta_C = 0.75
    alpha_C = 0.60

    C_limiter = _clip01(delta_C * Y4 + (1 - delta_C) * Y3)
    C_support = _clip01(alpha_C * Y1 + (1 - alpha_C) * Y2)
    C = _clip01((1 - C_limiter) * C_support)

    alpha_V = 0.50
    delta_V = 0.67

    V_driver = _clip01(alpha_V * Y3 + (1 - alpha_V) * Y4)
    V_reducer = _clip01(delta_V * Y1 + (1 - delta_V) * Y2)
    V = _clip01(V_driver * (1 - V_reducer))

    return dominant, scores, weights, C, V


def compute_forcing_term(DA, BH, TI, ENG, PU, SI, PS):
    dt = 0.01
    t = 800
    eta = 0.90

    _, _, _, C, V = compute_persona(DA, BH, TI, ENG, PU, SI, PS)
    S = _clip01(C * (1 - V))

    forcing = np.zeros(t)
    forcing[0] = 0.50

    for k in range(1, t):
        forcing[k] = forcing[k - 1] + eta * (S - forcing[k - 1]) * dt
        forcing[k] = _clip01(forcing[k])

    return float(forcing[-1])


def classify_forcing_term(value):
    value = round(value, 2)

    if value >= 0.70:
        return "Upgrade Soon"
    if value >= 0.30:
        return "Delay Upgrade"
    return "Churn Risk"


def recommend_actions(persona, intention):
    persona = (persona or "").strip()
    intention = (intention or "").strip()

    actions = ["Send general follow-up message."]

    if persona == "Loyalist":
        if intention == "Upgrade Soon":
            actions = ["Send priority upgrade offer.", "Give a small trade-in bonus."]
        elif intention == "Delay Upgrade":
            actions = ["Send a reminder.", "Offer a small accessory promotion."]
        else:
            actions = ["Request feedback.", "Send a loyalty thank-you coupon."]

    elif persona == "Fan":
        if intention == "Upgrade Soon":
            actions = ["Promote a device bundle.", "Highlight key feature improvements."]
        elif intention == "Delay Upgrade":
            actions = ["Explain the value of upgrading.", "Offer a modest trade-in top-up."]
        else:
            actions = ["Suggest a lower-priced device.", "Reduce communication frequency."]

    elif persona == "Switcher":
        if intention == "Upgrade Soon":
            actions = ["Highlight ecosystem benefits.", "Provide strong trade-in value."]
        elif intention == "Delay Upgrade":
            actions = ["Send comparison messaging.", "Offer a limited-time trade-in promotion."]
        else:
            actions = ["Send a win-back offer.", "Highlight long-term device value."]

    elif persona == "Drifter":
        if intention == "Upgrade Soon":
            actions = ["Suggest mid-range or refurbished devices.", "Keep communication simple."]
        elif intention == "Delay Upgrade":
            actions = ["Send an occasional generic promotion.", "Suggest older or budget devices."]
        else:
            actions = ["Send a final discount offer.", "Limit further marketing spend for this customer."]

    return actions


def build_scored_dataframe(raw_df):
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()

    if "id" not in df.columns:
        df["id"] = [f"C{i + 1:04d}" for i in range(len(df))]

    for col in INPUT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=INPUT_COLS).copy()

    personas = []
    score_list = []
    weight_list = []
    intentions = []
    forcing_terms = []
    Ns = []
    Bs = []
    Hs = []
    Cs = []
    Vs = []
    Ss = []
    actions_col = []

    for _, r in df.iterrows():
        DA, BH, TI, ENG, PU, SI, PS = map(
            float, [r.DA, r.BH, r.TI, r.ENG, r.PU, r.SI, r.PS]
        )

        persona, scores, weights, C, V = compute_persona(DA, BH, TI, ENG, PU, SI, PS)
        N, B, H = compute_behaviorals(DA, BH, TI, ENG, PU, SI, PS)
        S = _clip01(C * (1 - V))
        ft = round(compute_forcing_term(DA, BH, TI, ENG, PU, SI, PS), 3)
        intention = classify_forcing_term(ft)
        crm_actions = recommend_actions(persona, intention)

        personas.append(persona)
        score_list.append(scores)
        weight_list.append(weights)
        intentions.append(intention)
        forcing_terms.append(ft)
        Ns.append(N)
        Bs.append(B)
        Hs.append(H)
        Cs.append(C)
        Vs.append(V)
        Ss.append(S)
        actions_col.append(crm_actions)

    df["persona"] = personas
    df["persona_scores"] = score_list
    df["persona_weights"] = weight_list
    df["Need"] = Ns
    df["Bonding"] = Bs
    df["Hesitation"] = Hs
    df["Commitment"] = Cs
    df["Volatility"] = Vs
    df["ShortTermIntent"] = Ss
    df["forcing_term"] = forcing_terms
    df["long_term_upgrade_intent"] = forcing_terms
    df["intention"] = intentions
    df["decision"] = intentions
    df["crm_actions"] = actions_col

    return df


@st.cache_data
def load_source_csv():
    return pd.read_csv(DATASET_PATH)


@st.cache_data
def load_data_from_firestore():
    if db is None:
        return pd.DataFrame()

    docs = list(db.collection(TARGET_COLLECTION).stream())
    rows = []

    for doc in docs:
        d = doc.to_dict()
        rows.append(
            {
                "id": d.get("source_id", doc.id),
                "DA": d.get("DA"),
                "BH": d.get("BH"),
                "TI": d.get("TI"),
                "ENG": d.get("ENG"),
                "PU": d.get("PU"),
                "SI": d.get("SI"),
                "PS": d.get("PS"),
                "forcing_term": d.get("forcing_term"),
                "long_term_upgrade_intent": d.get("long_term_upgrade_intent"),
                "intention": d.get("intention", d.get("decision")),
                "created_at": d.get("created_at"),
                "crm_actions": d.get("crm_actions"),
            }
        )

    if not rows:
        return pd.DataFrame()

    return build_scored_dataframe(pd.DataFrame(rows))


def save_results_to_firestore(scored_df):
    if db is None:
        return 0

    saved = 0

    for _, r in scored_df.iterrows():
        out_doc = {
            "DA": float(r["DA"]),
            "BH": float(r["BH"]),
            "TI": float(r["TI"]),
            "ENG": float(r["ENG"]),
            "PU": float(r["PU"]),
            "SI": float(r["SI"]),
            "PS": float(r["PS"]),
            "Need": float(r["Need"]),
            "Bonding": float(r["Bonding"]),
            "Hesitation": float(r["Hesitation"]),
            "Commitment": float(r["Commitment"]),
            "Volatility": float(r["Volatility"]),
            "ShortTermIntent": float(r["ShortTermIntent"]),
            "forcing_term": float(r["forcing_term"]),
            "long_term_upgrade_intent": float(r["long_term_upgrade_intent"]),
            "intention": r["intention"],
            "decision": r["decision"],
            "persona": r["persona"],
            "persona_scores": r["persona_scores"],
            "persona_weights": r["persona_weights"],
            "crm_actions": r["crm_actions"],
            "source_id": str(r["id"]),
            "created_at": firestore.SERVER_TIMESTAMP,
        }

        db.collection(TARGET_COLLECTION).document(str(r["id"])).set(out_doc)
        saved += 1

    return saved


def get_active_dataframe():
    firestore_df = load_data_from_firestore()
    if not firestore_df.empty:
        return firestore_df, "firestore"

    source_df = load_source_csv()
    return build_scored_dataframe(source_df), "csv"


def radar_chart(scores_dict, title="Persona Radar"):
    labels = list(scores_dict.keys())
    values = list(scores_dict.values())
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(3.8, 3.8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.22)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
    ax.set_title(title, y=1.12, fontsize=10)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    return fig


def render_card(title, body):
    st.markdown(
        f"""
        <div class="soft-card">
            <div class="soft-card-title">{title}</div>
            <div class="soft-card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_action_list(actions):
    html_items = "".join([f"<li>{action}</li>" for action in actions])
    st.markdown(
        f"""
        <div class="action-card">
            <div class="action-title">Recommended CRM Actions</div>
            <ul>{html_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.025em;
        }

        h3 {
            margin-top: 0.5rem;
        }

        .hero-card {
            padding: 1.25rem 1.4rem;
            border-radius: 22px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            background: linear-gradient(180deg, rgba(248,250,252,0.96), rgba(241,245,249,0.72));
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: 1.85rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
            color: #0f172a;
        }

        .hero-subtitle {
            color: #475569;
            font-size: 0.98rem;
            margin-bottom: 0.85rem;
        }

        .pill {
            display: inline-block;
            padding: 0.26rem 0.62rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
            border-radius: 999px;
            background: #e2e8f0;
            color: #334155;
            font-size: 0.78rem;
            font-weight: 650;
        }

                div[data-testid="stMetric"] {
            background: rgba(248, 250, 252, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.28);
            padding: 0.85rem 1rem;
            border-radius: 16px;
            color: #0f172a !important;
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] label p {
            color: #0f172a !important;
            font-size: 0.82rem;
            font-weight: 650;
        }

        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] div {
            color: #0f172a !important;
        }

        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] div {
            color: #334155 !important;
        }

        .soft-card {
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            background: rgba(255,255,255,0.72);
            margin-bottom: 0.75rem;
        }

        .soft-card-title {
            font-weight: 750;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }

        .soft-card-body {
            color: #475569;
            font-size: 0.92rem;
            line-height: 1.45;
        }

        .action-card {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            background: rgba(248,250,252,0.82);
            margin-top: 0.75rem;
        }

        .action-title {
            font-weight: 750;
            color: #0f172a;
            margin-bottom: 0.35rem;
        }

        .action-card ul {
            margin-bottom: 0;
            padding-left: 1.1rem;
            color: #334155;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 0.55rem 0.85rem;
            border-radius: 999px;
        }

        .caption-text {
            color: #64748b;
            font-size: 0.86rem;
            margin-top: 0.15rem;
        }

        .section-note {
            color: #64748b;
            font-size: 0.92rem;
            margin-bottom: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Customer Upgrade Behavior Intelligent System</div>

    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Dashboard Controls")

    if st.button("Refresh data", use_container_width=True):
        load_data_from_firestore.clear()
        load_source_csv.clear()
        st.rerun()

    if db is None:
        st.info("Using local simulated CSV.")
    else:
        st.success("Firestore connected.")

df, data_source = get_active_dataframe()

tabs = st.tabs(
    ["Overview", "Persona Insights", "CRM Actions", "Customer Explorer", "Data Loader"]
)

tab_overview, tab_persona, tab_crm, tab_user, tab_loader = tabs

with tab_loader:
    st.subheader("Data Loader")
    st.markdown(
        """
        <div class="section-note">
            Upload a replacement CSV with the required customer behavior columns, then recompute the predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1.2])

    with c1:
        render_card(
            "Default Dataset",
            f"Current local dataset path: <code>{DATASET_PATH}</code>",
        )

        required_cols_df = pd.DataFrame(
            {
                "Column": ["id"] + INPUT_COLS,
                "Meaning": [
                    "Customer identifier",
                    "Device Age",
                    "Battery Health",
                    "Trade-In Value",
                    "Engagement",
                    "Past Upgrades",
                    "Social Influence",
                    "Price Sensitivity",
                ],
            }
        )
        st.markdown("**Required columns**")
        st.dataframe(required_cols_df, use_container_width=True, hide_index=True)

    with c2:
        uploaded = st.file_uploader("Upload replacement CSV", type=["csv"])

        preview_df = None
        if uploaded is not None:
            preview_df = pd.read_csv(uploaded)
        elif data_source == "csv":
            preview_df = load_source_csv()

        if preview_df is not None:
            with st.expander("Preview dataset", expanded=True):
                st.dataframe(preview_df.head(10), use_container_width=True)

        if uploaded is not None:
            missing = [c for c in ["id"] + INPUT_COLS if c not in preview_df.columns]

            if missing:
                st.error(f"Missing required columns: {missing}")
            elif st.button("Compute results and save", use_container_width=True):
                scored_df = build_scored_dataframe(preview_df)
                scored_df[["id"] + INPUT_COLS].to_csv(DATASET_PATH, index=False)
                load_source_csv.clear()

                saved = 0
                if db is not None:
                    with st.spinner("Saving computed results..."):
                        saved = save_results_to_firestore(scored_df)
                    load_data_from_firestore.clear()

                st.success(f"Processed {len(scored_df)} customer records.")
                if db is not None:
                    st.info(f"Saved {saved} records to Firestore.")
                else:
                    st.info("Saved dataset locally and refreshed dashboard.")
                st.rerun()
        else:
            st.info("Upload a CSV to replace the local simulated dataset.")

if df.empty:
    with tab_overview:
        st.warning("No records are available. Add a simulated dataset in the Data Loader tab.")

    with tab_persona:
        st.info("Persona insights will appear after data is loaded.")

    with tab_user:
        st.info("Customer Explorer will appear after data is loaded.")

    with tab_crm:
        st.info("CRM Actions will appear after data is loaded.")

    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("### Filters")

    selected_intentions = st.multiselect(
        "Predicted outcome",
        INTENTION_OPTIONS,
        default=INTENTION_OPTIONS,
    )

    selected_personas = st.multiselect(
        "Persona type",
        PERSONA_OPTIONS,
        default=PERSONA_OPTIONS,
    )

    forcing_min_val = float(df["forcing_term"].min())
    forcing_max_val = float(df["forcing_term"].max())

    forcing_min, forcing_max = st.slider(
        "Upgrade intent range",
        forcing_min_val,
        forcing_max_val,
        (forcing_min_val, forcing_max_val),
        step=0.05,
    )

filtered_df = df[
    df["intention"].isin(selected_intentions)
    & df["persona"].isin(selected_personas)
    & (df["forcing_term"] >= forcing_min)
    & (df["forcing_term"] <= forcing_max)
].copy()

total_customers = len(filtered_df)
avg_forcing = filtered_df["forcing_term"].mean() if total_customers else 0

upgrade_count = int((filtered_df["intention"] == "Upgrade Soon").sum())
delay_count = int((filtered_df["intention"] == "Delay Upgrade").sum())
churn_count = int((filtered_df["intention"] == "Churn Risk").sum())

upgrade_rate = upgrade_count / total_customers * 100 if total_customers else 0
delay_rate = delay_count / total_customers * 100 if total_customers else 0
churn_rate = churn_count / total_customers * 100 if total_customers else 0

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Total Customers", f"{total_customers}")

with m2:
    st.metric("Upgrade Soon", f"{upgrade_count}", f"{upgrade_rate:.1f}%")

with m3:
    st.metric("Delay Upgrade", f"{delay_count}", f"{delay_rate:.1f}%")

with m4:
    st.metric("Churn Risk", f"{churn_count}", f"{churn_rate:.1f}%")

st.markdown(
    f"""
    <div class="soft-card">
        <div class="soft-card-title">Average Upgrade Intent</div>
        <div class="soft-card-body">
            Average long-term upgrade intent for the selected segment: <strong>{avg_forcing:.3f}</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with tab_overview:
    st.subheader("Overview")
    st.markdown(
        """
        <div class="section-note">
            High-level distribution of upgrade intent and predicted customer outcomes for the selected segment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.35, 1])

    with c1:
        st.markdown("**Long-term upgrade intent distribution**")

        if not filtered_df.empty:
            arr = filtered_df["forcing_term"].to_numpy()
            fig_hist, ax_hist = plt.subplots(figsize=(5.8, 3.2))
            ax_hist.hist(arr, bins=10, edgecolor="black", alpha=0.85)
            ax_hist.set_xlabel("Upgrade intent")
            ax_hist.set_ylabel("Customers")
            ax_hist.grid(axis="y", linestyle="--", alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig_hist, use_container_width=False)
            st.caption("Shows how customer upgrade readiness is distributed after dynamic updates.")
        else:
            st.info("No upgrade intent values for the current filter.")

    with c2:
        st.markdown("**Prediction breakdown**")

        if not filtered_df.empty:
            intention_counts = filtered_df["intention"].value_counts().reindex(
                INTENTION_OPTIONS, fill_value=0
            )

            fig, ax = plt.subplots(figsize=(3.6, 3.6))
            ax.pie(
                intention_counts.values,
                labels=intention_counts.index,
                autopct="%1.0f%%",
                startangle=90,
                textprops={"fontsize": 8},
            )
            ax.axis("equal")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            st.caption("Breakdown of customers by predicted upgrade outcome.")
        else:
            st.info("No prediction data for the current filter.")

with tab_persona:
    st.subheader("Persona Insights")
    st.markdown(
        """
        <div class="section-note">
            Persona analysis explains how different behavioral groups relate to upgrade readiness and churn risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

    p_counts = filtered_df["persona"].value_counts().reindex(PERSONA_OPTIONS, fill_value=0)

    c1, c2 = st.columns([1, 1.4])

    with c1:
        st.markdown("**Persona distribution**")
        figp, axp = plt.subplots(figsize=(3.7, 3.7))
        axp.pie(
            p_counts.values,
            labels=p_counts.index,
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 8},
        )
        axp.axis("equal")
        plt.tight_layout()
        st.pyplot(figp, use_container_width=False)
        st.caption("Dominant customer personas in the selected segment.")

    with c2:
        st.markdown("**Average upgrade intent by persona**")
        persona_means = filtered_df.groupby("persona")[["forcing_term"]].mean()
        persona_means = persona_means.reindex(PERSONA_OPTIONS)
        st.bar_chart(persona_means, use_container_width=True)
        st.caption("Compares average upgrade readiness across personas.")

    with st.expander("Behavioral profile by persona", expanded=False):
        if not filtered_df.empty:
            st.markdown(
                """
                - **Need:** pressure created by device age, battery condition, trade-in value, and upgrade history  
                - **Bonding:** strength of customer engagement and ecosystem attachment  
                - **Hesitation:** resistance caused by price sensitivity and weaker upgrade signals  
                - **Commitment:** upgrade-leaning system state  
                - **Volatility:** delay or churn-leaning system state  
                """
            )

            beh_means = filtered_df.groupby("persona")[
                ["Need", "Bonding", "Hesitation", "Commitment", "Volatility"]
            ].mean()
            beh_means = beh_means.reindex(PERSONA_OPTIONS)
            st.bar_chart(beh_means, use_container_width=True)
        else:
            st.info("No data available to show behavioral profiles.")

with tab_crm:
    st.subheader("CRM Actions")
    st.markdown(
        """
        <div class="section-note">
            These actions are generated from the selected customer segment and predicted behavioral outcomes.
        </div>
        """,
        unsafe_allow_html=True,
    )

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

            c1, c2 = st.columns([1, 1.25])

            with c1:
                st.markdown("**CRM action frequency**")
                action_table = action_counts.rename("Count").to_frame()
                st.dataframe(action_table, use_container_width=True)

                export_df = filtered_df[["id", "persona", "intention", "crm_actions"]].copy()
                csv = export_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download selected CRM segment",
                    data=csv,
                    file_name="crm_segment_export.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with c2:
                st.markdown("**Top CRM actions**")
                top_n = min(8, len(action_counts))
                subset = action_counts.head(top_n)[::-1]
                wrapped_labels = [textwrap.fill(lbl, width=28) for lbl in subset.index]

                fig_act, ax_act = plt.subplots(figsize=(6.5, 3.8))
                ax_act.barh(range(len(subset)), subset.values)
                ax_act.set_yticks(range(len(subset)))
                ax_act.set_yticklabels(wrapped_labels, fontsize=8)
                ax_act.set_xlabel("Count")
                ax_act.set_title("Top CRM actions", fontsize=10)
                ax_act.grid(axis="x", linestyle="--", alpha=0.35)
                plt.tight_layout()
                st.pyplot(fig_act, use_container_width=False)
                st.caption("Most frequent CRM actions for the current filter.")
        else:
            st.info("This segment currently has no CRM actions.")
    else:
        st.info("No CRM action data available. Load and compute data first.")

with tab_user:
    st.subheader("Customer Explorer")
    st.markdown(
        """
        <div class="section-note">
            Inspect one customer record, model factors, persona scores, and CRM recommendations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_df.empty:
        st.info("No customers match the current filters.")
    else:
        selected_customer_id = st.selectbox(
            "Select customer ID",
            options=filtered_df["id"].tolist(),
        )

        customer_row = filtered_df[filtered_df["id"] == selected_customer_id].iloc[0]
        persona = customer_row["persona"]
        scores = customer_row["persona_scores"]
        intention = customer_row["intention"]

        left, right = st.columns([1, 1.15])

        with left:
            summary_html = f"""
                <strong>Customer ID:</strong> <code>{selected_customer_id}</code><br>
                <strong>Predicted outcome:</strong> {intention}<br>
                <strong>Upgrade intent:</strong> <code>{customer_row['forcing_term']:.3f}</code><br>
                <strong>Persona:</strong> {persona}
            """
            render_card("Customer Summary", summary_html)

            st.markdown("**Model factors**")
            factor_df = pd.DataFrame(
                {
                    "Factor": [
                        "Need",
                        "Bonding",
                        "Hesitation",
                        "Commitment",
                        "Volatility",
                        "Short-Term Intent",
                    ],
                    "Value": [
                        customer_row["Need"],
                        customer_row["Bonding"],
                        customer_row["Hesitation"],
                        customer_row["Commitment"],
                        customer_row["Volatility"],
                        customer_row["ShortTermIntent"],
                    ],
                }
            )

            st.bar_chart(factor_df, x="Factor", y="Value", use_container_width=True)

            action_list = customer_row.get("crm_actions") or recommend_actions(persona, intention)
            render_action_list(action_list)

        with right:
            st.markdown("**Persona radar**")
            fig = radar_chart(scores, title=f"{persona} profile")
            st.pyplot(fig, use_container_width=False)

            scores_df = pd.DataFrame(
                {
                    "Persona": list(scores.keys()),
                    "Score": [round(v, 3) for v in scores.values()],
                }
            )
            st.markdown("**Persona scores**")
            st.dataframe(scores_df, use_container_width=True, hide_index=True)