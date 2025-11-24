import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

CSV_FILE = "C:\\Users\\banya\\OneDrive\\Desktop\\ICT338\\apple_user_dataset.csv"
SERVICE_ACCOUNT = "serviceAccountKey.json"
TARGET_COLLECTION = "apple_upgrade_predictions"

# ---------- Firebase init ----------
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT)
    firebase_admin.initialize_app(cred)

db = firestore.client()
print("✅ Connected to Firestore")


# ---------- Helper: Persona layer (H1–H4) ----------
def compute_personas(N, B, H):
    """
    Personas derived from behavioral signals:
    H1 = Loyalist
    H2 = Fan
    H3 = Switcher
    H4 = Drifter
    """
    H1_loyalist = 0.6 * N + 0.3 * B - 0.5 * H
    H2_fan      = 0.7 * B + 0.2 * N - 0.3 * H
    H3_switcher = 0.6 * H - 0.3 * B + 0.1 * N
    H4_drifter  = 1.0 * H - 0.5 * N - 0.2 * B

    return H1_loyalist, H2_fan, H3_switcher, H4_drifter


# ---------- 1. Forcing-term model (unchanged) ----------
def compute_forcing_term(DA, BH, TI, ENG, PU, SI, PS):
    dt = 0.01
    eta = 0.9
    alpha = 0.7
    omega = 0.5
    t = 800

    X = np.zeros(t)
    Y = np.zeros(t)
    S = np.zeros(t)
    forcing_term = np.zeros(t)

    # initial conditions (your corrected version)
    X[0] = alpha * (1 - BH) + (1 - alpha) * DA
    Y[0] = (omega * DA + omega * (1 - BH)) * PS
    S[0] = X[0] * (1 - Y[0])
    forcing_term[0] = 0.1

    # Need / Bonding / Hesitation are constant per user
    N = (DA + TI + ENG + PU + SI) / 5.0
    B = (ENG + PU + SI) / 3.0
    H = ((1 - DA) + BH + (1 - TI) + (1 - ENG) +
         (1 - PU) + (1 - SI) + PS * (1 - TI)) / 7.0

    # simulation loop (same as yours)
    for k in range(1, t):
        X[k] = alpha * B + (1 - alpha) * N - (alpha * H)
        Y[k] = (omega * N + omega * B) * H
        S[k] = X[k] * (1 - Y[k])

        forcing_term[k] = forcing_term[k - 1] + eta * (S[k - 1] - forcing_term[k - 1]) * dt

    return float(forcing_term[-1]), N, B, H


# ---------- 2. Decision rule ----------
def classify_forcing_term(value: float) -> str:
    value = round(value, 2)
    if value >= 0.60:
        return "Upgrade Soon"
    elif value >= 0.10:
        return "Delay Upgrade"
    else:
        return "Churn Risk"


# ---------- 3. Process CSV → Compute → Save ----------
def main():
    print(f"📄 Loading CSV: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)

    for _, row in df.iterrows():
        user_id = str(row['id'])

        DA  = float(row['DA'])
        BH  = float(row['BH'])
        TI  = float(row['TI'])
        ENG = float(row['ENG'])
        PU  = float(row['PU'])
        SI  = float(row['SI'])
        PS  = float(row['PS'])

        # compute forcing term + behavioral signals
        raw_value, N, B, H = compute_forcing_term(DA, BH, TI, ENG, PU, SI, PS)
        forcing_value = round(raw_value, 3)

        # persona layer
        H1, H2, H3, H4 = compute_personas(N, B, H)

        # decision
        decision = classify_forcing_term(forcing_value)

        out_doc = {
            # inputs
            "DA": DA, "BH": BH, "TI": TI, "ENG": ENG,
            "PU": PU, "SI": SI, "PS": PS,

            # behavioral layer
            "Need_N": round(N, 3),
            "Bonding_B": round(B, 3),
            "Hesitation_H": round(H, 3),

            # personas
            "H1_Loyalist": round(H1, 3),
            "H2_Fan": round(H2, 3),
            "H3_Switcher": round(H3, 3),
            "H4_Drifter": round(H4, 3),

            # output
            "forcing_term": forcing_value,
            "decision": decision,

            "source_id": user_id,
            "created_at": firestore.SERVER_TIMESTAMP,
        }

        db.collection(TARGET_COLLECTION).document(user_id).set(out_doc)
        print(f"✅ Saved {user_id}: {forcing_value} → {decision} | "
              f"H1={H1:.2f}, H2={H2:.2f}, H3={H3:.2f}, H4={H4:.2f}")

    print("🎉 DONE — All CSV rows processed.")

if __name__ == "__main__":
    main()
