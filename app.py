"""
Smart Grid Stability Classifier - Gradio App
"""

import pickle
import numpy as np
import gradio as gr

# Load model
with open("grid_stability_model.pkl", "rb") as f:
    model_package = pickle.load(f)

model = model_package["model"]
class_names = model_package["class_names"]
metrics = model_package["metrics"]

# Feature engineering
def engineer_features(tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4):
    total_reaction_time = tau1 + tau2 + tau3 + tau4
    reaction_time_variance = float(np.var([tau1, tau2, tau3, tau4]))
    avg_consumer_tau = (tau2 + tau3 + tau4) / 3
    producer_consumer_ratio = tau1 / avg_consumer_tau
    avg_price_elasticity = (g1 + g2 + g3 + g4) / 4
    net_power_balance = p1 + p2 + p3 + p4
    tau1_x_g1 = tau1 * (1 - g1)
    return [
        tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4,
        total_reaction_time, reaction_time_variance, producer_consumer_ratio,
        avg_price_elasticity, net_power_balance, tau1_x_g1
    ], {
        "total_reaction_time": round(total_reaction_time, 3),
        "reaction_time_variance": round(reaction_time_variance, 3),
        "producer_consumer_ratio": round(producer_consumer_ratio, 3),
        "avg_price_elasticity": round(avg_price_elasticity, 3),
        "net_power_balance": round(net_power_balance, 6),
        "tau1_x_g1": round(tau1_x_g1, 3),
    }

# Prediction
def predict(tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4):
    features, engineered = engineer_features(tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4)
    X = np.array(features).reshape(1, -1)
    pred_index = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    prediction = class_names[pred_index]
    confidence = float(pred_proba.max())
    stable_prob = float(pred_proba[0])
    unstable_prob = float(pred_proba[1])

    if prediction == "unstable":
        if confidence >= 0.80:
            color = "#ff3b6b"
            bg = "rgba(255,59,107,0.10)"
            border = "#ff3b6b"
            status = "UNSTABLE - CRITICAL"
            action = "Immediate automated protective action required. Load shedding or circuit rerouting should be triggered now."
        else:
            color = "#ffa502"
            bg = "rgba(255,165,2,0.10)"
            border = "#ffa502"
            status = "UNSTABLE - WARNING"
            action = "Monitor grid closely. Prepare protective measures. Human operator review recommended."
    else:
        if confidence >= 0.75:
            color = "#2ed573"
            bg = "rgba(46,213,115,0.10)"
            border = "#2ed573"
            status = "STABLE - NORMAL"
            action = "Normal operation. No intervention required."
        else:
            color = "#ffb700"
            bg = "rgba(255,183,0,0.10)"
            border = "#ffb700"
            status = "STABLE - MONITOR"
            action = "Grid appears stable but confidence is low. Increase monitoring frequency."

    eng_rows = "".join([
        f"<tr>"
        f"<td style='padding:5px 20px 5px 0;color:#94a3b8;font-family:monospace;font-size:13px;'>{k}</td>"
        f"<td style='padding:5px 0;color:#60a5fa;font-family:monospace;font-size:13px;font-weight:700;'>{v}</td>"
        f"</tr>"
        for k, v in engineered.items()
    ])

    return f"""
<div style='font-family:system-ui,sans-serif;'>

  <div style='background:{bg};border:2px solid {border};border-radius:16px;padding:28px 32px;margin-bottom:20px;'>
    <div style='color:{color};font-size:28px;font-weight:800;letter-spacing:1px;margin-bottom:6px;'>{status}</div>
    <div style='color:#94a3b8;font-size:13px;margin-bottom:14px;'>
      Confidence: <strong style='color:{color};font-size:16px;'>{confidence*100:.1f}%</strong>
    </div>
    <div style='background:#1e293b;border-radius:8px;height:10px;overflow:hidden;margin-bottom:16px;'>
      <div style='background:{color};height:100%;width:{int(confidence*100)}%;border-radius:8px;'></div>
    </div>
    <p style='color:#cbd5e1;font-size:15px;margin:0;line-height:1.7;'>{action}</p>
  </div>

  <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;'>

    <div style='background:#111827;border:1px solid #1e293b;border-radius:14px;padding:20px;'>
      <div style='color:#94a3b8;font-size:13px;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;font-family:monospace;'>Class Probabilities</div>
      <div style='margin-bottom:14px;'>
        <div style='display:flex;justify-content:space-between;font-size:14px;font-weight:600;margin-bottom:6px;'>
          <span style='color:#34d399;'>Stable</span>
          <span style='color:#34d399;font-family:monospace;'>{stable_prob:.4f}</span>
        </div>
        <div style='background:#1e293b;border-radius:6px;height:8px;overflow:hidden;'>
          <div style='background:#34d399;height:100%;width:{int(stable_prob*100)}%;border-radius:6px;'></div>
        </div>
      </div>
      <div>
        <div style='display:flex;justify-content:space-between;font-size:14px;font-weight:600;margin-bottom:6px;'>
          <span style='color:#f87171;'>Unstable</span>
          <span style='color:#f87171;font-family:monospace;'>{unstable_prob:.4f}</span>
        </div>
        <div style='background:#1e293b;border-radius:6px;height:8px;overflow:hidden;'>
          <div style='background:#f87171;height:100%;width:{int(unstable_prob*100)}%;border-radius:6px;'></div>
        </div>
      </div>
    </div>

    <div style='background:#111827;border:1px solid #1e293b;border-radius:14px;padding:20px;'>
      <div style='color:#94a3b8;font-size:13px;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;font-family:monospace;'>Engineered Features</div>
      <table style='border-collapse:collapse;width:100%;'>{eng_rows}</table>
    </div>

  </div>
</div>
"""

# Presets
PRESETS = {
    "Stable - High Tau, Balanced Elasticity": {
        "tau": [9.304, 4.903, 3.048, 1.369],
        "p": [5.068, -1.940, -1.873, -1.255],
        "g": [0.413, 0.862, 0.562, 0.782],
        "why": "High reaction times but well-balanced elasticity keeps the grid stable."
    },
    "Stable - Mixed Tau, Low Elasticity": {
        "tau": [5.930, 6.731, 6.245, 0.533],
        "p": [2.327, -0.703, -1.117, -0.508],
        "g": [0.240, 0.563, 0.164, 0.754],
        "why": "Low elasticity values yet still stable — moderate power balance compensates."
    },
    "Unstable - Asymmetric Reaction Times": {
        "tau": [2.959, 3.080, 8.381, 9.781],
        "p": [3.763, -0.783, -1.257, -1.723],
        "g": [0.650, 0.860, 0.887, 0.958],
        "why": "tau3 and tau4 are 3x slower than tau1 and tau2 — dangerous coordination failure between nodes."
    },
    "Unstable - Very Fast Producer, Slow Consumer": {
        "tau": [0.716, 7.670, 4.487, 2.341],
        "p": [3.964, -1.027, -1.939, -0.997],
        "g": [0.446, 0.977, 0.929, 0.363],
        "why": "tau1=0.716 (very fast producer) but tau2=7.670 (very slow consumer) — system cannot coordinate."
    },
    "Unstable - Low Producer Elasticity": {
        "tau": [8.972, 8.848, 3.046, 1.215],
        "p": [3.405, -1.207, -1.277, -0.920],
        "g": [0.163, 0.767, 0.839, 0.110],
        "why": "g1=0.163 — producer barely responds to price signals, self-correction mechanism fails."
    }
}

# CSS
css = """
.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
    padding: 32px 24px !important;
    background: #0a0f1a !important;
}
.gr-button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
footer { display: none !important; }
"""

# UI
with gr.Blocks(title="Smart Grid Stability Classifier", css=css) as app:

    gr.HTML("""
<div style='text-align:center;padding:28px 0 24px;border-bottom:1px solid #1e293b;margin-bottom:28px;'>
  <div style='font-size:11px;font-family:monospace;letter-spacing:4px;color:#3b82f6;text-transform:uppercase;margin-bottom:12px;'>
    Saudi Vision 2030 — Smart Grid Modernization
  </div>
  <h1 style='font-size:40px;font-weight:800;color:#f1f5f9;letter-spacing:-1px;margin:0 0 8px;'>
    Smart Grid Stability Classifier
  </h1>
  <p style='color:#64748b;font-size:15px;margin:0 0 24px;'>
    Predicting electrical grid stability using machine learning — UCI Dataset
  </p>
  <div style='display:flex;justify-content:center;gap:12px;flex-wrap:wrap;'>
    <div style='background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:12px 24px;'>
      <div style='color:#60a5fa;font-size:22px;font-weight:800;'>XGBoost</div>
      <div style='color:#94a3b8;font-size:11px;letter-spacing:2px;margin-top:3px;'>MODEL</div>
    </div>
    <div style='background:#111827;border:1px solid #065f46;border-radius:12px;padding:12px 24px;'>
      <div style='color:#34d399;font-size:22px;font-weight:800;'>95.3%</div>
      <div style='color:#94a3b8;font-size:11px;letter-spacing:2px;margin-top:3px;'>ACCURACY</div>
    </div>
    <div style='background:#111827;border:1px solid #065f46;border-radius:12px;padding:12px 24px;'>
      <div style='color:#34d399;font-size:22px;font-weight:800;'>99.2%</div>
      <div style='color:#94a3b8;font-size:11px;letter-spacing:2px;margin-top:3px;'>ROC-AUC</div>
    </div>
    <div style='background:#111827;border:1px solid #065f46;border-radius:12px;padding:12px 24px;'>
      <div style='color:#34d399;font-size:22px;font-weight:800;'>97.6%</div>
      <div style='color:#94a3b8;font-size:11px;letter-spacing:2px;margin-top:3px;'>UNSTABLE RECALL</div>
    </div>
    <div style='background:#111827;border:1px solid #78350f;border-radius:12px;padding:12px 24px;'>
      <div style='color:#fbbf24;font-size:22px;font-weight:800;'>10K</div>
      <div style='color:#94a3b8;font-size:11px;letter-spacing:2px;margin-top:3px;'>TRAINING SAMPLES</div>
    </div>
  </div>
</div>
""")

    gr.HTML("<div style='color:#94a3b8;font-size:15px;font-family:monospace;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;'>Quick Presets — Real examples from dataset</div>")

    with gr.Row():
        btn_s1 = gr.Button("Stable - High Tau, Balanced Elasticity", variant="secondary", size="sm")
        btn_s2 = gr.Button("Stable - Mixed Tau, Low Elasticity", variant="secondary", size="sm")

    with gr.Row():
        btn_u1 = gr.Button("Unstable - Asymmetric Reaction Times", variant="secondary", size="sm")
        btn_u2 = gr.Button("Unstable - Very Fast Producer, Slow Consumer", variant="secondary", size="sm")
        btn_u3 = gr.Button("Unstable - Low Producer Elasticity", variant="secondary", size="sm")

    preset_info = gr.HTML("")

    gr.HTML("<div style='height:16px'></div>")
    gr.HTML("<div style='color:#94a3b8;font-size:15px;font-family:monospace;letter-spacing:2px;text-transform:uppercase;margin-bottom:14px;'>Grid Parameters</div>")

    with gr.Row():
        with gr.Column():
            gr.HTML("<div style='color:#60a5fa;font-size:14px;font-weight:700;margin-bottom:4px;'>Reaction Times (tau) — seconds</div><div style='color:#64748b;font-size:12px;margin-bottom:12px;'>Lower = faster response = safer</div>")
            tau1 = gr.Slider(0.5, 10.0, value=2.959, step=0.001, label="tau1 — Producer")
            tau2 = gr.Slider(0.5, 10.0, value=3.079, step=0.001, label="tau2 — Consumer 1")
            tau3 = gr.Slider(0.5, 10.0, value=8.381, step=0.001, label="tau3 — Consumer 2")
            tau4 = gr.Slider(0.5, 10.0, value=9.780, step=0.001, label="tau4 — Consumer 3")

        with gr.Column():
            gr.HTML("<div style='color:#60a5fa;font-size:14px;font-weight:700;margin-bottom:4px;'>Price Elasticity (g)</div><div style='color:#64748b;font-size:12px;margin-bottom:12px;'>Higher = more responsive = safer</div>")
            g1 = gr.Slider(0.05, 1.0, value=0.650, step=0.001, label="g1 — Producer")
            g2 = gr.Slider(0.05, 1.0, value=0.859, step=0.001, label="g2 — Consumer 1")
            g3 = gr.Slider(0.05, 1.0, value=0.887, step=0.001, label="g3 — Consumer 2")
            g4 = gr.Slider(0.05, 1.0, value=0.958, step=0.001, label="g4 — Consumer 3")

    with gr.Row():
        with gr.Column():
            gr.HTML("<div style='color:#60a5fa;font-size:14px;font-weight:700;margin-bottom:4px;'>Power (p)</div><div style='color:#64748b;font-size:12px;margin-bottom:12px;'>Producer positive — Consumers negative</div>")
            p1 = gr.Slider(0.0, 6.0, value=3.763, step=0.001, label="p1 — Producer output")
            p2 = gr.Slider(-2.0, 0.0, value=-0.782, step=0.001, label="p2 — Consumer 1 draw")
            p3 = gr.Slider(-2.0, 0.0, value=-1.257, step=0.001, label="p3 — Consumer 2 draw")
            p4 = gr.Slider(-2.0, 0.0, value=-1.723, step=0.001, label="p4 — Consumer 3 draw")

    gr.HTML("<div style='height:12px'></div>")
    predict_btn = gr.Button("Predict Grid Stability", variant="primary", size="lg")
    gr.HTML("<div style='height:16px'></div>")
    output = gr.HTML()

    gr.HTML("""
<div style='margin-top:36px;padding-top:24px;border-top:1px solid #1e293b;display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;'>
  <div>
    <div style='color:#94a3b8;font-size:12px;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;font-family:monospace;'>Dataset</div>
    <div style='color:#cbd5e1;font-size:14px;line-height:1.7;'>UCI Electrical Grid Stability<br>10,000 samples · 12 features</div>
  </div>
  <div>
    <div style='color:#94a3b8;font-size:12px;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;font-family:monospace;'>Features</div>
    <div style='color:#cbd5e1;font-size:14px;line-height:1.7;'>12 raw + 6 engineered<br>= 18 total input features</div>
  </div>
  <div>
    <div style='color:#94a3b8;font-size:12px;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;font-family:monospace;'>Alignment</div>
    <div style='color:#cbd5e1;font-size:14px;line-height:1.7;'>Saudi Vision 2030<br>Smart Grid Modernization</div>
  </div>
</div>
""")

    all_sliders = [tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4]
    predict_btn.click(fn=predict, inputs=all_sliders, outputs=output)

    def load_preset(name):
        p = PRESETS[name]
        info = f"<div style='background:#111827;border:1px solid #1e293b;border-radius:10px;padding:12px 18px;margin:8px 0;'><strong style='color:#60a5fa;font-size:14px;'>{name}</strong><br><span style='color:#64748b;font-size:13px;line-height:1.6;'>{p['why']}</span></div>"
        return (
            p["tau"][0], p["tau"][1], p["tau"][2], p["tau"][3],
            p["p"][0], p["p"][1], p["p"][2], p["p"][3],
            p["g"][0], p["g"][1], p["g"][2], p["g"][3],
            info
        )

    po = all_sliders + [preset_info]
    btn_s1.click(fn=lambda: load_preset("Stable - High Tau, Balanced Elasticity"), outputs=po)
    btn_s2.click(fn=lambda: load_preset("Stable - Mixed Tau, Low Elasticity"), outputs=po)
    btn_u1.click(fn=lambda: load_preset("Unstable - Asymmetric Reaction Times"), outputs=po)
    btn_u2.click(fn=lambda: load_preset("Unstable - Very Fast Producer, Slow Consumer"), outputs=po)
    btn_u3.click(fn=lambda: load_preset("Unstable - Low Producer Elasticity"), outputs=po)

if __name__ == "__main__":
    app.launch(share=True)
