import gradio as gr
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# ✅ Train model
def train_model():
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = X @ [0.2, 1.5, 10.0] + np.random.randn(100) * 0.5
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

model = train_model()

# ✅ Prediction
def predict_energy(batch_size, temperature, load):
    input_data = np.array([[batch_size, temperature, load]])
    prediction = model.predict(input_data)
    return f"⚡ Estimated Energy Consumption: {prediction[0]:.2f} kWh"

# 🎨 Vibrant CSS
custom_css = """
body {
    background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 50%, #fbc2eb 100%);
    font-family: 'Segoe UI', sans-serif;
    background-attachment: fixed;
}
#title {
    text-align: center;
    color: #ffffff;
    font-size: 2.8em;
    font-weight: bold;
    padding: 20px;
    text-shadow: 1px 1px 6px #000;
}
.card {
    background-color: rgba(255, 255, 255, 0.95);
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 0px 18px rgba(0,0,0,0.25);
    height: 100%;
}
.gr-button {
    font-weight: bold;
    font-size: 1.1em;
}
.footer {
    text-align: center;
    color: #222;
    font-size: 14px;
    margin-top: 40px;
}
"""

# 🚀 Build Gradio App
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 id='title'>🔋 Battery Energy Predictor</h1>")
    gr.Markdown("<p style='text-align:center; color:white; font-size:18px;'>Enter parameters to predict battery cell manufacturing energy usage.</p>")

    with gr.Row():
        with gr.Column(scale=1, elem_classes="card"):
            gr.Markdown("### 🧠 How It Works:")
            gr.Markdown(
                "- Fill in batch size, machine load, and temperature.\n"
                "- Press **Predict Energy** to get estimated energy consumption in kWh.\n"
                "- Powered by Random Forest ML model (trained on dummy data)."
            )

        with gr.Column(scale=1, elem_classes="card"):
            batch_size = gr.Number(label="📦 Batch Size", value=250)
            temperature = gr.Number(label="🌡 Temperature (°C)", value=30.0)
            load = gr.Number(label="⚙️ Machine Load", value=1.0)
            submit_btn = gr.Button("🚀 Predict Energy", variant="primary")
            clear_btn = gr.Button("🧹 Reset", variant="secondary")

        with gr.Column(scale=1, elem_classes="card"):
            result = gr.Textbox(label="📊 Estimated Output", interactive=False, lines=2)
            gr.Markdown("#### 📌 Model Info:")
            gr.Markdown("- Algorithm: Random Forest\n- Trained samples: 100\n- Output: **Energy (kWh)**")

    submit_btn.click(predict_energy, inputs=[batch_size, temperature, load], outputs=result)
    clear_btn.click(lambda: (250, 30.0, 1.0, ""), outputs=[batch_size, temperature, load, result])

    gr.Markdown("<div class='footer'>✨ Created by Team Turtle 🐢 • Open Source on Hugging Face • Har Har Mahadev 🔱</div>")

demo.launch()
