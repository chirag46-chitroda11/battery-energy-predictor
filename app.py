import gradio as gr
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Train dummy model
def train_model():
    X = np.random.rand(100, 3)
    y = X @ [0.2, 1.5, 10.0] + np.random.randn(100) * 0.5
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

model = train_model()

# Prediction for single input
def predict_energy(batch_size, temperature, load):
    input_data = np.array([[batch_size, temperature, load]])
    prediction = model.predict(input_data)[0]

    # Line chart of Energy vs Batch Size
    batch_range = np.linspace(100, 500, 50)
    inputs = np.array([[b, temperature, load] for b in batch_range])
    preds = model.predict(inputs)

    plt.figure(figsize=(6, 3.5))
    plt.plot(batch_range, preds, color="orange", marker="o", label="Predicted")
    plt.axvline(x=batch_size, color="blue", linestyle="--", label="Your Batch")
    plt.title("Predicted Energy vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Energy (kWh)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return f"⚡ Estimated Energy Consumption: {prediction:.2f} kWh", plt

# Batch prediction using CSV
def predict_from_csv(file):
    try:
        df = pd.read_csv(file.name)
        if not {"Batch Size", "Temperature", "Load"}.issubset(df.columns):
            return "❌ CSV must contain columns: Batch Size, Temperature, Load"
        X = df[["Batch Size", "Temperature", "Load"]]
        y_pred = model.predict(X)
        df["Predicted Energy (kWh)"] = y_pred
        return df
    except Exception as e:
        return f"❌ Error: {e}"

# Theme toggle logic
def get_theme(toggle):
    return "soft" if toggle == "Light" else "dark"

# UI Blocks
with gr.Blocks(theme=get_theme("Light")) as demo:
    gr.Markdown("## 🔋 **Battery Cell Energy Predictor**")
    gr.Markdown("Use this tool to estimate energy consumption for a battery cell batch. You can also upload CSV for batch predictions.")

    with gr.Row():
        with gr.Column():
            batch = gr.Slider(100, 500, value=250, label="📦 Batch Size")
            temp = gr.Slider(20, 40, value=30, label="🌡 Temperature (°C)")
            load = gr.Slider(0.5, 1.5, value=1.0, label="⚙️ Machine Load")
            theme_toggle = gr.Radio(["Light", "Dark"], value="Light", label="🎨 Theme")
            predict_btn = gr.Button("🚀 Predict Energy")
            reset_btn = gr.Button("🧹 Reset")

        with gr.Column():
            output_text = gr.Textbox(label="🔋 Estimated Output")
            output_plot = gr.Plot(label="📊 Energy vs Batch Size Chart")

    with gr.Row():
        gr.Markdown("### 📤 Upload CSV for Batch Predictions")
        csv_input = gr.File(file_types=[".csv"], label="Upload CSV with columns: Batch Size, Temperature, Load")
        csv_output = gr.Dataframe(label="📄 Prediction Output")

    # Event bindings
    predict_btn.click(
        fn=predict_energy,
        inputs=[batch, temp, load],
        outputs=[output_text, output_plot]
    )

    reset_btn.click(
        fn=lambda: ("", None),
        inputs=[],
        outputs=[output_text, output_plot]
    )

    csv_input.change(
        fn=predict_from_csv,
        inputs=csv_input,
        outputs=csv_output
    )

    theme_toggle.change(
        fn=lambda x: demo.set_theme(get_theme(x)),
        inputs=theme_toggle
    )

demo.launch()
