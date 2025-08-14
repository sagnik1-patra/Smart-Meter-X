import os, json, gradio as gr
from infer import forecast_next24, detect_anomalies

ART_DIR = os.path.dirname(__file__)

def do_forecast(data_dir):
    try:
        res = forecast_next24(data_dir, ART_DIR)
        return f"OK\nPlot: {res['plot_path']}", res["plot_path"]
    except Exception as e:
        return f"ERROR: {e}", None

def do_anomaly(data_dir):
    try:
        res = detect_anomalies(data_dir, ART_DIR)
        msg = f"OK\nCSV: {res['csv_path']}\nPlot: {res['plot_path']}"
        return msg, res["plot_path"]
    except Exception as e:
        return f"ERROR: {e}", None

with gr.Blocks(title="SmartMeterX Inference") as demo:
    gr.Markdown("# SmartMeterX — Forecast & Anomaly")
    with gr.Tab("Forecast next 24h"):
        data_dir_f = gr.Textbox(label="AMPds folder", value=r"C:\Users\sagni\Downloads\SmartMeterX\archive (2)")
        out_txt_f = gr.Textbox(label="Status / Paths")
        out_img_f = gr.Image(label="Forecast Plot")
        btn_f = gr.Button("Run Forecast")
        btn_f.click(fn=do_forecast, inputs=data_dir_f, outputs=[out_txt_f, out_img_f])

    with gr.Tab("Anomaly Detection"):
        data_dir_a = gr.Textbox(label="AMPds folder", value=r"C:\Users\sagni\Downloads\SmartMeterX\archive (2)")
        out_txt_a = gr.Textbox(label="Status / Paths")
        out_img_a = gr.Image(label="Anomaly Timeline")
        btn_a = gr.Button("Run Anomaly")
        btn_a.click(fn=do_anomaly, inputs=data_dir_a, outputs=[out_txt_a, out_img_a])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False)
