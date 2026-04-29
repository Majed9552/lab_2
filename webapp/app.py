from flask import Flask, request, jsonify
import numpy as np
from transformers import RobertaTokenizer
import onnxruntime as ort
import sys

app = Flask(__name__)

print("--- Starting initialization ---", flush=True)
try:
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    print("Tokenizer loaded.", flush=True)
    session = ort.InferenceSession("model.onnx")
    print("ONNX Model loaded successfully.", flush=True)
except Exception as e:
    print(f"Initialization Error: {e}", flush=True)
    sys.exit(1)

@app.route("/predict", methods=["POST"])
def predict():
    print("--- Request received ---", flush=True)
    try:
        data = request.get_json()
        text = data.get("text", "")
        print(f"Processing text: {text}", flush=True)
        
        inputs = tokenizer.encode_plus(text, return_tensors="np")
        
        # تحويل البيانات بشكل صريح لضمان التوافق
        input_ids = inputs['input_ids'].astype(np.int64)
        
        ort_inputs = {session.get_inputs()[0].name: input_ids}
        
        if len(session.get_inputs()) > 1:
            attention_mask = inputs['attention_mask'].astype(np.int64)
            ort_inputs[session.get_inputs()[1].name] = attention_mask
            
        print("Running inference...", flush=True)
        outputs = session.run(None, ort_inputs)
        print("Inference completed.", flush=True)
        
        prediction = int(np.argmax(outputs[0]))
        return jsonify({"positive": bool(prediction)})
        
    except Exception as e:
        print(f"Prediction Error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)