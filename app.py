from flask import Flask, request, jsonify
import numpy as np
from transformers import RobertaTokenizer
import onnxruntime as ort

app = Flask(__name__)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
session = ort.InferenceSession("model.onnx")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        inputs = tokenizer.encode_plus(text, return_tensors="np")
        
        # ربط المدخلات بذكاء حسب ما يتوقعه نموذج ONNX الفعلي
        ort_inputs = {session.get_inputs()[0].name: inputs['input_ids'].astype(np.int64)}
        
        # إذا كان النموذج يدعم مدخلاً ثانياً، نمرره، وإلا نتجاهله لمنع الانهيار
        if len(session.get_inputs()) > 1:
            ort_inputs[session.get_inputs()[1].name] = inputs['attention_mask'].astype(np.int64)
            
        outputs = session.run(None, ort_inputs)
        prediction = int(np.argmax(outputs[0]))
        
        result = {"positive": bool(prediction)}
        return jsonify(result)
        
    except Exception as e:
        # إرجاع الخطأ كـ JSON بدلاً من صفحة ويب ليسهل علينا قراءته في الـ Terminal
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)