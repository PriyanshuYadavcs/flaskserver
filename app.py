from flask import Flask, request, jsonify
from flask_cors import CORS 
import tensorflow as tf 
import numpy as np 
import cv2, base64 , io
from PIL import Image 
import os 
from categories import categories

app= Flask(__name__)
CORS(app)

#loading the model ok 
model= tf.keras.models.load_model("model/quickdraw_model_drive.h5")

@app.route("/") 
def home():
    return "flask server is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data= request.get_json()
        image_b64= data.get("imageData")

        image_bytes= base64.b64decode(image_b64.split(",")[1])
        image= np.array(Image.open(io.BytesIO(image_bytes)).convert("L"))


        #inverting if the image is dark; 
        if(np.mean(image)<127):
            image= 255-image 
                    # Otsu thresholding
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find stroke bounds
        ys, xs = np.where(image < 255)
        if len(xs) == 0 or len(ys) == 0:
            return jsonify({"error": "No drawing found"}), 400

        # Crop & resize to 28x28
        crop = image[ys.min():ys.max()+1, xs.min():xs.max()+1]
        h, w = crop.shape
        scale = min(28 / w, 28 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(crop, (new_w, new_h))
        final = np.ones((28, 28), dtype=np.uint8) * 255
        x_offset, y_offset = (28 - new_w) // 2, (28 - new_h) // 2
        final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        # Normalize
        final = 255 - final
        normalized = 1.0 - (final.astype(np.float32) / 255.0)
        img_array = np.expand_dims(normalized, axis=(0, -1))


        #predict 
        pred= model.predict(img_array)[0]
        top_indices=pred.argsort()[-10:][::-1]
        results= [{"label":categories[i], "prob": float(pred[i])} for i in top_indices]

        return jsonify({"predictions": results})
    
    except Exception as e: 
        return jsonify({"error": str(e)}),500
    


#0.0.0.0 ka mtlb h ki puri internet me se kahi se bhi request ajaye toh ok h 
if __name__=="__main__":
    app.run(host="0.0.0.0", port= int(os.environ.get("PORT", 10000)))