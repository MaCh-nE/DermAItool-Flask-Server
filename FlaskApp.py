## Flask and request handling tools import
from flask import Flask, request, jsonify

import PredictionLoader
import GradCAM_Saver
from collections import OrderedDict

app = Flask(__name__)

## REST API :
# only POST methods since its the prediction loading server, and its *receiving* data (image path) via JSON
# 1st -> Receives path, loads predicted lesion via JSON's body
# 2nd -> Receives path, loads probability array

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    path = data["path"]
    result = PredictionLoader.GetPrediction(path)
    return jsonify(result)

@app.route('/predictions', methods=['POST'])
def predictions():
    data = request.get_json()
    path = data["path"]
    return PredictionLoader.GetPredictionProbs(path)


@app.route('/gradCAM', methods=['POST'])
def grad_cam():
    data = request.get_json()
    path = data["path"]
    id = int(data["imageId"])
    colormap = data["colormap"]
    alpha = float(data["alpha"])
    GradCAM_Saver.save_gradcam(path, id, colormap, alpha)
    return jsonify("GRAD-Cam method succefully applied.")

@app.route("/test", methods=['POST'])
def testi():
    return jsonify("GOOD !")


if __name__ == '__main__':
    app.run(debug = True)