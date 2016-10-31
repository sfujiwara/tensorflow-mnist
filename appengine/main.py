# -*- coding: utf-8 -*-

import json
import logging

import numpy as np
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery

PROJECT_NAME = "cpb100demo1"
MODEL_NAME = "mnist"

# webapp
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(784)
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build("ml", "v1beta1", credentials=credentials)
    data = {"instances": [{"image": input.tolist(), "key": 0}]}
    req1 = ml.projects().predict(
        body=data, name="projects/{0}/models/{1}/versions/{2}".format(PROJECT_NAME, MODEL_NAME, "softmax_regression")
    )
    req2 = ml.projects().predict(
        body=data, name="projects/{0}/models/{1}".format(PROJECT_NAME, MODEL_NAME)
    )
    res1 = req1.execute()
    res2 = req2.execute()
    logging.info("response1: {}".format(json.dumps(res1)))
    output1 = res1["predictions"][0]["scores"]
    output2 = res2["predictions"][0]["scores"]
    return jsonify(results=[output1, output2])


@app.route('/', methods=["GET"])
def main():
    return render_template('index.html')
