# -*- coding: utf-8 -*-

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
    req = ml.projects().predict(
        body=data, name="projects/{0}/models/{1}".format(PROJECT_NAME, MODEL_NAME)
    )
    res = req.execute()
    output1 = res["predictions"][0]["scores"]
    output2 = [0] * 10
    return jsonify(results=[output1, output2])


@app.route('/', methods=["GET"])
def main():
    return render_template('index.html')
