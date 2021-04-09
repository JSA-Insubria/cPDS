import json
import datetime
import numpy as np

import phe

from flask import Flask
from flask import request

import requests

app = Flask(__name__)


@app.route('/compute_time/', methods=['POST'])
def compute_time():
    json_data = request.get_json()
    x_enc = json.loads(json_data)

    x = np.asarray([phe.EncryptedNumber.deserialize(x_enc[i]) for i in range(len(x_enc))])

    time_post = datetime.datetime.utcnow().isoformat()
    result = {'time_post': time_post}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/send_msg/', methods=['POST'])
def send_msg():
    json_data = request.get_json()
    json_xtrain = json.loads(json_data)
    xtrain_len = json_xtrain['xtrain_len']

    mpk, msk, pk_list, sk_list = phe.generate_cPDS_keypair(10)
    x = np.random.normal(0, 1, (10, xtrain_len + 1))
    x_enc = pk_list[0].encryptMatrix(x[0])

    # Serialize and send to EU server
    time_pre = datetime.datetime.utcnow()

    # Serialize
    x_enc = [x_enc[i].serialize() for i in range(len(x_enc))]

    res = requests.post("https://cpds-test-eu.herokuapp.com/compute_time/", json=json.dumps(x_enc))  # EU
    # res = requests.post("https://cpds-test.herokuapp.com/compute_time/", json=json.dumps(x_enc))  # US

    res = res.json()
    time_post = datetime.datetime.fromisoformat(res['time_post'])

    time = (time_post - time_pre).total_seconds()

    result = {'time': time}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/compute_time_notenc/', methods=['POST'])
def compute_time_notenc():
    json_data = request.get_json()
    x_enc = json.loads(json_data)

    x = np.asarray(x_enc)

    time_post = datetime.datetime.utcnow().isoformat()
    result = {'time_post': time_post}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/send_msg_notenc/', methods=['POST'])
def send_msg_notenc():
    json_data = request.get_json()
    json_xtrain = json.loads(json_data)
    xtrain_len = json_xtrain['xtrain_len']

    x = np.random.normal(0, 1, (10, xtrain_len + 1))
    # Serialize and send to EU server
    time_pre = datetime.datetime.utcnow()

    # Serialize
    x_enc = x[0].tolist()

    res = requests.post("https://cpds-test-eu.herokuapp.com/compute_time_notenc/", json=json.dumps(x_enc))  # EU
    # res = requests.post("https://cpds-test.herokuapp.com/compute_time_notenc/", json=json.dumps(x_enc))  # US

    res = res.json()
    time_post = datetime.datetime.fromisoformat(res['time_post'])

    time = (time_post - time_pre).total_seconds()

    result = {'time': time}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == '__main__':
    app.run(debug=True)
