import os
import json
import datetime
import numpy as np
import pandas as pd

import requests

import phe


def test_to_region(region, xtrain_len):
    mpk, msk, pk_list, sk_list = phe.generate_cPDS_keypair(10)
    x = np.random.normal(0, 1, (10, xtrain_len + 1))
    x_enc = pk_list[0].encryptMatrix(x[0])

    time_pre = datetime.datetime.utcnow()

    # Serialize
    x_enc = [x_enc[i].serialize() for i in range(len(x_enc))]

    if region == 'US':
        res = requests.post("https://cpds-test.herokuapp.com/compute_time/", json=json.dumps(x_enc))  # US
    else:
        res = requests.post("https://cpds-test-eu.herokuapp.com/compute_time/", json=json.dumps(x_enc))  # EU

    res = res.json()
    time_post = datetime.datetime.fromisoformat(res['time_post'])

    time = (time_post - time_pre).total_seconds()
    return time


def test_US_to_EU(xtrain_len):
    res = requests.post("https://cpds-test.herokuapp.com/send_msg/", json=json.dumps({'xtrain_len': xtrain_len}))  # US
    res = res.json()
    return res['time']


def test_to_region_notenc(region, xtrain_len):
    x = np.random.normal(0, 1, (10, xtrain_len + 1))

    time_pre = datetime.datetime.utcnow()

    # Serialize
    x_enc = x[0].tolist()

    if region == 'US':
        res = requests.post("https://cpds-test.herokuapp.com/compute_time_notenc/", json=json.dumps(x_enc))  # US
    else:
        res = requests.post("https://cpds-test-eu.herokuapp.com/compute_time_notenc/", json=json.dumps(x_enc))  # EU

    res = res.json()
    time_post = datetime.datetime.fromisoformat(res['time_post'])

    time = (time_post - time_pre).total_seconds()
    return time


def test_US_to_EU_notenc(xtrain_len):
    res = requests.post("https://cpds-test.herokuapp.com/send_msg_notenc/", json=json.dumps({'xtrain_len': xtrain_len}))  # US
    res = res.json()
    return res['time']


# compute auc mean
def save_mean_communication_time(time):
    time.to_csv('logs' + os.sep + "communication_time.csv", mode='a')


if __name__ == "__main__":
    xtrain_len = 5
    times = 2

    # 1 - EU -> US
    # 2 - IT -> US
    # 3 - IT -> EU

    time_mean_1 = []
    time_mean_2 = []
    time_mean_3 = []
    time_mean_1_notenc = []
    time_mean_2_notenc = []
    time_mean_3_notenc = []

    for i in range(times):
        time_mean_1.append(test_US_to_EU(xtrain_len))
        time_mean_2.append(test_to_region('US', xtrain_len))
        time_mean_3.append(test_to_region('EU', xtrain_len))
        time_mean_1_notenc.append(test_US_to_EU_notenc(xtrain_len))
        time_mean_2_notenc.append(test_to_region_notenc('US', xtrain_len))
        time_mean_3_notenc.append(test_to_region('EU', xtrain_len))

    array = np.array([time_mean_1, time_mean_2, time_mean_3,
                      time_mean_1_notenc, time_mean_2_notenc, time_mean_3_notenc])

    time = pd.DataFrame(array.T, columns=['EU->US', 'IT->US', 'IT->EU', 'EU->US_notenc', 'IT->US_notenc', 'IT->EU_notenc'])
    save_mean_communication_time(time)

