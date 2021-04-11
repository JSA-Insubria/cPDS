import os
import sys
import json
import datetime
import numpy as np
import pandas as pd

import requests

import phe


def test_to_region(region, x_enc):
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


def test_to_region_notenc(region, x):
    time_pre = datetime.datetime.utcnow()

    # Serialize
    x_enc = x.tolist()

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


def get_json_size(x_enc, xt_len, type):
    file = 'logs' + os.sep + 'test.json'
    data = json.dumps(x_enc)
    with open(file, 'w') as outfile:
        outfile.write(data)

    size = str(os.path.getsize(file))

    with open('logs' + os.sep + 'json_size.csv', 'a') as outfile:
        outfile.write('' + str(xt_len) + ',' + type + ',' + size + '\n')

    return size


def get_alive():
    mpk, msk, pk_list, sk_list = phe.generate_cPDS_keypair(10)
    x = np.random.normal(0, 1, (10, 5 + 1))
    x_enc = pk_list[0].encryptMatrix(x[0])

    print(test_US_to_EU(5))
    print(test_to_region('US', x_enc))
    print(test_to_region('EU', x_enc))
    print('---')
    print(test_US_to_EU_notenc(5))
    print(test_to_region_notenc('US', x[0]))
    print(test_to_region_notenc('EU', x[0]))

    return '--- UP ---'


# compute auc mean
def save_mean_communication_time(time, xtrain_len):
    time.to_csv('logs' + os.sep + "communication_time" + "_" + str(xtrain_len) + ".csv", mode='a')


if __name__ == "__main__":
    print(get_alive())
    xtrain_len = [5, 10, 15, 20]
    times = 10

    for xt_len in xtrain_len:
        # 1 - EU -> US
        # 2 - IT -> US
        # 3 - IT -> EU

        mpk, msk, pk_list, sk_list = phe.generate_cPDS_keypair(10)
        x = np.random.normal(0, 1, (10, xt_len + 1))
        x_enc = pk_list[0].encryptMatrix(x[0])

        x_ser = [x_enc[i].serialize() for i in range(len(x_enc))]
        get_json_size(x_ser, xt_len, 'enc')
        get_json_size(x[0].tolist(), xt_len, 'not_enc')

        time_mean_1 = []
        time_mean_2 = []
        time_mean_3 = []
        time_mean_1_notenc = []
        time_mean_2_notenc = []
        time_mean_3_notenc = []

        for j in range(times):
            time_mean_1.append(test_US_to_EU(xt_len))
            time_mean_2.append(test_to_region('US', x_enc))
            time_mean_3.append(test_to_region('EU', x_enc))
            time_mean_1_notenc.append(test_US_to_EU_notenc(xt_len))
            time_mean_2_notenc.append(test_to_region_notenc('US', x[0]))
            time_mean_3_notenc.append(test_to_region_notenc('EU', x[0]))

        array = np.array([time_mean_1, time_mean_2, time_mean_3,
                          time_mean_1_notenc, time_mean_2_notenc, time_mean_3_notenc])

        time = pd.DataFrame(array.T, columns=['EU->US', 'IT->US', 'IT->EU', 'EU->US_notenc', 'IT->US_notenc', 'IT->EU_notenc'])
        save_mean_communication_time(time, xt_len)

