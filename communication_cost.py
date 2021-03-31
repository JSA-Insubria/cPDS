import os
import json
import datetime
import numpy as np

import requests

import phe
import load_save_data


def send_message(xtrain):
    m = 10

    mpk, msk, pk_list, sk_list = phe.generate_cPDS_keypair(m)
    x = np.random.normal(0, 1, (m, xtrain.shape[1] + 1))
    x_enc = mpk.encryptMatrix(x[0])

    time_pre = datetime.datetime.utcnow()

    # Serialize
    x_enc = [x_enc[i].serialize() for i in range(len(x_enc))]

    #res = requests.post("https://cpds-test-eu.herokuapp.com/compute_time/", json=json.dumps(x_enc))  # EU
    res = requests.post("https://cpds-test.herokuapp.com/compute_time/", json=json.dumps(x_enc))  # US

    res = res.json()
    time_post = datetime.datetime.fromisoformat(res['time_post'])

    time = (time_post - time_pre).total_seconds()
    return time


# compute auc mean
def save_mean_communication_time(time):
    with open('logs' + os.sep + "communication_time_mean.csv", 'a') as fd:
        fd.write(str(time) + '\n')


if __name__ == "__main__":
    xtrain, _, _, _ = load_save_data.loadData()

    times = 10
    time_sum = 0
    for i in range(times):
        time = send_message(xtrain)
        save_mean_communication_time(time)
        time_sum += time
        print(time)

    print('Mean: ', time_sum/times)

