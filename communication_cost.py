import json
import datetime
import numpy as np

import requests

import phe
import load_save_data


def send_message():
    m = 10
    xtrain, ytrain, xtest, ytest = load_save_data.loadData()

    mpk, msk, pk_list, sk_list = phe.generate_cPDS_keypair(m)
    x = np.random.normal(0, 1, (m, xtrain.shape[1] + 1))
    x_enc = mpk.encryptMatrix(x)

    time_pre = datetime.datetime.utcnow()

    # Serialize
    x_enc = [[x_enc[i][j].serialize() for i in range(np.shape(x_enc)[0])] for j in range(np.shape(x_enc)[1])]

    res = requests.post("https://cpds-test.herokuapp.com/compute_time/", json=json.dumps(x_enc))

    res = res.json()
    time_post = datetime.datetime.fromisoformat(res['time_post'])

    time = (time_post - time_pre).total_seconds()
    print(str(time))
    return time


if __name__ == "__main__":
    times = 10
    sum = 0
    for i in range(times):
        sum += send_message()

    mean = sum / times
    print('mean: ', mean)
