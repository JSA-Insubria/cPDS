import datetime

import util as util


class Aggregator:
    def __init__(self, pk, c, S, L):
        self.pk = pk
        self.encrypted_x = None
        self.L = L
        self.S = S
        self.c = c

    def send_encrypted_x(self, x):
        self.encrypted_x = x

    def sum_lambaa_x(self, m, lambdaa):
        lambdaa_encrypted_k = self.pk.encryptMatrix(lambdaa)

        time_pre = datetime.datetime.now()

        # lambda = lambda + c * S2 * L2 * x;
        x_w = self.c * self.S @ self.L @ self.encrypted_x
        lambdaa_encrypted_k_plus_1 = lambdaa_encrypted_k + x_w

        time_post = datetime.datetime.now()
        util.writeIntoCSV(m, 'aggregator', str((time_post - time_pre).total_seconds()))

        return lambdaa_encrypted_k_plus_1
