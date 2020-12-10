import numpy as np


class Aggregator:
    def __init__(self, pk, L):
        self.pk = pk
        self.encrypted_x = None
        self.L = L

    def send_encrypted_x(self, x):
        self.encrypted_x = x

    def sum_lambaa_x(self, lambdaa):
        lambdaa_encrypted_k = self.pk.encryptMatrix(lambdaa)
        x_w = self.encrypted_x - (self.L @ self.encrypted_x)
        lambdaa_encrypted_k_plus_1 = lambdaa_encrypted_k + x_w
        return lambdaa_encrypted_k_plus_1
