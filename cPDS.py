import numpy as np
import pywt


class cPDS:
    def __init__(self, agent_id, tau, rho, theta, gammas, data, labels, q, n, x, L):
        m = len(L)
        self.tau = tau/m
        self.rho = rho/m
        self.theta = theta  # theta[j][j] -> 1x1
        self.gammas = gammas  # gammas[j] -> 1x250
        self.data = data  # data[j] -> 250x5
        self.labels = labels  # labels[j] -> 250x1
        self.q = q  # q[j] -> 1x250
        self.q_kminus1 = np.zeros((1, n))  # 1x250 init:0
        self.x = x  # x[j] -> 1xp (p=n°features+1)
        self.lamda_kminus1 = np.zeros(len(x))  # 1x6 init:0
        self.n = n
        self.agent_id = agent_id
        self.L = L
        self.check = True

    def compute(self, lambdaa):  # lambdaa[j]
        # x-Update
        beta_k_j = self.x[:-1]
        beta_k_j0 = self.x[-1]
        lambda_d_k = lambdaa[:-1]
        lambda_d_kminus1 = self.lamda_kminus1[:-1]
        lambda_dplus1_k = lambdaa[-1]
        lamdba_dplus1_kminus1 = self.lamda_kminus1[-1]

        # Beta_k+1_jt
        v1 = sum(np.diag(np.squeeze(self.gammas * self.labels * np.subtract(2 * self.q, self.q_kminus1))) @ self.data)
        v2 = np.subtract(2 * lambda_d_k, lambda_d_kminus1)
        v3 = - self.theta * beta_k_j
        u = -1 / (self.tau + self.theta) * (v1 + v2 + v3)
        mu = (self.rho) / (self.tau + self.theta)
        beta_kplus1_jt = pywt.threshold(u, mu, 'soft')

        # Beta_k+1_j0
        v1_0 = np.sum(self.gammas * self.labels * np.subtract(2 * self.q, self.q_kminus1))
        v2_0 = np.subtract(2 * lambda_dplus1_k, lamdba_dplus1_kminus1)
        v3_0 = - self.theta * beta_k_j0
        beta_kplus1_j0 = - (v1_0 + v2_0 + v3_0) / self.theta

        self.x = np.append([beta_kplus1_jt], beta_kplus1_j0)

        # y-Update
        t1 = np.ones((self.n, 1))
        t2 = np.transpose(self.gammas * self.q)

        t3 = np.einsum('ij,ij->i', np.concatenate((np.diag(np.squeeze((self.gammas ** 2) * self.labels)) @ self.data,
                                                     np.transpose((self.gammas ** 2) * self.labels)), axis=1),
                         np.tile(self.x, (self.n, 1))).reshape(-1, 1)

        y_tilda_1 = np.transpose(1 / (self.gammas ** 2)) * (t1 + t2 + t3)
        y_tilda_2 = np.transpose(1 / (self.gammas ** 2)) * (t2 + t3)

        y_kplus1_j = np.ones((self.n, 1))
        y_kplus1_j[1 > y_tilda_1] = y_tilda_1[1 > y_tilda_1]
        y_kplus1_j[1 < y_tilda_2] = y_tilda_2[1 < y_tilda_2]
        y = np.transpose(y_kplus1_j)

        # q-Update
        self.q_kminus1 = self.q
        self.q = self.q + self.gammas * (np.transpose(np.einsum('ij,ij->i', np.concatenate(
            (np.diag(np.squeeze(self.labels)) @ self.data,
             self.labels.reshape(-1, 1)), axis=1), np.tile(self.x, (self.n, 1)))) - y)

        self.lamda_kminus1 = lambdaa

        return self.x
