import numpy as np
import pywt
import datetime

import util as util


class cPDS:
    def __init__(self, agent_id, pk, S, L_p, c, theta, gammas, data, labels, q, n, x):
        self.tau = 1
        self.rho = 1
        self.S = S
        self.L_p = L_p  # L_p[j]
        self.c = c
        self.theta = theta  # theta[j] -> 1xm
        self.gammas = gammas  # gammas[j] -> 1x250
        self.data = data  # data[j] -> 250x5
        self.labels = labels  # labels[j] -> 250x1
        self.q = q  # q[j] -> 1x250
        self.q_kminus1 = np.zeros((1, n))  # 1x250 init:0
        self.x = x  # x[j] -> 1xp (p=n°features+1)
        self.lamda_kminus1 = np.zeros((len(S[0]), 6))  # # 1x6 init:0
        self.n = n
        self.pk = pk
        self.agent_id = agent_id

    def compute(self, m, lambdaa):
        # x-Update
        beta_k_j = self.x[:-1]
        beta_k_j0 = self.x[-1]
        lambda_d_k = lambdaa[:, -1]
        lambda_d_kminus1 = self.lamda_kminus1[:, -1]
        lamda_dplus1_k = lambdaa[:, -1]
        lamda_dplus1_kminus1 = self.lamda_kminus1[:, -1]

        # Beta_k+1_jt
        mu = (2 * self.rho) / (self.tau + self.theta)
        # vec1 = sum(diag(gammas{j}.*labels{j}.*((1+1/c)*q{j}-1/c*q_kminus1{j}))*data{j});
        v1 = np.sum(np.diag(self.gammas @ self.labels @ ((1+1/self.c) * self.q - 1 / self.c * self.q_kminus1)) @ self.data)
        # vec2 = sum(S*diag(L(j,:))*((1+1/c)*lambda_1d_k-1/c*lambda_1d_kminus1));
        v2 = np.sum(self.S * np.diag(self.L_p) * ((1+1/self.c) * lambda_d_k - 1 / self.c * lambda_d_kminus1))
        v3 = -1 * (self.theta * beta_k_j)
        u = -1 / (self.tau + self.theta) * (v1 + v2 + v3)
        beta_kplus1_jt = pywt.threshold(u, mu, 'soft')

        # Beta_k+1_j0
        # term1 = sum(gammas{j}.*labels{j}.*((1+1/c)*q{j}+1/c*q_kminus1{j}));
        v1_0 = np.sum(self.gammas @ self.labels @ ((1+1/self.c) * self.q + 1 / self.c * self.q_kminus1))
        # term2 = sum(S*diag(L(j,:))*((1+1/c)*lambda_dplus1_k-1/c*lambda_dplus1_kminus1));
        v2_0 = np.sum(self.S * np.diag(self.L_p) * ((1+1/self.c) * lamda_dplus1_k - 1 / self.c * lamda_dplus1_kminus1))
        v3_0 = -1 * (self.theta * beta_k_j0)
        beta_kplus1_j0 = (v1_0 + v2_0 + v3_0) / self.theta

        self.x = np.append([beta_kplus1_jt], beta_kplus1_j0)

        # y-Update
        # term1 = -1*ones(n{j},1);
        t1 = np.ones((self.n, 1))
        # term2 = -(gammas{j}.*q{j})';
        t2 = np.transpose(self.gammas * self.q)
        # term3 = - dot([diag(gammas{j}.^2 .*labels{j})*data{j}, (gammas{j}.^2 .*labels{j})' ], repmat(x(j,:),n{j},1),2);
        t3 = np.einsum('ij,ij->i', np.concatenate(
            [
                np.diag(np.ravel(self.gammas ** 2) * self.labels) @ self.data,
                np.transpose((self.gammas ** 2) * self.labels)
            ], axis=1), np.tile(self.x, (self.n, 1))).reshape(self.n, 1)

        y_tilda_1 = np.transpose(np.ones((1, self.n)) / (self.gammas ** 2)) * (t1 + t2 + t3)
        y_tilda_2 = np.transpose(np.ones((1, self.n)) / (self.gammas ** 2)) * (t2 + t3)

        y_kplus1_j = np.ones((self.n, 1))
        y_kplus1_j[1 > y_tilda_1] = y_tilda_1[1 > y_tilda_1]
        y_kplus1_j[1 < y_tilda_2] = y_tilda_2[1 < y_tilda_2]

        # q-Update
        # q{j} = q{j}+ c*gammas{j}.*(dot([diag(labels{j})*data{j}, labels{j}'],repmat(x(j,:),n{j},1),2)'-y{j});
        q_kplus1_j = self.q + self.c * self.gammas * np.transpose(
            np.transpose(np.einsum('ij,ij->i', np.concatenate(
                [
                    np.diag(self.labels) @ self.data,
                    np.transpose(self.labels).reshape(self.n, 1)
                ], axis=1), np.tile(self.x, (self.n, 1)))).reshape(self.n, 1) - y_kplus1_j)

        self.q = q_kplus1_j
        self.lamda_kminus1 = lambdaa

        #time_pre = datetime.datetime.now()
        #x_enc = self.pk.encryptMatrix(self.x)
        #time_post = datetime.datetime.now()
        #util.writeIntoCSV(m, 'agent_' + str(self.agent_id), str((time_post - time_pre).total_seconds()))
        #return x_enc

        return self.x
