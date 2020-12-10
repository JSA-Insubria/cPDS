import networkx as nx
import matplotlib.pyplot as plt
from statistics import mean
import numpy as np
import scipy.io as spio
import pywt

# the following is purely for the purpose of pretty printing matrices
from IPython.display import display
import sympy; sympy.init_printing()


def display_matrix(m):
    display(sympy.Matrix(m))


def local_degree(P, eps_deg):
    n = P.shape[0]
    W = np.zeros((n,n))
    deg = P.sum(axis=1)
    for i in range(0,n):
        for j in range(0, n):
            if P[i,j]==1:
                W[i,j]=1.0/(max(deg[i],deg[j])+eps_deg)
    W = np.diag(np.ones(n)-(W.sum(axis=1))) + W
    W = (W+np.transpose(W)+2*np.eye(n,n))/4;
    return W



# --------------------- Agent ---------------------

# build the network (with an Erdos-Renyi graph)
m = 4  # number of agents
print("Number of agents: %d" % (m))
p = 0.5  # probability of connecting two agents in the network

nc = 0
while (nc != 1):  # assert that there will be one connected component
    G = nx.erdos_renyi_graph(m, p)
    nc = nx.number_connected_components(G)

nx.draw(G, with_labels=True)
plt.show()

adj = nx.adjacency_matrix(G)
print("Adjacency matrix (in dense representation):")
display_matrix(adj.todense())

degrees = dict(nx.degree(G))
print("Node degrees:\n", degrees)
print("Average degree: ", mean(degrees.values()))





# --------------------- Doubly Stochastic Matrix ---------------------

W = local_degree(adj, 0.1)
L = np.eye(m) - W # doubly stochastic weight matrix of the graph




# --------------------- Load Data ---------------------
# load the data (for now synthetic mat datasets)
mat = spio.loadmat('../../test/data/dataset3.mat', squeeze_me=True)
xtrain = mat['xtrain']
ytrain = mat['ytrain']
training_label_distr = np.unique(ytrain, return_counts=True)
xtest = mat['xtest']
ytest = mat['ytest']
testing_label_distr = np.unique(ytest, return_counts=True)
print("Training data dimensions: %d x %d (%d:%d, %d:%d)" %(xtrain.shape[0],xtrain.shape[1], training_label_distr[0][0], training_label_distr[1][0], training_label_distr[0][1], training_label_distr[1][1]))
print("Test data dimensions: %d x %d (%d:%d, %d:%d)" %(xtest.shape[0],xtest.shape[1], testing_label_distr[0][0], testing_label_distr[1][0], testing_label_distr[0][1], testing_label_distr[1][1]))

# permute order of datapoints in training set to mix {+1}, {-1} examples
random_idx = np.random.rand(xtrain.shape[0]).argsort()
np.take(xtrain,random_idx,axis=0,out=xtrain)
np.take(ytrain,random_idx,axis=0,out=ytrain)


# assign training datapoints to agents (TODO: revisit if the data are not equally split)
n = xtrain.shape[0] # number of training datapoints
nppa = int(n/m) # number of (data)points per agent
print("Number of training datapoints: %d, number of datapoints per agent: %d" %(n,nppa))
n_j_start = np.arange(start=0, stop=n, step=nppa, dtype=np.int)
n_j_stop = np.arange(start=nppa-1, stop=n, step=nppa, dtype=np.int)

for j in range(m):
    label_distr = np.unique(ytrain[n_j_start[j]:n_j_stop[j]], return_counts=True)
    print("Agent %d is holding data %d to %d (%d:%d, %d:%d)" %(j, n_j_start[j], n_j_stop[j], label_distr[0][0], label_distr[1][0], label_distr[0][1], label_distr[1][1]))


gammas = np.zeros((m,), dtype = np.object)
number_of_points_per_agent = np.zeros((m), dtype=np.int)

for j in range(m):
    number_of_points_per_agent[j] = n_j_stop[j]-n_j_start[j]+1
    gammas[j] = np.ones((1, number_of_points_per_agent[j]))




# --------------------- Load Data Centralized Approach ---------------------

# Load optimal solution computed by centralized approach
mat = spio.loadmat('../../test/data/sSVM_gurobi.mat', squeeze_me=True)
x_opt = mat['theta_opt_SSVM']
# x_opt = np.array([0.7738, 0.7131, 0.0000, 0.0433, -0.0112, 0.0462])
w_SSVM = x_opt[:-1]
b_SSVM = x_opt[-1]





# --------------------- Parameters ---------------------

# define parameters // TODO: tune parameters (output max AUC and also keep track of time)
t = 5
tau = 10
rho = 10
c = 1
print("Parameters: t=%d, tau=%d, rho=%d, c=%d" % (t, tau, rho, c))

# define dimension variables
m = L.shape[0] # number of agents
d = xtrain.shape[1]
p = d+1

# inititalize algorithm parameters
L_p = np.eye(m) # size: m x m
Theta = t*np.eye(m) + np.diag(np.random.uniform(0,1,m)) # size: m x m
S = np.eye(m) # size: m x m





# --------------------- Variables ---------------------

# initialize algorithm variables
x = np.random.normal(0, 1, (m,p)) # miu=0, sigma=1, size: m x p
lamda_kminus1 = np.zeros((m,p)) # size: m x p
lamda = c*S@L@x # size: m x p
q_kminus1 = np.zeros((m,), dtype=np.object)
q = np.zeros((m,), dtype=np.object)
data = np.zeros((m,), dtype=np.object)
labels = np.zeros((m,), dtype=np.object)
y = np.zeros((m,), dtype=np.object)
n = np.zeros((m,), dtype=np.object)

for j in range(m):
    n[j] = n_j_stop[j]-n_j_start[j]+1
    y[j] = np.random.normal(0,1,(1,n[j]))
    q_kminus1[j] = np.zeros((1,n[j]))
    data[j] = xtrain[n_j_start[j]:n_j_stop[j]+1,:] # size n_j x d+1
    labels[j] = ytrain[n_j_start[j]:n_j_stop[j]+1] # size n_j x 1
    q[j] = c * gammas[j] * (
        np.einsum(
            'ij,ij->i',
            np.concatenate(
                [
                    np.diag(labels[j]) @ data[j],
                    labels[j].reshape(n[j],1)
                ],
                axis=1
            ),
            np.tile(x[j,:],(n[j],1))
        ) - y[j]
    )


max_iters = 100
residuals_x = np.zeros(max_iters, dtype = np.double)






# --------------------- Algorithm ---------------------

for iter in range(max_iters):

    # x-UPDATE
    for j in range(m):
        x_j = x[j, :]

        # find beta_1..d
        beta_j_k = x_j[:-1]
        lamda_1d_k = lamda[:, :-1]
        lamda_1d_kminus1 = lamda_kminus1[:, :-1]

        vec1 = np.sum(
            np.diag(
                gammas[j] * labels[j] * (
                    # (1 + (1 / c)) * q[j] - (1 / (c * q_kminus1[j]))
                        2 * q[j] - q_kminus1[j]
                )
            ) * data[j]
            , axis=0)  # 1 x d

        vec2 = L_p[j, :] @ S @ (
            #     (1+1/c)*lamda_1d_k-1/c*lamda_1d_kminus1
                2 * lamda_1d_k - lamda_1d_kminus1
        )
        # vec2 = 2 * lamda_1d_k - lamda_1d_kminus1

        vec3 = - Theta[j, j] * beta_j_k

        # u_j = -1/(tau/m+Theta[j,j])*(vec1+vec2+vec3)
        u_j = - (vec1 + vec2 + vec3) / (tau + Theta[j, j])

        # miu = (rho/m)/(tau/m+Theta[j,j]);
        miu = (2 * rho) / (tau + Theta[j, j])

        x_new_1d = pywt.threshold(u_j, miu, 'soft')

        # find beta_0
        beta_j0_k = x_j[-1]

        lamda_dplus1_k = lamda[:, -1]
        lamda_dplus1_kminus1 = lamda_kminus1[:, -1]

        term1 = np.sum(
            gammas[j] * labels[j] * (
                # (1+1/c)*q[j]-1/c*q_kminus1[j]
                    2 * q[j] - q_kminus1[j]
            ),
            axis=1
        )
        term2 = L_p[j, :] @ S @ (
            # (1+1/c)*lamda_dplus1_k-1/c*lamda_dplus1_kminus1
                2 * lamda_dplus1_k - lamda_dplus1_kminus1
        )
        # term2 =  2 * lamda_dplus1_k - lamda_dplus1_kminus1
        term3 = - Theta[j, j] * beta_j0_k

        x_new_dplus1 = -(term1 + term2 + term3) / Theta[j, j]

        # update x_j
        x[j, :] = np.concatenate([x_new_1d, x_new_dplus1])

    # y-UPDATE
    for j in range(m):
        term0 = np.transpose(np.ones((1, n[j])) / (gammas[j] ** 2))
        term1 = np.ones((n[j], 1))
        term2 = np.transpose(gammas[j] * q[j])
        term3 = np.einsum(
            'ij,ij->i',
            np.concatenate(
                [
                    np.diag(np.ravel(gammas[j] ** 2) * labels[j]) @ data[j],
                    np.transpose((gammas[j] ** 2) * labels[j])
                ], axis=1
            ), np.tile(x[j, :], (n[j], 1))
        ).reshape(n[j], 1)

        y_tilda_1 = np.transpose(np.ones((1, n[j])) / (gammas[j] ** 2)) * (term1 + term2 + term3)
        y_tilda_2 = np.transpose(np.ones((1, n[j])) / (gammas[j] ** 2)) * (term2 + term3)

        y_new = np.ones((n[j], 1))
        y_new[1 > y_tilda_1] = y_tilda_1[1 > y_tilda_1]
        y_new[1 < y_tilda_2] = y_tilda_2[1 < y_tilda_2]

        y[j] = y_new

    # q-UPDATE
    for j in range(m):
        q_kminus1[j] = q[j]
        # q[j] = q[j] + c*gammas[j]*np.transpose(np.transpose(np.einsum('ij,ij->i',np.concatenate([np.diag(labels[j])@data[j], np.transpose(labels[j]).reshape(n[j],1)],axis=1), np.tile(x[j,:],(n[j],1)))).reshape(n[j],1)-y[j])
        q[j] = q[j] + c * gammas[j] * np.transpose(
            np.transpose(
                np.einsum('ij,ij->i',
                          np.concatenate(
                              [
                                  np.diag(labels[j]) @ data[j],
                                  np.transpose(labels[j]).reshape(n[j], 1)
                              ],
                              axis=1),
                          np.tile(x[j, :], (n[j], 1))
                          )
            ).reshape(n[j], 1) - y[j]
        )

    # lambda-UPDATE
    lamda_kminus1 = lamda
    # lamda = lamda +c*S@L@x
    lamda = lamda + S @ L @ x

    # calculate residual
    residuals_x[iter] = np.linalg.norm(x - (np.ones((m, 1)) * x_opt))


print(lamda)


# --------------------- Plot ---------------------

# plot residuals
plt.plot(residuals_x)
plt.ylabel('Residuals x')
plt.xlabel('Iterations')
plt.show()
# plot residuals on the log scale
plt.semilogy(residuals_x)
plt.ylabel('Residuals x in logscale')
plt.xlabel('Iterations')
plt.show()

x_return = np.mean(x,0)
w_cPDS = x_return[:-1]
b_cPDS = x_return[-1]

x1 = np.arange(-2+np.min(np.concatenate((xtrain[:,0], xtest[:,0]), axis=0)), np.max(np.concatenate((xtrain[:,0], xtest[:,0]), axis=0))+2, 0.1)
x2_cPDS = (-w_cPDS[0]/w_cPDS[1])*x1-b_cPDS/w_cPDS[1]
x2_SSVM = (-w_SSVM[0]/w_SSVM[1])*x1-b_SSVM/w_SSVM[1]

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax.scatter(xtrain[ytrain==-1,0], xtrain[ytrain==-1,1], c='red', marker= 'o', label='training class -1')
ax.scatter(xtrain[ytrain==1,0], xtrain[ytrain==1,1], c='blue', marker = 'x', label='training class +1')
ax.scatter(xtest[ytest==-1,0], xtest[ytest==-1,1], c='green', marker= 's', label='test class -1')
ax.scatter(xtest[ytest==1,0], xtest[ytest==1,1], c='yellow', marker = 'v', label='test class +1')
ax.plot(x1, x2_SSVM, linewidth=2, markersize=12, label='SSVM')
ax.plot(x1, x2_cPDS, linewidth=2, markersize=12, label = 'cPDS')
ax.plot()
plt.legend(loc='lower left');
plt.grid()
plt.show()

# calculate AUC by SSVM
pred_vals_SSVM = (xtest@w_SSVM) + b_SSVM
thresholds = np.sort(pred_vals_SSVM)
miss = np.zeros(thresholds.size)
false_alarm = np.zeros(thresholds.size)
for i_thr in range(len(thresholds)):
    ypred = (pred_vals_SSVM <= thresholds[i_thr]) + 0
    ypred[ypred==0] = -1
    miss[i_thr] = np.sum(np.logical_and(ypred == -1, ytest == 1)) / np.sum(ytest == 1)
    false_alarm[i_thr] = np.sum(np.logical_and(ypred == 1, ytest == -1)) / np.sum(ytest == -1)

AUC_SSVM = np.abs(np.trapz(false_alarm, 1-miss))
print("AUC SSVM: ", AUC_SSVM)

# calculate AUC by cPDS
pred_vals_cPDS = xtest@w_cPDS+b_cPDS
thresholds = np.sort(pred_vals_cPDS)
miss = np.zeros(thresholds.size)
false_alarm = np.zeros(thresholds.size)
for i_thr in range(thresholds.size):
    ypred = (pred_vals_cPDS <= thresholds[i_thr])+0
    ypred[ypred==0] = -1
    miss[i_thr] = np.sum(np.logical_and(ypred == -1, ytest == 1))/np.sum(ytest == 1)
    false_alarm[i_thr] = np.sum(np.logical_and(ypred == 1, ytest == -1))/np.sum(ytest == -1)

AUC_cPDS = np.abs(np.trapz(false_alarm,1-miss))
print("AUC cPDS: ", AUC_cPDS)