import numpy as np
from phe import paillier


def get_number_of_neighbors(L):
    return [np.count_nonzero(L[i]) for i in range(len(L[0]))]


def gen_keys(L):
    keys_dict = {}
    num_of_neigh = get_number_of_neighbors(L)
    for i in range(len(num_of_neigh)):
        mpk, msk, pk_list, sk_list = paillier.generate_cPDS_keypair(num_of_neigh[i])
        keys_dict['mpk' + str(i)] = mpk
        keys_dict['msk' + str(i)] = msk
        keys_dict['pk_list' + str(i)] = pk_list
        keys_dict['sk_list' + str(i)] = sk_list

    return keys_dict
