import numpy as np
from phe import paillier


def get_number_of_neighbors(L):
    return [np.count_nonzero(L[i]) for i in range(len(L[0]))]


def gen_keys(L):
    keys_dict = {}
    num_of_neigh = get_number_of_neighbors(L)
    alone_list = get_alone_listL(L)
    for i in range(len(num_of_neigh)):
        if num_of_neigh[i] == 2:
            mpk, msk, pk_list, sk_list = paillier.generate_cPDS_keypair(num_of_neigh[i]+1)
        else:
            mpk, msk, pk_list, sk_list = paillier.generate_cPDS_keypair(num_of_neigh[i])

        keys_dict['mpk' + str(i)] = mpk
        keys_dict['msk' + str(i)] = msk
        keys_dict['pk_list' + str(i)] = pk_list
        keys_dict['sk_list' + str(i)] = sk_list

    return keys_dict


def get_alone_listL(L):
    m = len(L)
    alone_list = np.full(m, -1)
    for alone in range(m):
        if np.count_nonzero(L[alone]) == 2:
            friend_index = get_friend_index(L[alone], alone)
            first_friend_index = get_friend_friend_index(L[friend_index], friend_index, alone)
            alone_list[alone] = first_friend_index
    return alone_list


def get_friend_index(L, index_alone):
    for i in range(len(L)):
        if L[i] != 0:
            if i != index_alone:
                return i


def get_friend_friend_index(L, index_friend, index_alone):
    for i in range(len(L)):
        if L[i] != 0:
            if (i != index_friend) & (i != index_alone):
                return i