#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 20:14:33 2021

@author: lihongyi
"""
from data_schema import get_train_test_ml_set, plot_cluster, get_scaler
import numpy as np
import kmeans
import kgauss
import variable_trans
import neural_net

seed = 123


def abalone_add_gender(independent_matrix, reduced_matrix):
    n, _ = independent_matrix.shape
    _, m = reduced_matrix.shape
    m += 2
    new_matrix = np.zeros(shape=(n, m))
    new_matrix[:, :2] += independent_matrix[:, :2]
    new_matrix[:, 2:] += reduced_matrix
    return new_matrix


def add_claster(independent_matrix, labels, k):
    n, m = independent_matrix.shape
    m1 = m + k-1
    new_matrix = np.zeros(shape=(n, m1))
    new_matrix[:, :m] += independent_matrix
    for i in range(k-1):
        new_v = labels * 0
        new_v[np.where(labels == i+1)] = 1.0
        new_matrix[:, m+i] += new_v
    return new_matrix


def clusters_wrapper(ca_independent_matrix, train_set, suffix, x, y, ca_test_matrix=None, test_set=None):
    data_set_name = train_set.schema.schema_name
    scores, clusters = kmeans.get_silhouette_by_k(ca_independent_matrix, name=data_set_name+suffix)
    cluster_sil = clusters[np.argmax(scores)]
    k_sil = cluster_sil.labels_.max()+1
    clusters_sil = np.random.RandomState(seed=seed).permutation(k_sil)[:min(10, k_sil)]
    plot_cluster(train_set, cluster_sil.labels_, x=x, y=y, clusters=clusters_sil, name=f'{data_set_name}{suffix}_kmeans_silhouette')
    variable_trans.avg_entropy(train_set.dependent_vector, cluster_sil.labels_, name=f'{data_set_name}{suffix}_kmeans_silhouette')
    
    if ca_test_matrix is not None:
        nn = neural_net.find_nn(add_claster(ca_independent_matrix, cluster_sil.labels_, k_sil), train_set.dependent_vector, f'{data_set_name}{suffix}_kmeans_silhouette')
        neural_net.error_rate(add_claster(ca_test_matrix, cluster_sil.predict(ca_test_matrix), k_sil), test_set.dependent_vector, nn, name=f'{data_set_name}{suffix}_kmeans_silhouette_test')
    
    gs, ss, clusters = kmeans.get_gap_by_k(ca_independent_matrix, name=data_set_name+suffix)
    good_ks = np.where(np.array(gs[:-1]) >= np.array(gs[1:])-np.array(ss[1:]))[0]
    if len(good_ks) > 0:
        cluster_gap = clusters[good_ks.min()]
        k_gap = cluster_gap.labels_.max()+1
        if k_gap != k_sil:
            clusters_gap = np.random.RandomState(seed=seed).permutation(k_gap)[:min(10, k_gap)]
            plot_cluster(train_set, cluster_gap.labels_, x=x, y=y, clusters=clusters_gap, name=f'{data_set_name}{suffix}_kmeans_gap')
            variable_trans.avg_entropy(train_set.dependent_vector, cluster_gap.labels_, name=f'{data_set_name}{suffix}_kmeans_gap')
            if ca_test_matrix is not None:
                nn = neural_net.find_nn(add_claster(ca_independent_matrix, cluster_gap.labels_, k_gap), train_set.dependent_vector, f'{data_set_name}{suffix}_kmeans_gap')
                neural_net.error_rate(add_claster(ca_test_matrix, cluster_gap.predict(ca_test_matrix), k_gap), test_set.dependent_vector, nn, name=f'{data_set_name}{suffix}_kmeans_gap_test')
        else:
            with open(f'{data_set_name}{suffix}_kmeans_gap', 'w') as f:
                f.write('same k as sil')
    
    scores, clusters = kgauss.get_silhouette_by_k(ca_independent_matrix, name=data_set_name+suffix)
    cluster_sil = clusters[np.argmax(scores)]
    k_sil = cluster_sil.predict(ca_independent_matrix).max()+1
    clusters_sil = np.random.RandomState(seed=seed).permutation(k_sil)[:min(10, k_sil)]
    plot_cluster(train_set, cluster_sil.predict(ca_independent_matrix), x=x, y=y, clusters=clusters_sil, name=f'{data_set_name}{suffix}_kgauss_silhouette')
    variable_trans.avg_entropy(train_set.dependent_vector, cluster_sil.predict(ca_independent_matrix), name=f'{data_set_name}{suffix}_kgauss_silhouette')
    
    if ca_test_matrix is not None:
        nn = neural_net.find_nn(add_claster(ca_independent_matrix, cluster_sil.predict(ca_independent_matrix), k_sil), train_set.dependent_vector, f'{data_set_name}{suffix}_kgauss_silhouette')
        neural_net.error_rate(add_claster(ca_test_matrix, cluster_sil.predict(ca_test_matrix), k_sil), test_set.dependent_vector, nn, name=f'{data_set_name}{suffix}_kgauss_silhouette_test')
    
    gs, ss, clusters = kgauss.get_gap_by_k(ca_independent_matrix, name=data_set_name+suffix)
    good_ks = np.where(np.array(gs[:-1]) >= np.array(gs[1:])-np.array(ss[1:]))[0]
    if len(good_ks) > 0:
        cluster_gap = clusters[good_ks.min()]
        k_gap = cluster_gap.predict(ca_independent_matrix).max()+1
        if k_gap != k_sil:
            clusters_gap = np.random.RandomState(seed=seed).permutation(k_gap)[:min(10, k_gap)]
            plot_cluster(train_set, cluster_gap.predict(ca_independent_matrix), x=x, y=y, clusters=clusters_gap, name=f'{data_set_name}{suffix}_kgauss_gap')
            variable_trans.avg_entropy(train_set.dependent_vector, cluster_gap.predict(ca_independent_matrix), name=f'{data_set_name}{suffix}_kgauss_gap')
            if ca_test_matrix is not None:
                nn = neural_net.find_nn(add_claster(ca_independent_matrix, cluster_gap.predict(ca_independent_matrix), k_gap), train_set.dependent_vector, f'{data_set_name}{suffix}_kgauss_gap')
                neural_net.error_rate(add_claster(ca_test_matrix, cluster_gap.predict(ca_test_matrix), k_gap), test_set.dependent_vector, nn, name=f'{data_set_name}{suffix}_kgauss_gap_test')
        else:
            with open(f'{data_set_name}{suffix}_kgauss_gap', 'w') as f:
                f.write('same k as sil')


def run_abalone():
    data_set_name = 'abalone'
    train_set, test_set = get_train_test_ml_set(data_set_name)
    
    scaler = get_scaler(train_set.independent_matrix)
    normalized_independent_matrix = scaler.transform(train_set.independent_matrix)
    
    clusters_wrapper(normalized_independent_matrix, train_set, '_vanila', 3, 8)
    plot_cluster(train_set, train_set.dependent_vector-1, x=3, y=8, clusters=[0, 1, 2], name=data_set_name)


def run_iris():
    data_set_name = 'iris'
    train_set, test_set = get_train_test_ml_set(data_set_name)
    
    scaler = get_scaler(train_set.independent_matrix)
    normalized_independent_matrix = scaler.transform(train_set.independent_matrix)
    
    clusters_wrapper(normalized_independent_matrix, train_set, '_vanila', 0, 1)
    plot_cluster(train_set, train_set.dependent_vector-1, x=0, y=1, clusters=[0, 1, 2], name=data_set_name)


def run_pca_abalone():
    data_set_name = 'abalone'
    suffix = '_pca'
    train_set, test_set = get_train_test_ml_set(data_set_name)
    
    scaler = get_scaler(train_set.independent_matrix)
    normalized_independent_matrix = scaler.transform(train_set.independent_matrix)
    normalized_test_matrix = scaler.transform(test_set.independent_matrix)
    
    pca = variable_trans.perform_pca(normalized_independent_matrix[:, 2:], name=data_set_name)
    nb_pc = 2
    pca_independent_matrix = abalone_add_gender(normalized_independent_matrix, pca.transform(normalized_independent_matrix[:, 2:])[:, :nb_pc])
    pca_test_matrix = abalone_add_gender(normalized_test_matrix, pca.transform(normalized_test_matrix[:, 2:])[:, :nb_pc])
    
    nn = neural_net.find_nn(pca_independent_matrix, train_set.dependent_vector, 'abalone_pca')
    neural_net.error_rate(pca_test_matrix, test_set.dependent_vector, nn, name='abalone_pca_test')
    
    clusters_wrapper(pca_independent_matrix, train_set, suffix, 3, 8, pca_test_matrix, test_set)
    
    
def run_pca_iris():
    data_set_name = 'iris'
    suffix = '_pca'
    train_set, test_set = get_train_test_ml_set(data_set_name)
    
    scaler = get_scaler(train_set.independent_matrix)
    normalized_independent_matrix = scaler.transform(train_set.independent_matrix)
    
    pca = variable_trans.perform_pca(normalized_independent_matrix, name=data_set_name)
    nb_pc = 3
    pca_independent_matrix = pca.transform(normalized_independent_matrix)[:, :nb_pc]
    
    clusters_wrapper(pca_independent_matrix, train_set, suffix, 0, 1)


def run_ica_abalone():
    data_set_name = 'abalone'
    suffix = '_ica'
    train_set, test_set = get_train_test_ml_set(data_set_name)
    
    scaler = get_scaler(train_set.independent_matrix)
    normalized_independent_matrix = scaler.transform(train_set.independent_matrix)
    normalized_test_matrix = scaler.transform(test_set.independent_matrix)
    
    ica = variable_trans.perform_ica(normalized_independent_matrix[:, 2:], name=data_set_name)
    ics = [2, 3, 5, ]
    ica_independent_matrix = abalone_add_gender(normalized_independent_matrix, ica.transform(normalized_independent_matrix[:, 2:])[:, ics])
    ica_test_matrix = abalone_add_gender(normalized_test_matrix, ica.transform(normalized_test_matrix[:, 2:])[:, ics])
    
    nn = neural_net.find_nn(ica_independent_matrix, train_set.dependent_vector, 'abalone_ica')
    neural_net.error_rate(ica_test_matrix, test_set.dependent_vector, nn, name='abalone_ica_test')
    
    clusters_wrapper(ica_independent_matrix, train_set, suffix, 3, 8, ica_test_matrix, test_set)


def run_ica_iris():
    data_set_name = 'iris'
    suffix = '_ica'
    train_set, test_set = get_train_test_ml_set(data_set_name)
    
    scaler = get_scaler(train_set.independent_matrix)
    normalized_independent_matrix = scaler.transform(train_set.independent_matrix)
    
    ica = variable_trans.perform_ica(normalized_independent_matrix, name=data_set_name)
    ics = [0, 2, 3, ]
    ica_independent_matrix = variable_trans.ic_wt2(ica.transform(normalized_independent_matrix)[:, ics], 2)
    
    clusters_wrapper(ica_independent_matrix, train_set, suffix, 0, 1)


def run_kca_abalone():
    data_set_name = 'abalone'
    suffix = '_kca'
    train_set, test_set = get_train_test_ml_set(data_set_name)
    
    scaler = get_scaler(train_set.independent_matrix)
    normalized_independent_matrix = scaler.transform(train_set.independent_matrix)
    normalized_test_matrix = scaler.transform(test_set.independent_matrix)
    
    n_components = 20
    kca = variable_trans.perform_kernel_pca(normalized_independent_matrix[:, 2:], n_components=n_components, name=data_set_name)
    nb_kc = 16
    kca_independent_matrix = abalone_add_gender(normalized_independent_matrix, kca.transform(normalized_independent_matrix[:, 2:])[:, :nb_kc])
    kca_test_matrix = abalone_add_gender(normalized_test_matrix, kca.transform(normalized_test_matrix[:, 2:])[:, :nb_kc])
    
    nn = neural_net.find_nn(kca_independent_matrix, train_set.dependent_vector, 'abalone_kca')
    neural_net.error_rate(kca_test_matrix, test_set.dependent_vector, nn, name='abalone_kca_test')
    
    clusters_wrapper(kca_independent_matrix, train_set, suffix, 3, 8, kca_test_matrix, test_set)


def run_kca_iris():
    data_set_name = 'iris'
    suffix = '_kca'
    train_set, test_set = get_train_test_ml_set(data_set_name)
    
    scaler = get_scaler(train_set.independent_matrix)
    normalized_independent_matrix = scaler.transform(train_set.independent_matrix)
    
    n_components = 10
    kca = variable_trans.perform_kernel_pca(normalized_independent_matrix, n_components=n_components, name=data_set_name)
    nb_kc = 7
    kca_independent_matrix = kca.transform(normalized_independent_matrix)[:, :nb_kc]
    
    clusters_wrapper(kca_independent_matrix, train_set, suffix, 0, 1)


def run_rca_abalone():
    data_set_name = 'abalone'
    suffix = '_rca'
    train_set, test_set = get_train_test_ml_set(data_set_name)
    
    scaler = get_scaler(train_set.independent_matrix)
    normalized_independent_matrix = scaler.transform(train_set.independent_matrix)
    normalized_test_matrix = scaler.transform(test_set.independent_matrix)
    
    n_components = 9
    rca = variable_trans.perform_rca(normalized_independent_matrix, n_components=n_components, name=data_set_name)
    rcs = [0, 5, 7, ]
    rca_independent_matrix = rca.transform(normalized_independent_matrix)[:, rcs]
    rw = (rca.inverse_transform(rca_independent_matrix, rcs) ** 2).sum() / (normalized_independent_matrix ** 2).sum()
    print(rw)
    rca_test_matrix = rca.transform(normalized_test_matrix)[:, rcs]
    
    nn = neural_net.find_nn(rca_independent_matrix, train_set.dependent_vector, 'abalone_rca')
    neural_net.error_rate(rca_test_matrix, test_set.dependent_vector, nn, name='abalone_rca_test')
    
    clusters_wrapper(rca_independent_matrix, train_set, suffix, 3, 8, rca_test_matrix, test_set)


def run_rca_iris():
    data_set_name = 'iris'
    suffix = '_rca'
    train_set, test_set = get_train_test_ml_set(data_set_name)
    
    scaler = get_scaler(train_set.independent_matrix)
    normalized_independent_matrix = scaler.transform(train_set.independent_matrix)
    
    n_components = 4
    rca = variable_trans.perform_rca(normalized_independent_matrix, n_components=n_components, name=data_set_name)
    rcs = [0, 1]
    rca_independent_matrix = rca.transform(normalized_independent_matrix)[:, rcs]
    rw = (rca.inverse_transform(rca_independent_matrix, rcs) ** 2).sum() / (normalized_independent_matrix ** 2).sum()
    print(rw)
    
    clusters_wrapper(rca_independent_matrix, train_set, suffix, 0, 1)


if __name__ == '__main__':
    run_abalone()
    run_iris()
    run_pca_abalone()
    run_pca_iris()
    run_ica_abalone()
    run_ica_iris()
    run_kca_abalone()
    run_kca_iris()
    run_rca_abalone()
    run_rca_iris()
