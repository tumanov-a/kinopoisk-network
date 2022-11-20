import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools

from collections import defaultdict


def labels_list_parameters(graph, layout, parameter, delete_by_len_name):
    list_net_params = []
    if parameter == 'degree':
        select_parameter = dict(nx.degree_centrality(graph))
        multiple_coef = 5
    elif parameter == 'betweeness':
        select_parameter = dict(nx.betweenness_centrality(graph))
        multiple_coef = 5
    elif parameter == 'closeness':
        select_parameter = dict(nx.closeness_centrality(graph))
        multiple_coef = 5
    elif parameter == 'eigenvector':
        select_parameter = dict(nx.eigenvector_centrality(graph))
        multiple_coef = 5
    elif parameter == 'katz':
        select_parameter = dict(nx.katz_centrality(graph))
        multiple_coef = 5

    borders = np.c_[np.arange(0, 1, 0.05), np.arange(0.05, 1.05, 0.05)]

    for i, border in enumerate(borders):
        selected_nodes_name = [node for node in graph.nodes if select_parameter[node] > np.quantile(list(
            select_parameter.values()), border[0]) and select_parameter[node] <= np.quantile(list(select_parameter.values()), border[1])]
        net_params = dict()
        if selected_nodes_name and i >= 19:
            net_params['font_size'] = borders[i][1] * 10
            net_params['labels'] = dict()
            for name in selected_nodes_name:
                if delete_by_len_name and len(name) > 25:
                    continue
                viz_x, viz_y = layout[name][0], layout[name][1]
#                 new_viz_y = viz_y - (i // 2) ** 1/4 * 0.025
                if parameter not in ['eigenvector', 'katz']:
                    new_viz_y = viz_y + 0.001
                    new_viz_x = viz_x + 0.02
                    layout[name] = np.array([new_viz_x, new_viz_y])
                else:
                    viz_x, viz_y = layout[name][0], layout[name][1]
                    layout[name] = np.array([viz_x, viz_y])
                net_params['labels'][name] = name

            net_params['pos'] = layout
            net_params['G'] = graph
            list_net_params.append(net_params)
    return list_net_params
    raise NotImplementedError()


def plot_graph_by_centrality_measure(graph, layout, centrality_measure, delete_by_len_name=True):

    if centrality_measure == 'degree':
        centralities = nx.degree_centrality(graph)
        def node_size_func(x): return x * 8000
    elif centrality_measure == 'betweeness':
        centralities = nx.betweenness_centrality(graph)
        def node_size_func(x): return x ** 1/4 * 10000
    elif centrality_measure == 'closeness':
        centralities = nx.closeness_centrality(graph)
#         def node_size_func(x): return x ** 1/2 * 10000
        def node_size_func(x): return -np.log(x) * 100
    elif centrality_measure == 'eigenvector':
        centralities = nx.eigenvector_centrality(graph)
        def node_size_func(x): return x ** 1/2 * 1000
    elif centrality_measure == 'katz':
        centralities = nx.katz_centrality(graph)
        def node_size_func(x): return x * 1000

    top_n_names = list(labels_list_parameters(
        graph, layout, centrality_measure, delete_by_len_name)[0]['labels'].keys())
    top_n_inds = [i for i, (name, value) in enumerate(
        centralities.items()) if name in top_n_names]

    plt.figure(figsize=(17, 16))
    nx.draw_networkx_nodes(graph,
                           pos=layout,
#                            edgecolors='#cccccc',
                           node_color=['#EA0599' if i in top_n_inds else '#bbbbbb' for i, val in enumerate(
                               centralities)],
                           node_size=[node_size_func(dgc) for name, dgc in centralities.items()])
    nx.draw_networkx_edges(graph, pos=layout, alpha=0.2)

    for params in labels_list_parameters(graph, layout, centrality_measure, delete_by_len_name):
        nx.draw_networkx_labels(
            **params, verticalalignment='baseline', horizontalalignment='left', )
    plt.title(f'Visualization of {centrality_measure} centrality')
    plt.axis('off')
    plt.show()


def drawing_edges_params(graph, cliques):

    net_params = dict()
    net_params['G'] = graph
    net_params['pos'] = nx.kamada_kawai_layout(graph)
    net_params['edge_color'] = list()
    net_params['style'] = list()
    net_params['width'] = list()
    edges_in_cliques = [sorted(edge) for clique in cliques for edge in list(
        itertools.combinations(clique, 2))]

    for edge in graph.edges:
        include_edge = False
        s_edge = sorted(edge)

        if s_edge in edges_in_cliques:
            edge_width = net_params['width']
            edge_new_width = 2
            edge_width.append(edge_new_width)

            edge_color = net_params['edge_color']
            edge_new_color = '#35477D'
            edge_color.append(edge_new_color)

            edge_style = net_params['style']
            edge_new_style = 'solid'
            edge_style.append(edge_new_style)
        else:
            edge_width = net_params['width']
            edge_new_width = .5
            edge_width.append(edge_new_width)

            edge_color = net_params['edge_color']
            edge_new_color = '#F67280'
            edge_color.append(edge_new_color)

            edge_style = net_params['style']
            edge_new_style = 'dotted'
            edge_style.append(edge_new_style)

    return net_params
    raise NotImplementedError()


def drawing_nodes_params(graph, cliques):
    import itertools
    net_params = dict()
    net_params['G'] = graph
    net_params['pos'] = nx.kamada_kawai_layout(graph)
    net_params['node_color'] = list()
    net_params['node_size'] = list()

    edges_in_cliques = [sorted(edge) for clique in cliques for edge in list(
        itertools.combinations(clique, 2))]
    nodes_in_cliques = set(
        [node for edge in edges_in_cliques for node in edge])

    for node in graph.nodes:
        if node in nodes_in_cliques:
            node_colors = net_params['node_color']
            node_color = '#35477D'
            node_colors.append(node_color)

            node_sizes = net_params['node_size']
            node_size = 80
            node_sizes.append(node_size)
        else:
            node_colors = net_params['node_color']
            node_color = '#C06C84'
            node_colors.append(node_color)

            node_sizes = net_params['node_size']
            node_size = 30
            node_sizes.append(node_size)
    return net_params
    raise NotImplementedError()


def remove_bridges(G):
    n_cc = len(list(nx.connected_components(G)))
    while True:
        edge = max(nx.edge_betweenness(G).items(), key=lambda x: x[1])[0]
        G.remove_edge(edge[0], edge[1])
        if len(list(nx.connected_components(G))) > n_cc:
            break


def girvan_newman(G, n):
    labels = np.zeros((n, len(G)))
    _G = G.copy()
    for division in range(n):
        remove_bridges(_G)
        for i, cc in enumerate(nx.connected_components(_G)):
            labels[division, list(cc)] = i
    return labels


def modularity_gain(A, ee, mnode, old_comm, new_comm, m):
    old_mat = np.zeros(A.shape[0])
    old_mat[old_comm] = 1
    new_mat = np.zeros(A.shape[0])
    new_mat[new_comm] = 1
    return (((A - ee)[np.where(new_mat == 1)[0]].sum(axis=0) - (A - ee)[np.where(old_mat == 1)[0]].sum(axis=0)) / m)[mnode]


def louvain_method(G):

    # Phase 1: community unfolding
    communities = unfolded_communities(G)

    # Create labels
    labels = np.zeros(len(G))
    for i, c in enumerate(communities):
        labels[c] = i

    # Phase 2: network aggregation
    nextG = nx.empty_graph(len(communities), nx.MultiGraph)
    for e in G.edges:
        nextG.add_edge(labels[int(e[0])], labels[int(e[1])])

    return communities, labels, nextG


def unfolded_communities(G):
    # Proposed template:
    A = nx.to_numpy_array(G)
    m = A.sum() / 2
    ee = expected_edges(A, m)
    communities = [[n] for n in G.nodes]  # initial partition
    max_modularity_gain = 1
    for node in G.nodes:
        '''
        1) Remove the node from the initial community.
        2) Iterate all neighboring communities and put a node 
            in the community with maximal modularity gain. If 
            there is no modularity gain, return the node into 
            the initial community.
        '''
        path_lengths = nx.single_source_dijkstra_path_length(G, node)
        neighbors_nodes = [node for node,
                           length in path_lengths.items() if length == 1]
        initial_communities = communities.copy()

        m1 = nx.algorithms.community.modularity(G, initial_communities)

        gains = []
        potential_new_communities = []
        for neighbor_node in neighbors_nodes:
            new_communities = []
            for communitie in initial_communities:
                if neighbor_node in communitie and node not in communitie:
                    extended_communitie = communitie.copy()
                    extended_communitie.append(node)
                    new_communities.append(extended_communitie)
                    continue
                elif node in communitie and len(communitie) == 1:
                    continue
                new_communities.append(communitie)
            try:
                m2 = nx.algorithms.community.modularity(G, new_communities)
            except:
                continue
            nx_gain = m2 - m1
            gains.append(nx_gain)
            potential_new_communities.append(new_communities)

        gains_with_potential_communitites = zip(
            gains, potential_new_communities)
        max_potential_gain, max_gain_potential_community = max(
            gains_with_potential_communitites, key=lambda x: x[0])
        if max_potential_gain > 0:
            communities = max_gain_potential_community
        else:
            continue
    return [c for c in communities if len(c)]


def expected_edges(A, m):
    degree = A.sum(axis=0)
    return degree[:, None] * degree[None, :] / 2 / m


def kronecker(A, communities):
    res = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            for com in communities:
                if i in com and j in com:
                    res[i, j] = 1
    return res


def modularity(A, communities):
    m = A.sum() / 2
    res = (A - expected_edges(A, m)) * kronecker(A, communities)
    return res.sum() / 2 / m


def edge_betw_modularity(G, n):
    A = nx.to_numpy_array(G)
    labels = girvan_newman(G, n)
    res = []
    for i in range(labels.shape[0]):
        comm = []
        for j in range(len(set(labels[i]))):
            comm.append(list(np.where(labels[i] == j)[0]))
        res.append(modularity(A, comm))
    return np.array(res)


def asyn_fluid(G, n):
    n_asyn_labels = np.empty([n, len(G.nodes)])
    for k in range(2, n + 2):
        asyn_fluidc_labels = list(nx.algorithms.community.asyn_fluidc(G, k))
        fluidc_labels = [[j for j, l in enumerate(
            asyn_fluidc_labels) if i in l][0] for i in range(len(G.nodes))]
        n_asyn_labels[k - 2, :] = fluidc_labels
#         print(n_asyn_labels)
    return n_asyn_labels


def asyn_fluidc_modularity(G, n):
    A = nx.to_numpy_array(G)
    labels = asyn_fluid(G, n)
    res = []
    for i in range(labels.shape[0]):
        comm = []
        for j in range(len(set(labels[i]))):
            comm.append(list(np.where(labels[i] == j)[0]))
        res.append(modularity(A, comm))
    return np.array(res)


def empirical_cdf(g: nx.Graph):
    from itertools import accumulate
    num_of_nodes = np.array(nx.degree_histogram(g))
    probs = num_of_nodes / num_of_nodes.sum()
    accumulate_probs = np.array(list(accumulate(probs)))
    return accumulate_probs


def power_law_cdf(x, alpha=3.5, x_min=1):
    return 1 - x**(-alpha+1) / x_min**(-alpha+1)