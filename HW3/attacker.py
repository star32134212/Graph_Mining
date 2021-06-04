import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from copy import deepcopy
import utils
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import scipy.sparse as sp
from numba import jit

class BaseAttack(Module):
    """Abstract base class for target attack classes.
    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'
    """

    def __init__(self, model, nnodes, attack_structure=True, attack_features=False, device='cpu'):
        super(BaseAttack, self).__init__()

        self.surrogate = model
        self.nnodes = nnodes
        self.attack_structure = attack_structure
        self.attack_features = attack_features
        self.device = device

        if model is not None:
            self.nclass = model.nclass
            self.nfeat = model.nfeat
            self.hidden_sizes = model.hidden_sizes

        self.modified_adj = None
        self.modified_features = None

    def attack(self, ori_adj, n_perturbations, **kwargs):
        """Generate perturbations on the input graph.
        Parameters
        ----------
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        Returns
        -------
        None.
        """
        pass

    def check_adj(self, adj):
        """Check if the modified adjacency is symmetric and unweighted.
        """

        if type(adj) is torch.Tensor:
            adj = adj.cpu().numpy()
        
        # adj matrix是對稱的，所以(n1,n2)變更，(n2,n1)也要跟著變
        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        """
        if sp.issparse(adj):
            print('adj.tocsr().max()',adj.tocsr().max())
            assert adj.tocsr().max() == 1, "Max value should be 1!"
            assert adj.tocsr().min() == 0, "Min value should be 0!"
        else:
            assert adj.max() == 1, "Max value should be 1!"
            assert adj.min() == 0, "Min value should be 0!"
        """
    def save_adj(self, root=r'/tmp/', name='mod_adj'):
        """Save attacked adjacency matrix.
        Parameters
        ----------
        root :
            root directory where the variable should be saved
        name : str
            saved file name
        Returns
        -------
        None.
        """
        assert self.modified_adj is not None, \
                'modified_adj is None! Please perturb the graph first.'
        name = name + '.npz'
        modified_adj = self.modified_adj

        if type(modified_adj) is torch.Tensor:
            sparse_adj = utils.to_scipy(modified_adj)
            sp.save_npz(osp.join(root, name), sparse_adj)
        else:
            sp.save_npz(osp.join(root, name), modified_adj)

    def save_features(self, root=r'/tmp/', name='mod_features'):
        """Save attacked node feature matrix.
        Parameters
        ----------
        root :
            root directory where the variable should be saved
        name : str
            saved file name
        Returns
        -------
        None.
        """

        assert self.modified_features is not None, \
                'modified_features is None! Please perturb the graph first.'
        name = name + '.npz'
        modified_features = self.modified_features

        if type(modified_features) is torch.Tensor:
            sparse_features = utils.to_scipy(modified_features)
            sp.save_npz(osp.join(root, name), sparse_features)
        else:
            sp.save_npz(osp.join(root, name), modified_features)


class RND(BaseAttack):
    """As is described in Adversarial Attacks on Neural Networks for Graph Data (KDD'19),
    'Rnd is an attack in which we modify the structure of the graph. Given our target node v,
    in each step we randomly sample nodes u whose label is different from v and
    add the edge u,v to the graph structure
    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'
    Examples
    --------
    >>> from dataset import Dataset
    >>> from attacker import RND
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Attack Model
    >>> target_node = 0
    >>> model = RND()
    >>> # Attack
    >>> model.attack(adj, labels, idx_train, target_node, n_perturbations=5)
    >>> modified_adj = model.modified_adj
    >>> modified_features = model.modified_features
    """

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=True, device='cpu'):
        super(RND, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)


    def compute_new_a_hat_uv(self, potential_edges, adj, adj_preprocessed, target_node, N):
        """
        Compute the updated A_hat_square_uv entries that would result from inserting/deleting the input edges,
        for every edge.

        Parameters
        ----------
        potential_edges: np.array, shape [P,2], dtype int
            The edges to check.

        Returns
        -------
        sp.sparse_matrix: updated A_hat_square_u entries, a sparse PxN matrix, where P is len(possible_edges).
        """

        edges = np.array(adj.nonzero()).T
        edges_set = {tuple(x) for x in edges}
        A_hat_sq = adj_preprocessed @ adj_preprocessed
        values_before = A_hat_sq[target_node].toarray()[0]
        node_ixs = np.unique(edges[:, 0], return_index=True)[1]
        twohop_ixs = np.array(A_hat_sq.nonzero()).T
        degrees = adj.sum(0).A1 + 1

        ixs, vals = compute_new_a_hat_uv(edges, node_ixs, edges_set, twohop_ixs, values_before, degrees,
                                         potential_edges, target_node)
        ixs_arr = np.array(ixs)
        a_hat_uv = sp.coo_matrix((vals, (ixs_arr[:, 0], ixs_arr[:, 1])), shape=[len(potential_edges), N])

        return a_hat_uv    
    
    def struct_score(self, a_hat_uv, XW, label_u):
        """
        Compute structure scores, cf. Eq. 15 in the paper

        Parameters
        ----------
        a_hat_uv: sp.sparse_matrix, shape [P,2]
            Entries of matrix A_hat^2_u for each potential edge (see paper for explanation)

        XW: sp.sparse_matrix, shape [N, K], dtype float
            The class logits for each node.
        N: node數
        K: label數
        P: 可能的edge數
        Returns
        -------
        np.array [P,]
            The struct score for every row in a_hat_uv
        """
        
        logits = a_hat_uv.dot(XW)
        label_onehot = np.eye(XW.shape[1])[label_u] #XW.shape(1) = label數 [self.label_u]代表第幾類的onehot
        best_wrong_class_logits = (logits - 1000 * label_onehot).max(1)
        logits_for_correct_class = logits[:,label_u]
        struct_scores = logits_for_correct_class - best_wrong_class_logits
        
        return struct_scores

    
    def attack(self, ori_features, ori_adj, labels, idx_train, w1, w2, target_node, n_perturbations, **kwargs):
        """
        Randomly sample nodes u whose label is different from v and
        add the edge u,v to the graph structure. This baseline only
        has access to true class labels in training set
        Parameters
        ----------
        ori_features : scipy.sparse.csr_matrix
            Origina (unperturbed) node feature matrix.
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        """
        
        # 處理會用到的參數
        # ori_adj: sp.csr_matrix
        #print('number of pertubations: %s' % n_perturbations)
        #ori_adj[target_node] 是稀疏矩陣格式
        modified_adj = ori_adj.tolil() # Convert this matrix to List of Lists format.
        features = ori_features.copy().tolil()
        features_orig = features.copy().tolil()
        modified_features = ori_features.tolil()
        
        label = labels.copy() #所有node的label
        label_u = label[target_node] #target node的label分佈
        K = np.max(label)+1 #target node的label
        #print("targete node label (K):",K)
        
        delta_cutoff = 0.004
        row = ori_adj[target_node].todense().A1
        #print('w1 shape',w1.shape)
        #print('w2 shape',w2.shape)
        w = sp.csr_matrix(w1.dot(w2))
        adj = ori_adj.copy().tolil()
        adj_orig = adj.copy().tolil()
        N = adj.shape[0] # node 數
        structure_perturbations = []
        potential_edges = []   
        adj_preprocessed = utils.preprocess_graph(adj).tolil() # adj 前處理
        #print('adj_preprocessed',adj_preprocessed.shape) #(2110,2110)
        
        # attack_surrogate
        
        ## compute_logits
        logits_start = adj_preprocessed.dot(adj_preprocessed).dot(features.dot(w))[target_node].toarray()[0]
        ## strongest_wrong_class
        label_u_onehot = np.eye(K)[label_u]
        best_wrong_class = (logits_start - 1000*label_u_onehot).argmax()
        #print('logits_start',logits_start,logits_start.shape)
        #print('label_u_onehot',label_u_onehot,label_u_onehot.shape)
        ## compute surrogate_losses
        surrogate_losses = [logits_start[label_u] - logits_start[best_wrong_class]]
        #print('surrogate_losses',surrogate_losses)
        # strongest_wrong_class
        # surrogate_losses


        # Setup starting values of the likelihood ratio test.
        # perturb_structure
        degree_sequence_start = adj_orig.sum(0).A1
        current_degree_sequence = adj.sum(0).A1
        d_min = 2
        S_d_start = np.sum(np.log(degree_sequence_start[degree_sequence_start >= d_min]))
        current_S_d = np.sum(np.log(current_degree_sequence[current_degree_sequence >= d_min]))
        n_start = np.sum(degree_sequence_start >= d_min)
        current_n = np.sum(current_degree_sequence >= d_min)
        #alpha_start = compute_alpha(n_start, S_d_start, d_min)
        alpha_start = n_start / (S_d_start - n_start * np.log(d_min - 0.5)) + 1
        #log_likelihood_orig = compute_log_likelihood(n_start, alpha_start, S_d_start, d_min)
        log_likelihood_orig = n_start * np.log(alpha_start) + n_start * alpha_start * np.log(d_min) + (alpha_start + 1) * S_d_start
        #print('alpha_start',alpha_start)
        #print('log_likelihood_orig',log_likelihood_orig)
        
        # direct attack
        influencers = [target_node]
        potential_edges = np.column_stack((np.tile(target_node, N-1), np.setdiff1d(np.arange(N), target_node)))


        ### Nettack
        
        potential_edges = potential_edges.astype("int32")
        for _ in range(n_perturbations):
            # Do not consider edges that, if removed, result in singleton edges in the graph.
            #"""
            # Update the values for the power law likelihood ratio test.
            deltas = 2 * (1 - adj[tuple(potential_edges.T)].toarray()[0] )- 1
            d_edges_old = current_degree_sequence[potential_edges]
            d_edges_new = current_degree_sequence[potential_edges] + deltas[:, None]
            new_S_d, new_n = update_Sx(current_S_d, current_n, d_edges_old, d_edges_new, d_min) 
            new_alphas = compute_alpha(new_n, new_S_d, d_min)
            new_ll = compute_log_likelihood(new_n, new_alphas, new_S_d, d_min)
            alphas_combined = compute_alpha(new_n + n_start, new_S_d + S_d_start, d_min)
            new_ll_combined = compute_log_likelihood(new_n + n_start, alphas_combined, new_S_d + S_d_start, d_min)
            new_ratios = -2 * new_ll_combined + 2 * (new_ll + log_likelihood_orig)
            # Do not consider edges that, if added/removed, would lead to a violation of the
            # likelihood ration Chi_square cutoff value.
            powerlaw_filter = filter_chisquare(new_ratios, delta_cutoff)
            #"""
            potential_edges_final = potential_edges[powerlaw_filter]

            # Compute new entries in A_hat_square_uv
            a_hat_uv_new = self.compute_new_a_hat_uv(potential_edges_final, adj, adj_preprocessed, target_node, N) 
            #print('a_hat_uv_new',a_hat_uv_new.shape)
            # Compute the struct scores for each potential edge 
            struct_scores = self.struct_score(a_hat_uv_new, features.dot(w), label_u)
            #print('struct_scores',struct_scores.shape)
            best_edge_ix = struct_scores.argmin()
            best_edge_score = struct_scores.min()
            best_edge = potential_edges_final[best_edge_ix]
            #print("Edge perturbation: {}".format(best_edge))
            # perform edge perturbation

            adj[tuple(best_edge)] = adj[tuple(best_edge[::-1])] = 1 - adj[tuple(best_edge)]
            adj_preprocessed = utils.preprocess_graph(adj)

            structure_perturbations.append(tuple(best_edge))
            surrogate_losses.append(best_edge_score)
            #"""
            # Update likelihood ratio test values
            current_S_d = new_S_d[powerlaw_filter][best_edge_ix]
            current_n = new_n[powerlaw_filter][best_edge_ix]
            current_degree_sequence[best_edge] += deltas[powerlaw_filter][best_edge_ix]
            #"""
        #print('structure_perturbations',structure_perturbations)
        #print('number_perturbations',len(structure_perturbations))
        
        for node1,node2 in structure_perturbations:
            modified_adj[node1, node2] = 1 - modified_adj[node1, node2]
            modified_adj[node2, node1] = 1 - modified_adj[node2, node1]
        adj_preprocessed = utils.preprocess_graph(modified_adj).tolil()
        #print('modified_adj',adj_preprocessed[:1]) #變成float了(被normalize)
        self.check_adj(adj_preprocessed)
        self.modified_adj = adj_preprocessed        
        #self.check_adj(modified_adj)
        #self.modified_adj = modified_adj
        self.modified_features = modified_features
    
    def reset(self):
        """
        Reset Nettack
        """
        self.adj = self.adj_orig.copy()
        self.X_obs = self.X_obs_orig.copy()
        self.structure_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []

def filter_chisquare(ll_ratios, cutoff):
    return ll_ratios < cutoff


def compute_log_likelihood(n, alpha, S_d, d_min):
    """
    Compute log likelihood of the powerlaw fit.

    Parameters
    ----------
    n: int
        Number of entries in the old distribution that are larger than or equal to d_min.

    alpha: float
        The estimated alpha of the power law distribution

    S_d: float
         Sum of log degrees in the distribution that are larger than or equal to d_min.

    d_min: int
        The minimum degree of nodes to consider

    Returns
    -------
    float: the estimated log likelihood
    """

    return n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * S_d

def compute_alpha(n, S_d, d_min):
    """
    Approximate the alpha of a power law distribution.

    Parameters
    ----------
    n: int or np.array of int
        Number of entries that are larger than or equal to d_min

    S_d: float or np.array of float
         Sum of log degrees in the distribution that are larger than or equal to d_min

    d_min: int
        The minimum degree of nodes to consider

    Returns
    -------
    alpha: float
        The estimated alpha of the power law distribution
    """

    return n / (S_d - n * np.log(d_min - 0.5)) + 1

def update_Sx(S_old, n_old, d_old, d_new, d_min):
    """
    Update on the sum of log degrees S_d and n based on degree distribution resulting from inserting or deleting
    a single edge.

    Parameters
    ----------
    S_old: float
         Sum of log degrees in the distribution that are larger than or equal to d_min.

    n_old: int
        Number of entries in the old distribution that are larger than or equal to d_min.

    d_old: np.array, shape [N,] dtype int
        The old degree sequence.

    d_new: np.array, shape [N,] dtype int
        The new degree sequence

    d_min: int
        The minimum degree of nodes to consider

    Returns
    -------
    new_S_d: float, the updated sum of log degrees in the distribution that are larger than or equal to d_min.
    new_n: int, the updated number of entries in the old distribution that are larger than or equal to d_min.
    """

    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min

    d_old_in_range = np.multiply(d_old, old_in_range)
    d_new_in_range = np.multiply(d_new, new_in_range)

    new_S_d = S_old - np.log(np.maximum(d_old_in_range, 1)).sum(1) + np.log(np.maximum(d_new_in_range, 1)).sum(1)
    new_n = n_old - np.sum(old_in_range, 1) + np.sum(new_in_range, 1)

    return new_S_d, new_n

@jit(nopython=True)
def connected_after(u, v, connected_before, delta):
    if u == v:
        if delta == -1:
            return False
        else:
            return True
    else:
        return connected_before

@jit(nopython=True)
def compute_new_a_hat_uv(edge_ixs, node_nb_ixs, edges_set, twohop_ixs, values_before, degs, potential_edges, u):
    """
    Compute the new values [A_hat_square]_u for every potential edge, where u is the target node. C.f. Theorem 5.1
    equation 17.

    Parameters
    ----------
    edge_ixs: np.array, shape [E,2], where E is the number of edges in the graph.
        The indices of the nodes connected by the edges in the input graph.
    node_nb_ixs: np.array, shape [N,], dtype int
        For each node, this gives the first index of edges associated to this node in the edge array (edge_ixs).
        This will be used to quickly look up the neighbors of a node, since numba does not allow nested lists.
    edges_set: set((e0, e1))
        The set of edges in the input graph, i.e. e0 and e1 are two nodes connected by an edge
    twohop_ixs: np.array, shape [T, 2], where T is the number of edges in A_tilde^2
        The indices of nodes that are in the twohop neighborhood of each other, including self-loops.
    values_before: np.array, shape [N,], the values in [A_hat]^2_uv to be updated.
    degs: np.array, shape [N,], dtype int
        The degree of the nodes in the input graph.
    potential_edges: np.array, shape [P, 2], where P is the number of potential edges.
        The potential edges to be evaluated. For each of these potential edges, this function will compute the values
        in [A_hat]^2_uv that would result after inserting/removing this edge.
    u: int
        The target node

    Returns
    -------
    return_ixs: List of tuples
        The ixs in the [P, N] matrix of updated values that have changed
    return_values:

    """
    N = degs.shape[0]

    twohop_u = twohop_ixs[twohop_ixs[:, 0] == u, 1]
    nbs_u = edge_ixs[edge_ixs[:, 0] == u, 1]
    nbs_u_set = set(nbs_u)

    return_ixs = []
    return_values = []

    for ix in range(len(potential_edges)):
        edge = potential_edges[ix]
        edge_set = set(edge)
        degs_new = degs.copy()
        delta = -2 * ((edge[0], edge[1]) in edges_set) + 1
        degs_new[edge] += delta

        nbs_edge0 = edge_ixs[edge_ixs[:, 0] == edge[0], 1]
        nbs_edge1 = edge_ixs[edge_ixs[:, 0] == edge[1], 1]

        affected_nodes = set(np.concatenate((twohop_u, nbs_edge0, nbs_edge1)))
        affected_nodes = affected_nodes.union(edge_set)
        a_um = edge[0] in nbs_u_set
        a_un = edge[1] in nbs_u_set

        a_un_after = connected_after(u, edge[0], a_un, delta)
        a_um_after = connected_after(u, edge[1], a_um, delta)

        for v in affected_nodes:
            a_uv_before = v in nbs_u_set
            a_uv_before_sl = a_uv_before or v == u

            if v in edge_set and u in edge_set and u != v:
                if delta == -1:
                    a_uv_after = False
                else:
                    a_uv_after = True
            else:
                a_uv_after = a_uv_before
            a_uv_after_sl = a_uv_after or v == u

            from_ix = node_nb_ixs[v]
            to_ix = node_nb_ixs[v + 1] if v < N - 1 else len(edge_ixs)
            node_nbs = edge_ixs[from_ix:to_ix, 1]
            node_nbs_set = set(node_nbs)
            a_vm_before = edge[0] in node_nbs_set

            a_vn_before = edge[1] in node_nbs_set
            a_vn_after = connected_after(v, edge[0], a_vn_before, delta)
            a_vm_after = connected_after(v, edge[1], a_vm_before, delta)

            mult_term = 1 / np.sqrt(degs_new[u] * degs_new[v])

            sum_term1 = np.sqrt(degs[u] * degs[v]) * values_before[v] - a_uv_before_sl / degs[u] - a_uv_before / \
                        degs[v]
            sum_term2 = a_uv_after / degs_new[v] + a_uv_after_sl / degs_new[u]
            sum_term3 = -((a_um and a_vm_before) / degs[edge[0]]) + (a_um_after and a_vm_after) / degs_new[edge[0]]
            sum_term4 = -((a_un and a_vn_before) / degs[edge[1]]) + (a_un_after and a_vn_after) / degs_new[edge[1]]
            new_val = mult_term * (sum_term1 + sum_term2 + sum_term3 + sum_term4)

            return_ixs.append((ix, v))
            return_values.append(new_val)

    return return_ixs, return_values