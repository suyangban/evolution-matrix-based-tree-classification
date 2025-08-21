import networkx as nx
import numpy as np
import random
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from scipy.stats import skew, kurtosis
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import matplotlib.pyplot as plt  

import warnings
warnings.filterwarnings("ignore")


ForestConfig = {    
    "prufer_random_trees": {
        "label": 1,
        "n_trees": 200,
        "seed": None,
        "n_nodes": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    },
    "ba_trees": {
        "label": 2,
        "n_trees": 200,
        "seed": None,
        "n_nodes":  [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    },
    "complete_binary_trees": {
        "label": 3,
        "n_trees": 200,
        "n_nodes": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] 
    },
}
ClusterMapping = {
    1: "Prüfer random trees",
    2: "BA trees",
    3: "Complete binary trees",
}

class TreeGenerator:
    """
    支持生成多种类型的树结构，包括：
    - Prüfer 随机树
    - BA 优先连接树
    - 完全平衡二叉树
    - 适合NLP的TreeBank语法树（简单二叉树结构）
    """

    @staticmethod
    def prufer_random_tree(n, seed=None):
        """生成 n 个节点的 Prufer 随机树"""
        if seed is not None:
            random.seed(seed)
        if n <= 1:
            return nx.empty_graph(n)
        prufer = [random.randint(0, n-1) for _ in range(n-2)]
        return nx.from_prufer_sequence(prufer)

    @staticmethod
    def ba_tree(n, m=1, seed=None):
        """生成 n 个节点的 BA 优先连接树（m=1时为树）"""
        if seed is not None:
            random.seed(seed)
        return nx.barabasi_albert_graph(n, m, seed=seed)

    @staticmethod
    def complete_binary_tree(n):
        """生成 n 个节点的完全平衡二叉树"""
        return nx.balanced_tree(r=2, h=(n-1).bit_length()-1)

    # @staticmethod
    # def generate_dependency_tree(text, language="en"):
    #     """
    #     从自然语言文本生成依存树
    #     :param text: 输入的文本
    #     :param language: 语言类型，支持 "en"（英语）或 "zh"（中文 not implemented now）
    #     :return: 生成的依存树（networkx.Graph）
    #     """
    #     G = nx.Graph()
    #     if language == "en":
    #         import spacy
    #         nlp = spacy.load("en_core_web_sm")
    #         doc = nlp(text)
    #         for token in doc:
    #             G.add_node(token.text, pos=token.pos_)
    #             if token.head != token:
    #                 G.add_edge(token.head.text, token.text, relation=token.dep_)
    #     else:
    #         raise ValueError("Unsupported language. Use 'en' or 'zh'.")
    #     return G

    @staticmethod
    def visualize_tree(G, with_labels=True, figsize=(8, 6)):
        """
        可视化树结构
        :param G: networkx 图对象
        :param with_labels: 是否显示节点标签
        :param figsize: 图像大小
        """
        plt.figure(figsize=figsize)
        if isinstance(G, nx.DiGraph):
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        else:
            pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=with_labels, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)


class ForestGenerator:
    """
    支持生成森林结构（多棵树的集合）
    """
    def __init__(self, forest_config:dict):
        self.forest_config = forest_config
        self.trees = []
        self.labels = []
    
    def generate_forest(self):
        """
        根据配置生成森林结构
        :return: 生成的树列表和对应标签
        """
        for tree_type, config in self.forest_config.items():
            if tree_type == "prufer_random_trees":
                for idx in range(config["n_trees"]):
                    # n = random.choice(config["n_nodes"])
                    n = config["n_nodes"][idx % len(config["n_nodes"])]
                    T = TreeGenerator.prufer_random_tree(n, seed=config.get("seed", random.randint(0, 1000)))
                    self.trees.append(T)
                    self.labels.append(config["label"])
            elif tree_type == "ba_trees":
                for idx in range(config["n_trees"]):
                    # n = random.choice(config["n_nodes"])
                    n = config["n_nodes"][idx % len(config["n_nodes"])]
                    T = TreeGenerator.ba_tree(n, m=1, seed=config.get("seed", random.randint(0, 1000)))
                    self.trees.append(T)
                    self.labels.append(config["label"])
            elif tree_type == "complete_binary_trees":
                for idx in range(config["n_trees"]):
                    # n = random.choice(config["n_nodes"])
                    n = config["n_nodes"][idx % len(config["n_nodes"])]
                    T = TreeGenerator.complete_binary_tree(n)
                    self.trees.append(T)
                    self.labels.append(config["label"])
            elif tree_type == "dependency_trees":
                text = config.get("text", "The quick brown fox jumps over the lazy dog.")
                language = config.get("language", "en")
                T = TreeGenerator.generate_dependency_tree(text, language)
                self.trees.append(T)
                self.labels.append(config["label"])
            else:
                raise ValueError(f"Unsupported tree type: {tree_type}")
        return self.trees, self.labels

class TreeClassifier:
    """
    树分类器，支持生成树的谱特征签名和计算树之间的距离矩阵
    """

    @staticmethod
    def build_A_from_tree(T: nx.Graph):
        """从树结构 T 构建边-边矩阵A"""
        if not nx.is_tree(T):
            raise ValueError("Input graph is not a tree.")
        edge_list = list(T.edges())
        m = len(edge_list)
        degrees = dict(T.degree())
        # 边 -> 索引 的映射
        edge_index = {frozenset(e): i for i, e in enumerate(edge_list)}
        A = np.zeros((m, m), dtype=np.float32)
        for i, (x, y) in enumerate(edge_list):
            A[i, i] = -(1 / degrees[x] + 1 / degrees[y])
            # x 处的相邻边
            for nbr in T.neighbors(x):
                if nbr != y:
                    j = edge_index[frozenset((x, nbr))]
                    A[i, j] = 1 / degrees[x]
            # y 处的相邻边
            for nbr in T.neighbors(y):
                if nbr != x:
                    j = edge_index[frozenset((y, nbr))]
                    A[i, j] = 1 / degrees[y]
        return A, edge_list
    
    @staticmethod
    def laplacian_matrix(G: nx.Graph):
        """计算拉普拉斯矩阵"""
        return nx.laplacian_matrix(G).toarray()
    
    @staticmethod
    def adjacency_matrix(G: nx.Graph):
        """计算邻接矩阵"""
        return nx.adjacency_matrix(G).toarray()
    
    @staticmethod
    def distance_matrix(G:nx.Graph):
        """树中每条边的距离是1, 计算树的距离矩阵"""
        for u, v in G.edges():
            G[u][v]['weight'] = 1
        return nx.floyd_warshall_numpy(G, weight='weight')

    @staticmethod
    def spectral_signature(M, topk=5, bottomk=5, eigval_bins=10, eigvec_bins=10, method="all_eigs"):
        """计算谱特征签名
        method: "all_eigs" 或 "stat" 或 "eigs"
        - "all_eigs": 计算所有特征值和最大特征向量的直方图
        - "eigs": 计算前 k 个和后 k 个特征值
        """
        # 有边界情况未考虑
        # 特征值的处理：
        if method == "all_eigs":
            # 取固定长度的特征值向量
            eigvals, eigvecs = eigh(M)
            eigvals = eigvals.real
            eigvecs = eigvecs.real

            # 方法一：z-score 标准化
            # eigvals_mean = eigvals.mean()
            # eigvals_std = eigvals.std()
            # if eigvals_std < 1e-12:
            #     eigvals_norm =  eigvals - eigvals_mean  # 如果标准差为0，直接返回均值向量
            # else:
            #     eigvals_norm = (eigvals - eigvals_mean) / (eigvals_std)
            # hist, _ = np.histogram(eigvals_norm, bins=eigval_bins, range=(0, 1), density=False)
            # eigvals_sig = hist / (hist.sum() + 1e-12)

            # 方法二：取特征向量的一些统计信息
            eigvals_sig = np.array([
                eigvals.min(),
                np.median(eigvals),
                eigvals.max(),
                eigvals.mean(),
                eigvals.std(),
                eigvals.var(),
                skew(eigvals),
                kurtosis(eigvals),
                np.percentile(eigvals, 25),
                np.percentile(eigvals, 75),
                np.sum(eigvals) / len(eigvals),  # 平均值
                np.sum(eigvals > 0) / len(eigvals),  # 正特征值
                np.sum(eigvals == 0) / len(eigvals),  # 零特征值
                np.sum(eigvals < 0) / len(eigvals)  # 负特征值
            ])

            # 取最大特征值及其对应特征向量
            idx_max = np.argmax(eigvals)
            vmax = eigvecs[:, idx_max]
            pv = np.abs(vmax)
            pv = pv / (pv.sum() + 1e-12)  # 防止除零错误
            # 计算直方图
            hist, _ = np.histogram(pv, bins=eigvec_bins, range=(0, 1), density=False)
            eigvec_sig = hist / (hist.sum() + 1e-12)  # 防止除零错误

            sig = np.concatenate((eigvals_sig, eigvec_sig))
            return sig
        elif method == "eigs":
            # 计算前 k 个特征值和特征向量
            eigvals_l, eigvecs_l = eigs(M, k=topk, which='LR') # LR: Largest Real part
            eigvals_s, eigvecs_s = eigs(M, k=bottomk, which='SR') # SR: Smallest Real part
            eigvals_l = eigvals_l.real[:topk]
            eigvals_s = eigvals_s.real[-bottomk:]
            eigvecs_l = eigvecs_l.real[:topk]
            eigvecs_s = eigvecs_s.real[-bottomk:]
            
            # 取最大特征值及其对应特征向量
            idx_max = np.argmax(eigvals_l)
            vmax = eigvecs_l[:, idx_max]
            pv = np.abs(vmax)
            pv = pv / (pv.sum() + 1e-12)
            # 计算直方图
            hist, _ = np.histogram(pv, bins=eigvec_bins, range=(0, 1), density=False)
            hist = hist / (hist.sum() + 1e-12)

            # 取最大特征值及其重数
            eigval_max = eigvals_l[idx_max]
            # eps = 1e-9  # 数值容忍误差
            # multiplicity = np.sum(np.abs(eigvals_l - lambda_max) < eps)

            sig = np.concatenate(([eigval_max], eigvals_l, eigvals_s, hist))
            return sig
    
    @staticmethod
    def feature_distance(signatures, metric='euclidean'):
        """计算谱特征签名之间的距离矩阵"""
        X = np.vstack(signatures)
        return squareform(pdist(X, metric='euclidean'))
    

class MatrixComparator:
    """
    对邻接矩阵、距离矩阵、拉普拉斯、ours对图的结构描述能力进行比较
    """
    def __init__(self):
        pass

    @staticmethod
    def tree_clustering(trees, labels, n_clusters=3, matrix_type='adjacency'):
        """
        使用 KMeans 对森林进行聚类
        :param trees: 生成的森林（树列表）
        :param labels: 每棵树的标签
        :param n_clusters: 聚类数目
        """
        
        signatures = []
        for T in trees:
            if matrix_type == 'adjacency':
                A, edges = TreeClassifier.adjacency_matrix(T), list(T.edges())
            elif matrix_type == 'laplacian':
                A, edges = TreeClassifier.laplacian_matrix(T), list(T.edges())
            elif matrix_type == 'distance':
                A, edges = TreeClassifier.distance_matrix(T), list(T.edges())
            elif matrix_type == 'ours':
                A, edges = TreeClassifier.build_A_from_tree(T)
            else:
                raise ValueError(f"Unsupported matrix type: {matrix_type}")
            sig = TreeClassifier.spectral_signature(A, topk=5, bottomk=5, eigval_bins=10, eigvec_bins=10, method="all_eigs")
            signatures.append(sig)
        D = TreeClassifier.feature_distance(signatures, metric='euclidean')

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(D)
        # cluster_labels = kmeans.fit_predict(signatures)
        return cluster_labels, D
    
    
    def compare_results(results, labels):

        fig = plt.figure(figsize=(10, 10))
        axs = fig.subplots(2, 2)
        for idx, (matrix_type, (cluster_labels, D)) in enumerate(results.items()):

            # 计算评价指标
            ari = Metrics.adjusted_rand_index(labels, cluster_labels)
            nmi = Metrics.normalized_mutual_info(labels, cluster_labels)
            print(f"{matrix_type:10s}: ARI: {ari:.4f}, NMI: {nmi:.4f}")

            # MDS可视化
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
            xy = mds.fit_transform(D)

            ax = axs[idx // 2, idx % 2]
            for c in set(cluster_labels):
                idx = [i for i, cl in enumerate(cluster_labels) if cl == c]
                ax.scatter(xy[idx, 0], xy[idx, 1], label=f'{ClusterMapping.get(c+1, c+1)}', alpha=0.6)
            ax.set_title(f"{matrix_type} (ARI={ari:.2f}, NMI={nmi:.2f})")
            ax.legend()
            # 在图内打印详细指标
            ax.text(0.05, 0.95, f"ARI={ari:.3f}\nNMI={nmi:.3f}", 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
        plt.tight_layout(h_pad=2)
        plt.savefig("tree_clustering_results.png")
        plt.show()

    def compare_results_metric(results, labels):
        metrics = []
        for idx, (matrix_type, (cluster_labels, D)) in enumerate(results.items()):

            # 计算评价指标
            ari = Metrics.adjusted_rand_index(labels, cluster_labels)
            nmi = Metrics.normalized_mutual_info(labels, cluster_labels)
            metrics.append(f"{matrix_type:10s}: {ari:.4f}, {nmi:.4f}")
        return metrics
            
        

class Metrics:
    """
    包含对分类效果的评价指标：ARI、NMI、F1-score
    """
    @staticmethod
    def adjusted_rand_index(labels_true, labels_pred):
        """计算调整后的兰德指数（ARI）"""
        return metrics.adjusted_rand_score(labels_true, labels_pred)

    @staticmethod
    def normalized_mutual_info(labels_true, labels_pred):
        """计算归一化互信息（NMI）"""
        return metrics.normalized_mutual_info_score(labels_true, labels_pred)

    @staticmethod
    def f1_score(labels_true, labels_pred):
        """计算F1-score"""
        return metrics.f1_score(labels_true, labels_pred, average='weighted')



if __name__ == "__main__":
    # 示例用法
    # tree_gen = TreeGenerator()

    # # 生成并可视化 Prufer 随机树
    # prufer_tree = tree_gen.prufer_random_tree(10, seed=42)
    # tree_gen.visualize_tree(prufer_tree, with_labels=True)

    # # 生成并可视化 BA 优先连接树
    # ba_tree = tree_gen.ba_tree(10, m=1)
    # tree_gen.visualize_tree(ba_tree, with_labels=True)

    # # 生成并可视化完全平衡二叉树
    # complete_tree = tree_gen.complete_binary_tree(15)
    # tree_gen.visualize_tree(complete_tree, with_labels=True)

    # plt.show()

    # tree_classifier = TreeClassifier()
    # A, edges = tree_classifier.build_A_from_tree(prufer_tree)
    # print("Edge-Edge Matrix A:\n", A)
    # sig = tree_classifier.spectral_signature(A)
    # print("Spectral Signature:\n", sig)

    # 生成森林
    forest_gen = ForestGenerator(ForestConfig)
    trees, labels = forest_gen.generate_forest()
    # print(f"Generated {len(trees)} trees with labels: {labels}")

    distance_res = MatrixComparator.tree_clustering(trees, labels, n_clusters=3, matrix_type="distance")
    adjacency_res = MatrixComparator.tree_clustering(trees, labels, n_clusters=3, matrix_type='adjacency')
    laplacian_res = MatrixComparator.tree_clustering(trees, labels, n_clusters=3, matrix_type='laplacian')
    Ours_res = MatrixComparator.tree_clustering(trees, labels, n_clusters=3, matrix_type='ours')

    results = {
        "Distance": distance_res,
        "Adjacency": adjacency_res,
        "Laplacian": laplacian_res,
        "Evolution (ours)": Ours_res,
    }

    MatrixComparator.compare_results(results, labels)


    

