import abc
import numpy as np
import tqdm
from sklearn.metrics import pairwise_distances


class SamplingMethod(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, X, y, seed, **kwargs):
        self.X = X
        self.y = y
        self.seed = seed

    def flatten_X(self):
        shape = self.X.shape
        flat_X = self.X
        if len(shape) > 2:
            flat_X = np.reshape(self.X, (-1, shape[1]))
        return flat_X


    @abc.abstractmethod
    def select_batch_(self):
        return

    def select_batch(self, **kwargs):
        return self.select_batch_(**kwargs)

    def to_dict(self):
        return None


class KCenterGreedy(SamplingMethod):
    def __init__(self, X, y, seed, metric='euclidean'):
        self.X = X
        self.y = y
        self.flat_X = self.flatten_X()
        self.name = 'kcenter'
        self.features = self.flat_X
        self.metric = metric
        self.min_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.
        Args:
        cluster_centers: indices of cluster centers
        only_new: only calculate distance for newly selected points and update
            min_distances.
        rest_dist: whether to reset min_distances.
        """

        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers if d not in self.already_selected]
        if cluster_centers:
            # Update min_distances for all examples given new cluster center.
            x = self.features[cluster_centers]
            dist = pairwise_distances(self.features, x, metric=self.metric)

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch(self, N):
        """
        Greedy法による特徴量選択
        """
        # N個の特徴量を選択
        new_batch = []
        print('k center greedy | select N features')
        for _ in tqdm.tqdm(range(N)):
            if self.already_selected is None:
                # 初期化: feat_num(Xの特徴数)の中からランダムに一つindexを選択
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                # 現在選択されている特徴量との距離が最大の点(特徴)を選択
                ind = np.argmax(self.min_distances)

            x = self.features[[ind]] # 特徴量取得
            dist = pairwise_distances(self.features, x, metric=self.metric) # kcenter指標でindで選択された特徴料xとの距離を計算
            
            # 各特徴量に対する最も小さな距離を残す
            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1,1) # (feat_num, 1)
            else:
                # 以前計算された距離と現時点で計算した距離の小さい方法残す
                self.min_distances = np.minimum(self.min_distances, dist) #  (feat_num, 1)
            new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'% max(self.min_distances))

        return new_batch