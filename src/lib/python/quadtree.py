class QuadTree:
    """Splits (x, y, idx) items into quadtree tiles."""

    # note: this is an impperfect system: if certain parts of the map are 
    # extremely more dense than others, edge artifacts may show

    MAX_TILE_POINTS = 64000

    # within the adaptive picks, weight = (1 – BETA) + BETA·(1/ρ^ALPHA)
    BETA = 0.05
    ALPHA = 0.1

    # fraction of picks to allocate to pure spatial uniformity
    SPATIAL_FRACTION = 0.85

    def __init__(
        self, center_x, center_y, size, items, depth=0
    ):  # items are a list of (x, y, idx)
        self.nodes = []
        self.children = []
        self.center = [center_x, center_y]
        self.size = size  # distance from center to edge
        self.depth = depth

        self.build(items)

    def build(self, items):
        if len(items) <= self.MAX_TILE_POINTS:
            self.nodes = items
            self.children = []
            return

        synthetic_sample = self.get_synthetic_sample(items)
        real_sample = self.snap_to_real(synthetic_sample, items)
        self.nodes = real_sample

        sampled_idxs = {pt[2] for pt in real_sample}
        remaining_items = [pt for pt in items if pt[2] not in sampled_idxs]
        quad_list = self.get_quad_lists(remaining_items)
        self.split(quad_list)

    def get_quad_lists(self, items):
        #    0  |  2
        #    -------
        #    1  |  3
        quad_list = [[], [], [], []]
        for item in items:
            if item[0] <= self.center[0]:
                if item[1] <= self.center[1]:
                    quad_list[0].append(item)
                else:
                    quad_list[1].append(item)
            else:
                if item[1] <= self.center[1]:
                    quad_list[2].append(item)
                else:
                    quad_list[3].append(item)
        return quad_list

    def get_synthetic_sample(self, items):
        import numpy as np
        from sklearn.neighbors import NearestNeighbors

        # build coords array
        coords = np.stack([[x, y] for x, y, _ in items])  # (n,2)
        n = len(coords)
        M = self.MAX_TILE_POINTS

        # how many we uniform vs adaptive density sample
        M_uni = int(self.SPATIAL_FRACTION * M)  # uniform-spatial
        M_adapt = M - M_uni  # density-adaptive

        # density proxy via k-NN
        k = 8
        nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
        dists, _ = nbrs.kneighbors(coords)
        r_k = dists[:, -1]
        dens = 1.0 / (np.pi * r_k**2 + 1e-12)

        #  build adaptive weights
        w = (1 - self.BETA) + self.BETA * (1.0 / (dens**self.ALPHA))
        w /= w.sum()

        # adaptive picks (actual data points)
        idx_adapt = np.random.choice(n, size=M_adapt, replace=False, p=w)
        pts_adapt = coords[idx_adapt]

        # uniform-spatial picks (synthetic coords)
        x0, y0 = coords[:, 0].min(), coords[:, 1].min()
        x1, y1 = coords[:, 0].max(), coords[:, 1].max()
        pts_uni = np.column_stack(
            [
                np.random.uniform(x0, x1, size=M_uni),
                np.random.uniform(y0, y1, size=M_uni),
            ]
        )

        synthetic = np.vstack([pts_adapt, pts_uni])

        return synthetic

    def snap_to_real(self, synthetic, items):
        from sklearn.neighbors import KDTree
        import numpy as np
        import random

        """
        synthetic : np.ndarray of shape (M,2)  — “ideal” floats
        items     : list of (x, y, idx)        — the full real dataset at this node

        Returns a list of (x, y, idx) real points, one per synthetic sample (deduped).
        """
        # Extract coords into an (n,2) array
        coords = np.array([[x, y] for x, y, _ in items])
        # Build the KD-tree on those coords
        tree = KDTree(coords, leaf_size=40)

        #  Query each synthetic point’s nearest neighbor
        #  dists: (M,1), idxs: (M,1) into the coords/items array
        dists, idxs = tree.query(synthetic, k=1)
        idxs = idxs.ravel()  # shape (M,)

        # Deduplicate while preserving order
        seen = set()
        real_sample = []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                real_sample.append(items[i])

        # If we’re short, fill up with random unused points (if multiple synthetics snap to the same real point)
        if len(real_sample) < self.MAX_TILE_POINTS:
            all_indices = set(range(len(items)))
            unused = list(all_indices - seen)
            need = self.MAX_TILE_POINTS - len(real_sample)

            if len(unused) <= need:
                fill_idxs = unused
            else:
                fill_idxs = random.sample(unused, need)

            for i in fill_idxs:
                real_sample.append(items[i])
        print("got real sample")

        return real_sample

    def split(self, quad_lists):
        self.children = [
            QuadTree(
                self.center[0] - self.size / 2,
                self.center[1] - self.size / 2,
                self.size / 2,
                quad_lists[0],
                self.depth + 1,
            ),
            QuadTree(
                self.center[0] - self.size / 2,
                self.center[1] + self.size / 2,
                self.size / 2,
                quad_lists[1],
                self.depth + 1,
            ),
            QuadTree(
                self.center[0] + self.size / 2,
                self.center[1] - self.size / 2,
                self.size / 2,
                quad_lists[2],
                self.depth + 1,
            ),
            QuadTree(
                self.center[0] + self.size / 2,
                self.center[1] + self.size / 2,
                self.size / 2,
                quad_lists[3],
                self.depth + 1,
            ),
        ]

    def print_tree(self, indent=0):
        spacing = " " * indent
        print(
            f"{spacing}Depth: {self.depth} | Center: {self.center} | Size: {self.size} | Items: {len(self.nodes)}"
        )
        # print(f"{spacing} Item values: {self.nodes[0:10]}")
        for i, child in enumerate(self.children):
            print(f"{spacing} Child {i}:")
            child.print_tree(indent + 4)
