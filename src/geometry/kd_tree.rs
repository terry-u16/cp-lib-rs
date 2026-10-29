use super::squared_distance;
use std::collections::BinaryHeap;

const NONE: usize = usize::MAX;

#[derive(Debug, Clone)]
struct Node {
    id: usize,
    children: [usize; 2],
    min: [i32; 2],
    max: [i32; 2],
}

/// 静的な2次元KD-tree。座標はi32、距離はユークリッド距離。
///
/// 入力順のIDを維持し、同じ座標の点も別IDとして保持する。点の追加・移動・削除は
/// 再構築で行う。問い合わせ点自体も入力に含まれていれば検索結果に含む。
/// 距離の二乗はi32全域で正確に計算できるu128で返す。
///
/// 構築O(N log N)、メモリO(N)。各検索は境界箱で枝刈りするが、最悪では全点を見る。
/// k近傍は最悪O(N log(k+1) + k log(k+1))（kはN以下に切り詰める）。
///
/// ```
/// use cp_lib_rs::geometry::KdTree;
/// let tree = KdTree::new(vec![[0, 0], [3, 4], [10, 0]]);
/// assert_eq!(tree.nearest([2, 3]), Some((1, 2))); // (点ID, 距離の二乗)
/// assert_eq!(tree.k_nearest([0, 0], 2), vec![(0, 0), (1, 25)]);
/// let mut ids = tree.within_radius([0, 0], 5);
/// ids.sort_unstable();
/// assert_eq!(ids, vec![0, 1]); // 境界上の点も含む
/// ```
#[derive(Debug, Clone)]
pub struct KdTree {
    points: Vec<[i32; 2]>,
    nodes: Vec<Node>,
    root: usize,
}

impl KdTree {
    pub fn new(points: Vec<[i32; 2]>) -> Self {
        let mut ids: Vec<_> = (0..points.len()).collect();
        let mut tree = Self {
            nodes: Vec::with_capacity(points.len()),
            points,
            root: NONE,
        };
        tree.root = tree.build(&mut ids, 0);
        tree
    }
    pub fn len(&self) -> usize {
        self.points.len()
    }
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
    /// 入力順の点列。
    pub fn points(&self) -> &[[i32; 2]] {
        &self.points
    }

    fn build(&mut self, ids: &mut [usize], axis: usize) -> usize {
        if ids.is_empty() {
            return NONE;
        }
        let mid = ids.len() / 2;
        ids.select_nth_unstable_by_key(mid, |&i| (self.points[i][axis], i));
        let id = ids[mid];
        let index = self.nodes.len();
        self.nodes.push(Node {
            id,
            children: [NONE; 2],
            min: self.points[id],
            max: self.points[id],
        });
        let l = self.build(&mut ids[..mid], 1 - axis);
        let r = self.build(&mut ids[mid + 1..], 1 - axis);
        self.nodes[index].children = [l, r];
        for child in [l, r] {
            if child == NONE {
                continue;
            }
            for axis in 0..2 {
                self.nodes[index].min[axis] =
                    self.nodes[index].min[axis].min(self.nodes[child].min[axis]);
                self.nodes[index].max[axis] =
                    self.nodes[index].max[axis].max(self.nodes[child].max[axis]);
            }
        }
        index
    }
    fn lower_bound(&self, node: usize, query: [i32; 2]) -> u128 {
        if node == NONE {
            return u128::MAX;
        }
        let n = &self.nodes[node];
        let closest = [
            query[0].clamp(n.min[0], n.max[0]),
            query[1].clamp(n.min[1], n.max[1]),
        ];
        squared_distance(query, closest)
    }
    fn ordered_children(&self, node: usize, query: [i32; 2]) -> [(usize, u128); 2] {
        let [l, r] = self.nodes[node].children;
        let a = (l, self.lower_bound(l, query));
        let b = (r, self.lower_bound(r, query));
        if a.1 <= b.1 { [a, b] } else { [b, a] }
    }
    /// 最近傍の(点ID, 距離の二乗)。空ならNone。同距離では小さいIDを優先する。
    /// ヒープや結果Vecの確保は行わない。
    pub fn nearest(&self, query: [i32; 2]) -> Option<(usize, u128)> {
        if self.is_empty() {
            return None;
        }
        let mut best = (u128::MAX, usize::MAX);
        self.search_nearest(self.root, query, &mut best);
        Some((best.1, best.0))
    }
    fn search_nearest(&self, node: usize, query: [i32; 2], best: &mut (u128, usize)) {
        let id = self.nodes[node].id;
        *best = (*best).min((squared_distance(query, self.points[id]), id));
        for (child, bound) in self.ordered_children(node, query) {
            if child != NONE && bound <= best.0 {
                self.search_nearest(child, query, best);
            }
        }
    }
    /// 最大k点の(点ID, 距離の二乗)を、距離・IDの昇順で返す。k=0なら空。
    pub fn k_nearest(&self, query: [i32; 2], k: usize) -> Vec<(usize, u128)> {
        let k = k.min(self.len());
        if k == 0 {
            return Vec::new();
        }
        let mut heap = BinaryHeap::with_capacity(k);
        self.search_k(self.root, query, k, &mut heap);
        heap.into_sorted_vec()
            .into_iter()
            .map(|(d, id)| (id, d))
            .collect()
    }
    fn search_k(
        &self,
        node: usize,
        query: [i32; 2],
        k: usize,
        heap: &mut BinaryHeap<(u128, usize)>,
    ) {
        let id = self.nodes[node].id;
        let value = (squared_distance(query, self.points[id]), id);
        if heap.len() < k {
            heap.push(value);
        } else if value < *heap.peek().unwrap() {
            *heap.peek_mut().unwrap() = value;
        }
        for (child, bound) in self.ordered_children(node, query) {
            if child != NONE && (heap.len() < k || bound <= heap.peek().unwrap().0) {
                self.search_k(child, query, k, heap);
            }
        }
    }
    /// 半径radius以内の点ID。境界を含む。結果の順序は未規定。
    pub fn within_radius(&self, query: [i32; 2], radius: u64) -> Vec<usize> {
        let mut result = Vec::new();
        self.within_radius_each(query, radius, |id| result.push(id));
        result
    }
    /// 結果のVecを確保せず、半径内の点IDをコールバックに渡す。
    pub fn within_radius_each(&self, query: [i32; 2], radius: u64, mut visit: impl FnMut(usize)) {
        if self.root != NONE {
            self.search_radius(self.root, query, u128::from(radius).pow(2), &mut visit);
        }
    }
    fn search_radius(
        &self,
        node: usize,
        query: [i32; 2],
        radius2: u128,
        visit: &mut impl FnMut(usize),
    ) {
        if self.lower_bound(node, query) > radius2 {
            return;
        }
        let id = self.nodes[node].id;
        if squared_distance(query, self.points[id]) <= radius2 {
            visit(id);
        }
        for child in self.nodes[node].children {
            if child != NONE {
                self.search_radius(child, query, radius2, visit);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn searches_match_exhaustive_on_duplicates_and_extreme_coordinates() {
        let mut random = 42u64;
        for n in [0, 1, 2, 31, 200] {
            let mut points: Vec<_> = (0..n)
                .map(|_| {
                    random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                    [(random % 21) as i32 - 10, ((random >> 32) % 21) as i32 - 10]
                })
                .collect();
            if n > 3 {
                points[0] = [i32::MIN; 2];
                points[1] = [i32::MAX; 2];
            }
            let tree = KdTree::new(points.clone());
            assert_eq!(tree.points(), points);
            for query in [[0, 0], [5, -2], [i32::MIN, i32::MAX], [i32::MAX; 2]] {
                let mut expected: Vec<_> = points
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (squared_distance(p, query), i))
                    .collect();
                expected.sort_unstable();
                assert_eq!(tree.nearest(query), expected.first().map(|&(d, i)| (i, d)));
                for k in [0, 1, 3, 32, usize::MAX] {
                    assert_eq!(
                        tree.k_nearest(query, k),
                        expected
                            .iter()
                            .take(k)
                            .map(|&(d, i)| (i, d))
                            .collect::<Vec<_>>()
                    );
                }
                for radius in [0u64, 5, 10, u32::MAX as u64, u64::MAX] {
                    let mut actual = tree.within_radius(query, radius);
                    actual.sort_unstable();
                    let expected: Vec<_> = points
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| squared_distance(**p, query) <= u128::from(radius).pow(2))
                        .map(|(i, _)| i)
                        .collect();
                    assert_eq!(actual, expected);
                }
            }
        }
    }
    #[test]
    fn collinear_and_identical_points() {
        for points in [vec![[7, 9]; 100], (0..100).map(|i| [0, i]).collect()] {
            let tree = KdTree::new(points.clone());
            let mut expected: Vec<_> = points
                .iter()
                .enumerate()
                .map(|(i, &p)| (squared_distance(p, [7, 9]), i))
                .collect();
            expected.sort_unstable();
            assert_eq!(
                tree.k_nearest([7, 9], 100),
                expected
                    .into_iter()
                    .map(|(d, i)| (i, d))
                    .collect::<Vec<_>>()
            );
        }
    }
}
