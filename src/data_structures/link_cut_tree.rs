//! splay木によるLink-Cut Tree。森の接続・切断と、向き付きパスの集約。
use ac_library::Monoid;

const NONE: usize = usize::MAX;

struct Node<S> {
    children: [usize; 2],
    parent: usize,
    reversed: bool,
    value: S,
    forward: S,
    backward: S,
}

/// 頂点値をモノイドで集約するLink-Cut Tree。
///
/// 各操作は償却O(log N)。モノイドの結合・cloneがO(1)の場合。
/// メモリO(N)、頂点数は構築時に固定。非可換モノイドにも対応する。
/// `path_prod(u, v)`はuからvの順に、両端を含む頂点値を集約する。
/// 辺の値を扱う場合は、辺も頂点として挿入し、元の頂点には単位元を置く。
/// 集約が不要なら[`NoAggregation`]を使う。部分木集約・パス一括更新は提供しない。
/// 範囲外の頂点IDはpanicする。問い合わせも内部のsplay木を変更する。
///
/// ```
/// use ac_library::Max;
/// use cp_lib_rs::data_structures::LinkCutTree;
/// let mut tree = LinkCutTree::<Max<i64>>::new(vec![3, 7, 2]);
/// assert!(tree.link(0, 1));
/// assert!(tree.link(1, 2));
/// assert_eq!(tree.path_prod(0, 2), Some(7));
/// tree.set(1, 1);
/// assert_eq!(tree.path_prod(0, 2), Some(3));
/// assert!(tree.cut(1, 2));
/// assert!(!tree.connected(0, 2));
/// ```
///
/// # 辺クエリ: 辺を追加ノードで表す
///
/// `u --(重みw)-- v`を`u -- e -- v`とする。元の頂点には単位元を置き、
/// 辺ノードeに重みを置く。辺の追加・削除にはlink・cutをそれぞれ2回使う。
/// 頂点数がNなら、同時に接続する辺ノードは最大N-1個。以下の例のように
/// 切断したノードを再利用できる。辺IDと端点の対応は利用側で管理する。
/// 辺ノードをあらかじめ全候補分確保する場合、計算量のNはそのノード数も含む。
///
/// ```
/// use ac_library::Monoid;
/// use cp_lib_rs::data_structures::LinkCutTree;
///
/// // 重みと辺IDを保持し、同点なら大きい辺IDを選ぶ。
/// enum MaxEdge {}
/// impl Monoid for MaxEdge {
///     type S = Option<(i64, usize)>;
///     fn identity() -> Self::S { None }
///     fn binary_operation(a: &Self::S, b: &Self::S) -> Self::S {
///         (*a).max(*b)
///     }
/// }
/// // 元の頂点: 0, 1, 2 / 辺ノード: 3, 4
/// let mut tree = LinkCutTree::<MaxEdge>::new(vec![None; 5]);
/// tree.set(3, Some((3, 0))); // 辺ID 0: 0--1、重み3
/// tree.set(4, Some((7, 1))); // 辺ID 1: 1--2、重み7
/// assert!(tree.link(0, 3));
/// assert!(tree.link(3, 1));
/// assert!(tree.link(1, 4));
/// assert!(tree.link(4, 2));
/// assert_eq!(tree.path_prod(0, 2), Some(Some((7, 1))));
/// assert_eq!(tree.path_prod(0, 0), Some(None)); // パスに辺がない
///
/// // 辺ID 1を外し、ノード4を新しい辺0--2用に再利用する。
/// assert!(tree.cut(1, 4));
/// assert!(tree.cut(4, 2));
/// tree.set(4, Some((5, 2)));
/// assert!(tree.link(0, 4));
/// assert!(tree.link(4, 2));
/// assert_eq!(tree.path_prod(1, 2), Some(Some((5, 2))));
/// ```
///
/// 頂点値と辺値を同時に集約したい場合は、例えば`(頂点和, 辺和)`をモノイド値に
/// し、元の頂点には`(値, 0)`、辺ノードには`(0, 重み)`を置く。
/// 非可換な集約ではuからvへの順番を守る。辺の通過方向で値を変える機能はない。
pub struct LinkCutTree<M: Monoid> {
    nodes: Vec<Node<M::S>>,
    ancestors: Vec<usize>,
}

/// 集約情報を保持しない単位型モノイド。
///
/// ```
/// use cp_lib_rs::data_structures::{LinkCutTree, NoAggregation};
/// let mut forest = LinkCutTree::<NoAggregation>::new(vec![(); 10]);
/// forest.link(2, 3);
/// assert!(forest.connected(2, 3));
/// ```
pub enum NoAggregation {}
impl Monoid for NoAggregation {
    type S = ();
    fn identity() {}
    fn binary_operation(_: &(), _: &()) {}
}

impl<M: Monoid> LinkCutTree<M> {
    /// 与えた頂点値から、辺のない森をO(N)で構築する。
    pub fn new(values: Vec<M::S>) -> Self {
        Self {
            nodes: values
                .into_iter()
                .map(|value| Node {
                    children: [NONE; 2],
                    parent: NONE,
                    reversed: false,
                    forward: value.clone(),
                    backward: value.clone(),
                    value,
                })
                .collect(),
            ancestors: Vec::new(),
        }
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    fn check(&self, v: usize) {
        assert!(v < self.len(), "vertex out of bounds");
    }
    fn is_root(&self, v: usize) -> bool {
        let p = self.nodes[v].parent;
        p == NONE || !self.nodes[p].children.contains(&v)
    }
    fn pull(&mut self, v: usize) {
        let [l, r] = self.nodes[v].children;
        let mut f = self.nodes[v].value.clone();
        let mut b = f.clone();
        if l != NONE {
            f = M::binary_operation(&self.nodes[l].forward, &f);
            b = M::binary_operation(&b, &self.nodes[l].backward);
        }
        if r != NONE {
            f = M::binary_operation(&f, &self.nodes[r].forward);
            b = M::binary_operation(&self.nodes[r].backward, &b);
        }
        self.nodes[v].forward = f;
        self.nodes[v].backward = b;
    }
    fn reverse(&mut self, v: usize) {
        if v == NONE {
            return;
        }
        let n = &mut self.nodes[v];
        n.children.swap(0, 1);
        std::mem::swap(&mut n.forward, &mut n.backward);
        n.reversed ^= true;
    }
    fn push(&mut self, v: usize) {
        if self.nodes[v].reversed {
            let [l, r] = self.nodes[v].children;
            self.reverse(l);
            self.reverse(r);
            self.nodes[v].reversed = false;
        }
    }
    fn rotate(&mut self, v: usize) {
        let p = self.nodes[v].parent;
        let g = self.nodes[p].parent;
        let d = usize::from(self.nodes[p].children[1] == v);
        let middle = self.nodes[v].children[1 - d];
        if !self.is_root(p) {
            let gd = usize::from(self.nodes[g].children[1] == p);
            self.nodes[g].children[gd] = v;
        }
        self.nodes[v].parent = g;
        self.nodes[p].children[d] = middle;
        if middle != NONE {
            self.nodes[middle].parent = p;
        }
        self.nodes[v].children[1 - d] = p;
        self.nodes[p].parent = v;
        self.pull(p);
        self.pull(v);
    }
    fn splay(&mut self, v: usize) {
        self.ancestors.clear();
        let mut x = v;
        self.ancestors.push(x);
        while !self.is_root(x) {
            x = self.nodes[x].parent;
            self.ancestors.push(x);
        }
        while let Some(x) = self.ancestors.pop() {
            self.push(x);
        }
        while !self.is_root(v) {
            let p = self.nodes[v].parent;
            if !self.is_root(p) {
                let g = self.nodes[p].parent;
                if (self.nodes[p].children[1] == v) == (self.nodes[g].children[1] == p) {
                    self.rotate(p);
                } else {
                    self.rotate(v);
                }
            }
            self.rotate(v);
        }
    }
    fn access(&mut self, v: usize) {
        let mut last = NONE;
        let mut x = v;
        while x != NONE {
            self.splay(x);
            self.nodes[x].children[1] = last;
            // 切り離された右部分木のparentは、実木のパス親として残す。
            if last != NONE {
                self.nodes[last].parent = x;
            }
            self.pull(x);
            last = x;
            x = self.nodes[x].parent;
        }
        self.splay(v);
    }
    /// vを、その連結成分の根にする。
    pub fn make_root(&mut self, v: usize) {
        self.check(v);
        self.access(v);
        self.reverse(v);
    }
    /// 現在の実木の根。link・パス問い合わせ等でも根は変わり得る。
    pub fn find_root(&mut self, v: usize) -> usize {
        self.check(v);
        self.access(v);
        let mut x = v;
        loop {
            self.push(x);
            let l = self.nodes[x].children[0];
            if l == NONE {
                break;
            }
            x = l;
        }
        self.splay(x);
        x
    }
    /// 同じ連結成分に属するか。同じ頂点ならtrue。
    pub fn connected(&mut self, u: usize, v: usize) -> bool {
        self.check(u);
        self.check(v);
        u == v || self.find_root(u) == self.find_root(v)
    }
    /// 辺を追加する。既に連結ならfalse（辺集合は変更しない）。
    pub fn link(&mut self, u: usize, v: usize) -> bool {
        self.check(u);
        self.check(v);
        self.make_root(u);
        if self.find_root(v) == u {
            return false;
        }
        self.nodes[u].parent = v;
        true
    }
    /// 直接の辺(u, v)を削除する。存在しなければfalse。
    pub fn cut(&mut self, u: usize, v: usize) -> bool {
        self.check(u);
        self.check(v);
        if u == v {
            return false;
        }
        self.make_root(u);
        self.access(v);
        if self.nodes[v].children[0] != u || self.nodes[u].children[1] != NONE {
            return false;
        }
        self.nodes[v].children[0] = NONE;
        self.nodes[u].parent = NONE;
        self.pull(v);
        true
    }
    /// 頂点値を置き換え、パスの集約情報を更新する。
    pub fn set(&mut self, v: usize, value: M::S) {
        self.check(v);
        self.access(v);
        self.nodes[v].value = value;
        self.pull(v);
    }
    /// 頂点単体の値。内部の集約を壊さないため、可変参照は返さない。
    pub fn get(&self, v: usize) -> &M::S {
        &self.nodes[v].value
    }
    /// 非連結ならNone。問い合わせ後、uが実木の根になる。
    pub fn path_prod(&mut self, u: usize, v: usize) -> Option<M::S> {
        self.check(u);
        self.check(v);
        self.make_root(u);
        if self.find_root(v) != u {
            return None;
        }
        self.access(v);
        Some(self.nodes[v].forward.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    enum Sequence {}
    impl Monoid for Sequence {
        type S = Vec<i32>;
        fn identity() -> Self::S {
            vec![]
        }
        fn binary_operation(a: &Self::S, b: &Self::S) -> Self::S {
            a.iter().chain(b).copied().collect()
        }
    }

    fn path(adj: &[Vec<bool>], u: usize, v: usize) -> Option<Vec<usize>> {
        let mut parent = vec![NONE; adj.len()];
        parent[u] = u;
        let mut queue = VecDeque::from([u]);
        while let Some(x) = queue.pop_front() {
            for (y, &edge) in adj[x].iter().enumerate() {
                if edge && parent[y] == NONE {
                    parent[y] = x;
                    queue.push_back(y);
                }
            }
        }
        if parent[v] == NONE {
            return None;
        }
        let mut result = vec![v];
        let mut x = v;
        while x != u {
            x = parent[x];
            result.push(x);
        }
        result.reverse();
        Some(result)
    }

    #[test]
    fn random_operations_match_naive_forest_in_both_directions() {
        let n = 24;
        let mut values: Vec<i32> = (0..n as i32).collect();
        let mut tree = LinkCutTree::<Sequence>::new(values.iter().map(|&v| vec![v]).collect());
        let mut adj = vec![vec![false; n]; n];
        let mut random = 123u64;
        for step in 0..20000 {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (random >> 16) as usize % n;
            let v = (random >> 40) as usize % n;
            match random % 7 {
                0 | 1 => {
                    let possible = path(&adj, u, v).is_none();
                    assert_eq!(tree.link(u, v), possible);
                    if possible {
                        adj[u][v] = true;
                        adj[v][u] = true;
                    }
                }
                2 | 3 => {
                    assert_eq!(tree.cut(u, v), adj[u][v]);
                    adj[u][v] = false;
                    adj[v][u] = false;
                }
                4 => {
                    values[u] = step;
                    tree.set(u, vec![step]);
                    assert_eq!(tree.get(u), &vec![step]);
                }
                5 => {
                    tree.make_root(u);
                    assert_eq!(tree.find_root(u), u);
                }
                _ => assert_eq!(tree.connected(u, v), path(&adj, u, v).is_some()),
            }
            let expected =
                path(&adj, u, v).map(|p| p.into_iter().map(|i| values[i]).collect::<Vec<_>>());
            assert_eq!(tree.path_prod(u, v), expected);
            assert_eq!(
                tree.path_prod(v, u),
                expected.map(|mut p| {
                    p.reverse();
                    p
                })
            );
        }
    }

    #[test]
    fn long_chain_without_aggregation_is_iterative() {
        let n = 30000;
        let mut tree = LinkCutTree::<NoAggregation>::new(vec![(); n]);
        for v in 1..n {
            assert!(tree.link(v - 1, v));
        }
        assert!(tree.connected(0, n - 1));
        assert!(!tree.link(0, n - 1));
        assert!(!tree.cut(0, n - 1));
        assert!(tree.cut(n / 2 - 1, n / 2));
        assert!(!tree.connected(0, n - 1));
        assert_eq!(tree.path_prod(0, n - 1), None);
        assert!(tree.link(0, n - 1));
        assert_eq!(tree.path_prod(n / 2 - 1, n / 2), Some(()));
        assert!(LinkCutTree::<NoAggregation>::new(vec![]).is_empty());
    }
}
