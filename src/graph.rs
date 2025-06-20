use std::ops::Index;

/// 隣接リスト形式のグラフを行圧縮したグラフ
///
/// # Examples
///
/// ```
/// use cp_lib_rs::graph::{RowCompressedGraph, UnweightedEdge};
/// use itertools::Itertools;
///
/// let nodes = vec!["hoge", "fuga", "piyo"];
/// let edges = vec![
///     (1, UnweightedEdge::new(2)),
///     (2, UnweightedEdge::new(0)),
///     (0, UnweightedEdge::new(1)),
/// ];
/// let graph = RowCompressedGraph::new(3, nodes, edges);
///
/// assert_eq!(graph.len(), 3);
/// assert_eq!(graph.nodes()[0], "hoge");
/// assert_eq!(graph[0].iter().sorted().copied().collect_vec(), [UnweightedEdge::new(1)]);
/// ```
#[derive(Debug, Clone)]
pub struct RowCompressedGraph<V, E> {
    nodes: Vec<V>,
    edges: Vec<E>,
    pos: Vec<usize>,
    len: usize,
}

impl<V, E> RowCompressedGraph<V, E> {
    pub fn new(n: usize, nodes: Vec<V>, edges: Vec<(usize, E)>) -> Self {
        assert!(nodes.len() == n);
        let mut raw_edges = edges;
        raw_edges.sort_unstable_by_key(|(u, _)| *u);
        let mut edges = Vec::with_capacity(raw_edges.len());
        let mut pos = vec![0; n + 1];
        let mut index = 0;

        for (i, (u, e)) in raw_edges.into_iter().enumerate() {
            assert!(u < n);

            while index < u {
                index += 1;
                pos[index] = i;
            }

            edges.push(e);
        }

        pos[index + 1..].fill(edges.len());

        Self {
            nodes,
            edges,
            pos,
            len: n,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn nodes(&self) -> &[V] {
        &self.nodes
    }

    pub fn nodes_mut(&mut self) -> &mut [V] {
        &mut self.nodes
    }
}

impl<E> RowCompressedGraph<(), E> {
    pub fn new_edges(n: usize, edges: Vec<(usize, E)>) -> Self {
        Self::new(n, vec![(); n], edges)
    }
}

impl<V, E> Index<usize> for RowCompressedGraph<V, E> {
    type Output = [E];

    fn index(&self, index: usize) -> &Self::Output {
        let start = self.pos[index];
        let end = self.pos[index + 1];
        &self.edges[start..end]
    }
}

pub trait EdgeTo {
    fn to(&self) -> usize;
}

pub trait EdgeWeight<T>: EdgeTo {
    fn weight(&self) -> &T;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnweightedEdge {
    to: usize,
}

impl UnweightedEdge {
    pub fn new(to: usize) -> Self {
        Self { to }
    }
}

impl EdgeTo for UnweightedEdge {
    fn to(&self) -> usize {
        self.to
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WeightedEdge<T> {
    to: usize,
    weight: T,
}

impl<T> WeightedEdge<T> {
    pub fn new(to: usize, weight: T) -> Self {
        Self { to, weight }
    }
}

impl<T> EdgeTo for WeightedEdge<T> {
    fn to(&self) -> usize {
        self.to
    }
}

impl<T> EdgeWeight<T> for WeightedEdge<T> {
    fn weight(&self) -> &T {
        &self.weight
    }
}
