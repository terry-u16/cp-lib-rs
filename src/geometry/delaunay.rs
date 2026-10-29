//! 分割統治とquad-edgeによるDelaunay三角形分割。
//!
//! トポロジー操作・マージ手順はKACTL FastDelaunay.hを参考にRustへ移植。
//! <https://github.com/kth-competitive-programming/kactl/blob/main/content/geometry/FastDelaunay.h>
//! 原実装: Philippe Legault, <https://github.com/Bathlamos/delaunay-triangulation/>
//! 整数の厳密判定・入力ID・重複点処理・面の列挙はこの実装で追加。
//
// Copyright (c) 2016 Philippe Legault
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

use super::{DELAUNAY_COORD_LIMIT, validate_point};

/// i32整数座標の静的な2次元Delaunay三角形分割。
///
/// 分割統治によりO(N log N)時間、O(N)メモリ。入力順の点IDを返す。
/// 外接円の内部に他の点がない三角形を生成する（円周上の点は許す）。
/// 各座標を±10^9以内に制限する（範囲外はpanic）。向きはi64、外接円判定の
/// 積と和だけi128で厳密に計算する。浮動小数点・多倍長整数は使わない。
///
/// - 同一座標は最小の入力IDにまとめる。[`Self::representatives`]で対応を返す。
/// - 全点が共線なら三角形は空、辺は座標順の隣接点を結ぶ鎖になる。
/// - 0点・1点なら辺も三角形も空。
/// - 共円などで分割が一意でない場合は、そのうち1つを決定的に返す。
///
/// ```
/// use cp_lib_rs::geometry::Delaunay;
/// let triangulation = Delaunay::new(&[[0, 0], [4, 0], [0, 3]]);
/// assert_eq!(triangulation.triangles().len(), 1);
/// assert_eq!(triangulation.edges(), &[[0, 1], [0, 2], [1, 2]]);
/// ```
///
/// # ユークリッドMSTの候補辺にする
///
/// 辺の長さを丸めないユークリッド距離なら、この辺集合からMSTを求められる。
/// 距離の二乗でソートしてよい。障害物迂回など別の辺コストでは保証されない。
/// 重複点も全て接続する場合は、代表点と重複点の間に距離0の辺を補う。
///
/// ```
/// use ac_library::Dsu;
/// use cp_lib_rs::geometry::Delaunay;
/// let points = [[0i32, 0], [3, 0], [0, 4], [0, 0]];
/// let dt = Delaunay::new(&points);
/// let mut edges: Vec<_> = dt.edges().iter().map(|&[u, v]| {
///     let dx = i64::from(points[u][0]) - i64::from(points[v][0]);
///     let dy = i64::from(points[u][1]) - i64::from(points[v][1]);
///     ((dx * dx + dy * dy) as i64, u, v)
/// }).collect();
/// for (id, &representative) in dt.representatives().iter().enumerate() {
///     if id != representative { edges.push((0, id, representative)); }
/// }
/// edges.sort_unstable();
/// let mut dsu = Dsu::new(points.len());
/// let mut mst = Vec::new();
/// for (_, u, v) in edges {
///     if !dsu.same(u, v) {
///         dsu.merge(u, v);
///         mst.push((u, v));
///     }
/// }
/// assert_eq!(mst.len(), points.len() - 1);
/// ```
#[derive(Debug, Clone)]
pub struct Delaunay {
    triangles: Vec<[usize; 3]>,
    edges: Vec<[usize; 2]>,
    representatives: Vec<usize>,
}

impl Delaunay {
    pub fn new(points: &[[i32; 2]]) -> Self {
        points
            .iter()
            .copied()
            .for_each(|p| validate_point(p, DELAUNAY_COORD_LIMIT));
        let mut ids: Vec<_> = (0..points.len()).collect();
        ids.sort_unstable_by_key(|&i| (points[i], i));
        let mut representatives = vec![0; points.len()];
        let mut unique: Vec<usize> = Vec::with_capacity(ids.len());
        for id in ids {
            if unique.last().is_none_or(|&last| points[last] != points[id]) {
                unique.push(id);
            }
            representatives[id] = *unique.last().unwrap();
        }
        let mut result = Self {
            triangles: Vec::new(),
            edges: Vec::new(),
            representatives,
        };
        if unique.len() < 2 {
            return result;
        }
        let mut mesh = Mesh {
            points,
            quads: Vec::new(),
            free: Vec::new(),
        };
        mesh.build(&unique);
        for q in &mesh.quads {
            if q.alive {
                let [u, v] = q.vertices;
                result.edges.push([u.min(v), u.max(v)]);
            }
        }
        result.edges.sort_unstable();
        let mut seen = vec![[false; 2]; mesh.quads.len()];
        for block in 0..mesh.quads.len() {
            if !mesh.quads[block].alive {
                continue;
            }
            for direction in [0, 2] {
                if seen[block][direction / 2] {
                    continue;
                }
                let start = block * 4 + direction;
                let mut edge = start;
                let mut vertices = [0; 3];
                let mut count = 0;
                loop {
                    seen[edge / 4][(edge % 4) / 2] = true;
                    if count < 3 {
                        vertices[count] = mesh.origin(edge);
                    }
                    count += 1;
                    edge = mesh.left_next(edge);
                    if edge == start {
                        break;
                    }
                }
                // 外側の面は時計回り。共線の場合も除外する。
                if count == 3
                    && orient(
                        points[vertices[0]],
                        points[vertices[1]],
                        points[vertices[2]],
                    ) > 0
                {
                    let min_pos = (0..3).min_by_key(|&i| vertices[i]).unwrap();
                    vertices.rotate_left(min_pos);
                    result.triangles.push(vertices);
                }
            }
        }
        result.triangles.sort_unstable();
        result
    }
    /// 入力点IDによる反時計回りの三角形。最小IDを先頭にし、辞書順に並べる。
    pub fn triangles(&self) -> &[[usize; 3]] {
        &self.triangles
    }
    /// 重複のない無向辺。各辺はu<v、一覧は辞書順。重複座標の点は代表IDのみ使う。
    pub fn edges(&self) -> &[[usize; 2]] {
        &self.edges
    }
    /// 各入力点と同じ座標の、最小の入力ID。代表点自身は自身のIDになる。
    pub fn representatives(&self) -> &[usize] {
        &self.representatives
    }
}

fn orient(a: [i32; 2], b: [i32; 2], c: [i32; 2]) -> i64 {
    let bx = i64::from(b[0]) - i64::from(a[0]);
    let by = i64::from(b[1]) - i64::from(a[1]);
    let cx = i64::from(c[0]) - i64::from(a[0]);
    let cy = i64::from(c[1]) - i64::from(a[1]);
    bx * cy - by * cx
}

/// a,b,cが反時計回りなら、pが外接円の厳密な内部にあるか。
fn in_circle(a: [i32; 2], b: [i32; 2], c: [i32; 2], p: [i32; 2]) -> bool {
    let norm = |v: [i32; 2]| i64::from(v[0]).pow(2) + i64::from(v[1]).pow(2);
    let p2 = norm(p);
    // 座標上限B=10^9。ノルム差は最大2B^2、orientは最大8B^2でi64内。
    // 積と和はi128へ拡張する。3項の絶対値の和は最大48B^4 = 4.8e37 < i128::MAX。
    // pからの相対座標のノルムではなく、原点からのノルム差で範囲を抑える。
    i128::from(orient(p, a, b)) * i128::from(norm(c) - p2)
        + i128::from(orient(p, b, c)) * i128::from(norm(a) - p2)
        + i128::from(orient(p, c, a)) * i128::from(norm(b) - p2)
        > 0
}

// 1つの無向辺に4つの有向quad-edgeを割り当てる。0,2は実頂点、1,3は双対側。
struct Quad {
    next: [usize; 4],
    vertices: [usize; 2],
    alive: bool,
}
struct Mesh<'a> {
    points: &'a [[i32; 2]],
    quads: Vec<Quad>,
    free: Vec<usize>,
}
fn rot(e: usize) -> usize {
    (e & !3) | ((e + 1) & 3)
}
fn symmetric(e: usize) -> usize {
    e ^ 2
}
impl Mesh<'_> {
    fn origin(&self, e: usize) -> usize {
        self.quads[e / 4].vertices[(e % 4) / 2]
    }
    fn destination(&self, e: usize) -> usize {
        self.origin(symmetric(e))
    }
    fn next(&self, e: usize) -> usize {
        self.quads[e / 4].next[e % 4]
    }
    fn set_next(&mut self, e: usize, next: usize) {
        self.quads[e / 4].next[e % 4] = next;
    }
    fn previous(&self, e: usize) -> usize {
        rot(self.next(rot(e)))
    }
    fn left_next(&self, e: usize) -> usize {
        self.previous(symmetric(e))
    }
    fn make_edge(&mut self, from: usize, to: usize) -> usize {
        let block = self.free.pop().unwrap_or_else(|| {
            self.quads.push(Quad {
                next: [0; 4],
                vertices: [0; 2],
                alive: false,
            });
            self.quads.len() - 1
        });
        let e = block * 4;
        self.quads[block] = Quad {
            next: [e, e + 3, e + 2, e + 1],
            vertices: [from, to],
            alive: true,
        };
        e
    }
    fn splice(&mut self, a: usize, b: usize) {
        let alpha = rot(self.next(a));
        let beta = rot(self.next(b));
        let an = self.next(alpha);
        let bn = self.next(beta);
        self.set_next(alpha, bn);
        self.set_next(beta, an);
        let an = self.next(a);
        let bn = self.next(b);
        self.set_next(a, bn);
        self.set_next(b, an);
    }
    fn connect(&mut self, a: usize, b: usize) -> usize {
        let e = self.make_edge(self.destination(a), self.origin(b));
        self.splice(e, self.left_next(a));
        self.splice(symmetric(e), b);
        e
    }
    fn delete(&mut self, e: usize) {
        self.splice(e, self.previous(e));
        self.splice(symmetric(e), self.previous(symmetric(e)));
        self.quads[e / 4].alive = false;
        self.free.push(e / 4);
    }
    fn side(&self, point: usize, edge: usize) -> i64 {
        orient(
            self.points[self.origin(edge)],
            self.points[self.destination(edge)],
            self.points[point],
        )
    }
    fn circle(&self, a: usize, b: usize, c: usize, p: usize) -> bool {
        in_circle(
            self.points[a],
            self.points[b],
            self.points[c],
            self.points[p],
        )
    }
    // 座標順の点列を分割し、左右の外側に沿う有向辺を返す。
    fn build(&mut self, ids: &[usize]) -> (usize, usize) {
        if ids.len() == 2 {
            let a = self.make_edge(ids[0], ids[1]);
            return (a, symmetric(a));
        }
        if ids.len() == 3 {
            let a = self.make_edge(ids[0], ids[1]);
            let b = self.make_edge(ids[1], ids[2]);
            self.splice(symmetric(a), b);
            let side = orient(
                self.points[ids[0]],
                self.points[ids[1]],
                self.points[ids[2]],
            );
            if side == 0 {
                return (a, symmetric(b));
            }
            let c = self.connect(b, a);
            return if side > 0 {
                (a, symmetric(b))
            } else {
                (symmetric(c), c)
            };
        }
        let mid = ids.len() / 2;
        let (mut outer_left, mut inner_left) = self.build(&ids[..mid]);
        let (mut inner_right, mut outer_right) = self.build(&ids[mid..]);
        loop {
            if self.side(self.origin(inner_right), inner_left) > 0 {
                inner_left = self.left_next(inner_left);
            } else if self.side(self.origin(inner_left), inner_right) < 0 {
                inner_right = self.next(symmetric(inner_right));
            } else {
                break;
            }
        }
        let mut base = self.connect(symmetric(inner_right), inner_left);
        if self.origin(inner_left) == self.origin(outer_left) {
            outer_left = symmetric(base);
        }
        if self.origin(inner_right) == self.origin(outer_right) {
            outer_right = base;
        }
        loop {
            let mut left = self.next(symmetric(base));
            if self.side(self.destination(left), base) < 0 {
                while self.circle(
                    self.destination(base),
                    self.origin(base),
                    self.destination(left),
                    self.destination(self.next(left)),
                ) {
                    let next = self.next(left);
                    self.delete(left);
                    left = next;
                }
            }
            let mut right = self.previous(base);
            if self.side(self.destination(right), base) < 0 {
                while self.circle(
                    self.destination(base),
                    self.origin(base),
                    self.destination(right),
                    self.destination(self.previous(right)),
                ) {
                    let next = self.previous(right);
                    self.delete(right);
                    right = next;
                }
            }
            let left_valid = self.side(self.destination(left), base) < 0;
            let right_valid = self.side(self.destination(right), base) < 0;
            if !left_valid && !right_valid {
                break;
            }
            if !left_valid
                || (right_valid
                    && self.circle(
                        self.destination(left),
                        self.origin(left),
                        self.origin(right),
                        self.destination(right),
                    ))
            {
                base = self.connect(right, symmetric(base));
            } else {
                base = self.connect(symmetric(base), symmetric(left));
            }
        }
        (outer_left, outer_right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::squared_distance;
    use ac_library::Dsu;
    use std::collections::BTreeSet;

    // 実装の外接円式から独立した4x4行列式。各項<=2B^4、24項でもi128内。
    fn determinant(a: [i32; 2], b: [i32; 2], c: [i32; 2], p: [i32; 2]) -> i128 {
        let rows = [a, b, c, p].map(|[x, y]| {
            let x = i64::from(x);
            let y = i64::from(y);
            [x, y, x * x + y * y, 1]
        });
        let mut result = 0;
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    for l in 0..4 {
                        let cols = [i, j, k, l];
                        if (0..4).any(|x| (x + 1..4).any(|y| cols[x] == cols[y])) {
                            continue;
                        }
                        let inversions: usize = (0..4)
                            .map(|x| (x + 1..4).filter(|&y| cols[x] > cols[y]).count())
                            .sum();
                        let product: i128 = (0..4).map(|x| i128::from(rows[x][cols[x]])).product();
                        result += if inversions % 2 == 0 {
                            product
                        } else {
                            -product
                        };
                    }
                }
            }
        }
        result
    }
    fn mst_weights(points: &[[i32; 2]], edges: impl IntoIterator<Item = [usize; 2]>) -> Vec<i64> {
        let mut weighted: Vec<_> = edges
            .into_iter()
            .map(|[u, v]| (squared_distance(points[u], points[v]), u, v))
            .collect();
        weighted.sort_unstable();
        let mut dsu = Dsu::new(points.len());
        let mut weights = Vec::new();
        let mut count = 0;
        for (weight, u, v) in weighted {
            if !dsu.same(u, v) {
                dsu.merge(u, v);
                weights.push(weight);
                count += 1;
            }
        }
        assert_eq!(count, points.len().saturating_sub(1));
        weights
    }
    fn check(points: &[[i32; 2]]) {
        let dt = Delaunay::new(points);
        let unique: Vec<_> = (0..points.len())
            .filter(|&i| dt.representatives[i] == i)
            .collect();
        for (i, &r) in dt.representatives.iter().enumerate() {
            assert_eq!(points[i], points[r]);
            assert!(r <= i);
        }
        assert!(dt.edges.windows(2).all(|w| w[0] < w[1]));
        assert!(dt.triangles.windows(2).all(|w| w[0] < w[1]));
        let edge_set: BTreeSet<_> = dt.edges.iter().copied().collect();
        for &[a, b, c] in &dt.triangles {
            assert!(orient(points[a], points[b], points[c]) > 0);
            for &p in points {
                assert!(
                    determinant(points[a], points[b], points[c], p) <= 0,
                    "point in circumcircle: {points:?}, triangle {:?}",
                    [a, b, c]
                );
            }
            for [u, v] in [[a, b], [b, c], [c, a]] {
                assert!(edge_set.contains(&[u.min(v), u.max(v)]));
            }
        }
        // 平面での辺の交差、辺上の未分割な入力点を禁止する。
        for &[a, b] in &dt.edges {
            for &p in &unique {
                if p != a && p != b && orient(points[a], points[b], points[p]) == 0 {
                    assert!(!(0..2).all(|axis| {
                        (points[a][axis].min(points[b][axis])
                            ..=points[a][axis].max(points[b][axis]))
                            .contains(&points[p][axis])
                    }));
                }
            }
            for &[c, d] in &dt.edges {
                let opposite = |x: i64, y: i64| (x < 0 && y > 0) || (x > 0 && y < 0);
                assert!(
                    !(opposite(
                        orient(points[a], points[b], points[c]),
                        orient(points[a], points[b], points[d])
                    ) && opposite(
                        orient(points[c], points[d], points[a]),
                        orient(points[c], points[d], points[b])
                    ))
                );
            }
        }
        assert_eq!(
            dt.triangles.len() + unique.len(),
            dt.edges.len() + usize::from(!unique.is_empty())
        );
        let duplicate_edges = dt
            .representatives
            .iter()
            .enumerate()
            .filter(|&(i, &r)| i != r)
            .map(|(i, &r)| [i, r]);
        let cost = mst_weights(points, dt.edges.iter().copied().chain(duplicate_edges));
        let complete = (0..points.len()).flat_map(|i| (i + 1..points.len()).map(move |j| [i, j]));
        assert_eq!(cost, mst_weights(points, complete));
    }

    #[test]
    fn degeneracies_and_coordinate_limits() {
        for points in [
            vec![],
            vec![[0, 0]],
            vec![[0, 0]; 5],
            vec![[3, 4], [-1, 2]],
            vec![[0, 0], [1, 0], [2, 0]],
            (0..12).map(|i| [i, i * 2]).collect(),
            (0..12).map(|i| [0, i]).collect(),
            vec![[0, 0], [4, 0], [0, 3], [0, 0]],
            vec![
                [5, 0],
                [4, 3],
                [3, 4],
                [0, 5],
                [-3, 4],
                [-4, 3],
                [-5, 0],
                [-4, -3],
                [-3, -4],
                [0, -5],
                [3, -4],
                [4, -3],
            ],
            (-2..=2)
                .flat_map(|x| (-2..=2).map(move |y| [x, y]))
                .collect(),
            vec![
                [-20000, -20000],
                [20000, -20000],
                [20000, 20000],
                [-20000, 20000],
                [0, 0],
                [19999, -19999],
            ],
        ] {
            check(&points);
        }
    }
    #[test]
    fn randomized_triangulations_and_predicates() {
        let mut random = 42u64;
        for seed in 0..120 {
            let range = match seed % 3 {
                0 => 5,
                1 => 20000,
                _ => DELAUNAY_COORD_LIMIT as u64,
            };
            let points: Vec<_> = (0..(seed % 25 + 2))
                .map(|_| {
                    random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                    [
                        (random % (2 * range + 1)) as i32 - range as i32,
                        ((random >> 32) % (2 * range + 1)) as i32 - range as i32,
                    ]
                })
                .collect();
            check(&points);
            for window in points.windows(4) {
                assert_eq!(
                    in_circle(window[0], window[1], window[2], window[3]),
                    determinant(window[0], window[1], window[2], window[3]) > 0
                );
            }
        }
    }
    #[test]
    fn full_coordinate_range() {
        let b = DELAUNAY_COORD_LIMIT;
        check(&[[-b, -b], [b, -b], [b, b], [-b, b], [0, 0], [b - 1, -b + 1]]);
        let corners = [[-b, -b], [b, -b], [b, b], [-b, b], [0, 0]];
        for &a in &corners {
            for &b in &corners {
                for &c in &corners {
                    for &p in &corners {
                        assert_eq!(in_circle(a, b, c, p), determinant(a, b, c, p) > 0);
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_out_of_range_even_for_single_point() {
        for p in [
            [DELAUNAY_COORD_LIMIT + 1, 0],
            [0, -DELAUNAY_COORD_LIMIT - 1],
            [i32::MIN, 0],
            [0, i32::MAX],
        ] {
            assert!(std::panic::catch_unwind(|| Delaunay::new(&[p])).is_err());
        }
    }
    #[test]
    fn larger_grid_has_expected_face_count() {
        let n = 45;
        let points: Vec<_> = (0..n).flat_map(|x| (0..n).map(move |y| [x, y])).collect();
        let dt = Delaunay::new(&points);
        assert_eq!(dt.triangles.len(), (2 * (n - 1) * (n - 1)) as usize);
        assert_eq!(dt.edges.len(), dt.triangles.len() + points.len() - 1);
    }
}
