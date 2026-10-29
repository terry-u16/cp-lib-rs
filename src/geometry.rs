//! i32整数座標を使う2次元幾何。
mod kd_tree;
pub use kd_tree::KdTree;

/// i32座標2点のユークリッド距離の二乗。i32全域ではu64に収まらない。
fn squared_distance(a: [i32; 2], b: [i32; 2]) -> u128 {
    let dx = i128::from(a[0]) - i128::from(b[0]);
    let dy = i128::from(a[1]) - i128::from(b[1]);
    (dx * dx + dy * dy) as u128
}
