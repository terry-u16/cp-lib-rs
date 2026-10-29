//! i32整数座標を使う2次元幾何。
mod kd_tree;
pub use kd_tree::KdTree;
mod delaunay;
pub use delaunay::Delaunay;

/// KD-treeで受け付ける各座標の絶対値の上限。
pub const KD_COORD_LIMIT: i32 = 1_000_000_000;
/// Delaunayの外接円判定をi128で正確に計算できる各座標の絶対値の上限。
pub const DELAUNAY_COORD_LIMIT: i32 = 1_000_000_000;

fn validate_point(p: [i32; 2], limit: i32) {
    assert!(
        p.into_iter().all(|x| (-limit..=limit).contains(&x)),
        "coordinates must be in -{limit}..={limit}"
    );
}

/// 検証済み座標の距離二乗。各差は最大2e9、和は最大8e18でi64に収まる。
fn squared_distance(a: [i32; 2], b: [i32; 2]) -> i64 {
    let dx = i64::from(a[0]) - i64::from(b[0]);
    let dy = i64::from(a[1]) - i64::from(b[1]);
    dx * dx + dy * dy
}
