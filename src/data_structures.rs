use ac_library::Monoid;
use itertools::Itertools;
use rand::prelude::*;
use rand::rng;
use std::ops::{Index, IndexMut};
use std::{
    ops::{Bound, RangeBounds},
    slice::Iter,
};

/// [0, n) の整数の集合を管理する定数倍が軽いデータ構造
///
/// https://topcoder-tomerun.hatenablog.jp/entry/2021/06/12/134643
#[derive(Debug, Clone)]
pub struct IndexSet {
    values: Vec<usize>,
    positions: Vec<Option<usize>>,
}

impl IndexSet {
    pub fn new(n: usize) -> Self {
        Self {
            values: vec![],
            positions: vec![None; n],
        }
    }

    pub fn add(&mut self, value: usize) {
        let pos = &mut self.positions[value];

        if pos.is_none() {
            *pos = Some(self.values.len());
            self.values.push(value);
        }
    }

    pub fn remove(&mut self, value: usize) {
        if let Some(index) = self.positions[value] {
            let last = *self.values.last().unwrap();
            self.values[index] = last;
            self.values.pop();
            self.positions[last] = Some(index);
            self.positions[value] = None;
        }
    }

    pub fn contains(&self, value: usize) -> bool {
        self.positions[value].is_some()
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn iter(&'_ self) -> Iter<'_, usize> {
        self.values.iter()
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.values
    }
}

/// BFSを繰り返すときに訪問済みかを記録する配列を毎回初期化しなくて良くするアレ
///
/// https://topcoder-tomerun.hatenablog.jp/entry/2022/11/06/145156
#[derive(Debug, Clone)]
pub struct FastClearArray {
    values: Vec<u64>,
    generation: u64,
}

impl FastClearArray {
    pub fn new(len: usize) -> Self {
        Self {
            values: vec![0; len],
            generation: 1,
        }
    }

    pub fn clear(&mut self) {
        self.generation += 1;
    }

    pub fn set_true(&mut self, index: usize) {
        self.values[index] = self.generation;
    }

    pub fn get(&self, index: usize) -> bool {
        self.values[index] == self.generation
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.values.len()
    }
}

/// Disjoint Sparse Table
///
/// モノイドに対して、区間クエリを O(1) で処理するデータ構造
///
/// - 初期化: O(N log N)
/// - クエリ: O(1)
///
/// # Examples
///
/// ```
/// use ac_library::Additive;
/// use cp_lib_rs::data_structures::DisjointSparseTable;
///
/// let v = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
/// let dst = DisjointSparseTable::<Additive<_>>::new(v);
///
/// assert_eq!(dst.prod(0..3), 8);
/// ```
#[derive(Debug, Clone)]
pub struct DisjointSparseTable<M: Monoid> {
    n: usize,
    /// 元配列（1要素区間クエリ用）
    original: Vec<M::S>,
    /// 各レベルのテーブル data[level][i]
    data: Vec<Vec<M::S>>,
}

impl<M: Monoid> DisjointSparseTable<M> {
    pub fn new(data: Vec<M::S>) -> Self {
        let mut original = data;
        let n = original.len();

        // 長さ0は空テーブルにしておく
        if n == 0 {
            return Self {
                n,
                original,
                data: Vec::new(),
            };
        }

        // 2 の冪に丸める（padding は identity）
        let ceil_n = n.next_power_of_two();
        original.resize(ceil_n, M::identity());

        let mut data: Vec<Vec<M::S>> = Vec::new();

        let mut double_len = 2_usize; // ブロック長 (2^1, 2^2, ...)
        let mut len = 1_usize; // half = double_len / 2

        while double_len <= ceil_n {
            let mut level = vec![M::identity(); ceil_n];

            // center はブロックの真ん中。
            // [center - len, center) が左半分
            // [center, center + len) が右半分
            for center in (len..ceil_n).step_by(double_len) {
                // 左側: i = center - 1 から center - len まで逆順で累積
                level[center - 1] = original[center - 1].clone();
                for i in (center - len..center - 1).rev() {
                    level[i] = M::binary_operation(&original[i], &level[i + 1]);
                }

                // 右側: i = center から center + len - 1 まで順方向で累積
                level[center] = original[center].clone();
                for i in center + 1..(center + len).min(ceil_n) {
                    // 非可換モノイドを想定してマージ順に注意する
                    level[i] = M::binary_operation(&level[i - 1], &original[i]);
                }
            }

            data.push(level);
            double_len <<= 1;
            len <<= 1;
        }

        Self { n, original, data }
    }

    pub fn prod(&self, range: impl RangeBounds<usize>) -> M::S {
        let (l, mut r) = as_half_open_range(range, self.n);
        assert!(l <= r && r <= self.n);

        let len = r - l;
        if len == 0 {
            return M::identity();
        }
        if len == 1 {
            // 1要素区間は元配列から
            return self.original[l].clone();
        }

        // [l, r) → [l, r-1] の閉区間に変換
        r -= 1;

        // l と r の MSB が初めて違うビットの位置をレベルとして使う
        let k = (l ^ r).ilog2() as usize;
        let level = &self.data[k];

        M::binary_operation(&level[l], &level[r])
    }
}

/// 与えられた RangeBounds を [l, r) の半開区間に正規化する
fn as_half_open_range(range: impl RangeBounds<usize>, n: usize) -> (usize, usize) {
    let l = match range.start_bound() {
        Bound::Included(&l) => l,
        Bound::Excluded(&l) => l + 1,
        Bound::Unbounded => 0,
    };

    let r = match range.end_bound() {
        Bound::Included(&r) => r + 1,
        Bound::Excluded(&r) => r,
        Bound::Unbounded => n,
    };

    (l, r)
}
/// 座標圧縮を行う構造体
///
/// # Examples
///
/// ```
/// use cp_lib_rs::data_structures::Compressor;
///
/// let original_values = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
/// let compressor = Compressor::new(original_values.clone());
/// let expected = vec![2, 0, 3, 0, 4, 6, 1, 5, 4, 2];
///
/// assert_eq!(compressor.len(), 10);
/// assert_eq!(compressor.unique_len(), 7);
/// assert_eq!(compressor.raw(0), &3);
/// assert_eq!(compressor[0], 2);
/// assert_eq!(compressor.originals(), &original_values);
/// assert_eq!(compressor.as_slice(), expected);
/// ```
#[derive(Debug, Clone)]
pub struct Compressor<T: Ord> {
    values: Vec<T>,
    indices: Vec<usize>,
    unique_len: usize,
}

impl<T: Ord> Compressor<T> {
    pub fn new(values: impl IntoIterator<Item = T>) -> Self {
        let values = values.into_iter().collect_vec();
        let mut sorted = values.iter().collect_vec();
        sorted.sort_unstable();
        sorted.dedup();

        let indices = values
            .iter()
            .map(|v| sorted.binary_search(&v).unwrap())
            .collect_vec();

        let unique_len = sorted.len();

        Self {
            values,
            indices,
            unique_len,
        }
    }

    /// 圧縮前の値の数を取得する
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// 元の配列を取得する
    pub fn originals(&self) -> &[T] {
        &self.values
    }

    /// 圧縮後の値の数を取得する
    pub fn unique_len(&self) -> usize {
        self.unique_len
    }

    /// 元の配列におけるA[index]を取得する
    pub fn raw(&self, index: usize) -> &T {
        &self.values[index]
    }

    /// 圧縮後の配列におけるA[index]を取得する
    pub fn as_slice(&self) -> &[usize] {
        &self.indices
    }
}

impl<T: Ord> Index<usize> for Compressor<T> {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.indices[index]
    }
}

/// ウェーブレット行列
#[derive(Clone)]
pub struct WaveletMatrix<T> {
    n: usize,
    /// 元配列のデータ
    raw_data: Vec<T>,
    /// number of bit levels (e.g., up to 64 for u64)
    max_log: usize,
    /// per level bitmap of '1's（高ビット→低ビットの順に格納）
    bitmaps: Vec<BitRank>,
    /// per level, number of zeros (split point)（インデックスはビット位置 0..max_log-1）
    mids: Vec<usize>,
    /// 末端（全ビット処理後＝値順）での元インデックス。列挙時に直接参照する。
    leaf_ids: Vec<usize>,
}

impl<T: Clone + Into<u64> + PartialOrd> WaveletMatrix<T> {
    /// Build from data. O(Nlogσ)
    ///
    /// Wavelet Matrixの計算量はデータの最大ビット長に依存する。
    /// 計算量を削減するため、あらかじめ座標圧縮しておくことを推奨。
    pub fn new<I>(data: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let raw_data = data.into_iter().collect_vec();
        let data: Vec<u64> = raw_data.iter().cloned().map(|x| x.into()).collect_vec();
        let n = data.len();
        let maxv = data.iter().copied().max().unwrap_or(0);
        let computed_log = 64usize.saturating_sub(maxv.leading_zeros() as usize).max(1);
        let max_log = computed_log;
        assert!(max_log < 64, "WaveletMatrix supports up to 63 bits");

        let mut bitmaps = Vec::with_capacity(max_log);
        let mut mids = vec![0usize; max_log];

        let mut cur = data;

        // 列挙のため、元インデックスを持ち回る
        let mut ids: Vec<usize> = (0..n).collect();

        // process from high bit to low bit
        for level in (0..max_log).rev() {
            let mut bits = vec![false; n];
            for (i, &v) in cur.iter().enumerate() {
                bits[i] = ((v >> level) & 1) == 1;
            }
            let br = BitRank::from_bools(&bits);

            // stable partition by bit (0 then 1)
            let mut zeros = Vec::with_capacity(n);
            let mut ones = Vec::with_capacity(n);

            // ids も同様に安定分割
            let mut z_ids = Vec::with_capacity(n);
            let mut o_ids = Vec::with_capacity(n);

            for (i, &v) in cur.iter().enumerate() {
                if !bits[i] {
                    zeros.push(v);
                    z_ids.push(ids[i]);
                } else {
                    ones.push(v);
                    o_ids.push(ids[i]);
                }
            }

            let mid = zeros.len();
            mids[level] = mid;

            zeros.extend(ones);
            z_ids.extend(o_ids);
            cur = zeros;
            ids = z_ids;

            bitmaps.push(br);
        }

        // bitmaps were pushed from high->low level; keep that order (same indexing).
        Self {
            n,
            raw_data,
            max_log,
            bitmaps,
            mids,
            leaf_ids: ids,
        }
    }

    /// Access: return value at index idx. O(1)
    ///
    /// self\[idx\] と同じ。
    ///
    /// # Examples
    ///
    /// ```
    /// use cp_lib_rs::data_structures::WaveletMatrix;
    ///
    /// let a = vec![5u64, 1, 7, 3, 3, 9, 0, 6];
    /// let wm = WaveletMatrix::new(a.clone());
    ///
    /// for i in 0..a.len() {
    ///     assert_eq!(*wm.access(i), a[i]);
    /// }
    /// ```
    pub fn access(&self, index: usize) -> &T {
        &self[index]
    }

    /// Count occurrences of 'value' in [l, r). O(logσ)
    ///
    /// # Examples
    ///
    /// ```
    /// use cp_lib_rs::data_structures::WaveletMatrix;
    ///
    /// let a = vec![5u64, 1, 7, 3, 3, 9, 0, 6];
    /// let wm = WaveletMatrix::new(a.clone());
    ///
    /// assert_eq!(wm.rank(.., 3), 2);
    /// assert_eq!(wm.rank(2..6, 3), 2);
    /// assert_eq!(wm.rank(.., 10), 0);
    /// ```
    pub fn rank(&self, range: impl RangeBounds<usize>, value: T) -> usize {
        let value: u64 = value.into();
        let (mut l, mut r) = self.bounds_to_lr(range);
        assert!(l <= r && r <= self.n);

        for level in (0..self.max_log).rev() {
            let br = &self.bitmaps[self.max_log - 1 - level];
            let bit = ((value >> level) & 1) != 0;
            if bit {
                let l1 = br.rank1(l);
                let r1 = br.rank1(r);
                l = self.mids[level] + l1;
                r = self.mids[level] + r1;
            } else {
                let l1 = br.rank1(l);
                let r1 = br.rank1(r);
                l -= l1;
                r -= r1;
            }
        }

        r - l
    }

    /// k-th smallest index in [l, r), 0-indexed k. O(logσ)
    ///
    /// 同値が複数ある場合のタイブレークは **元配列での相対順**（安定）
    ///
    /// # Examples
    ///
    /// ```
    /// use cp_lib_rs::data_structures::WaveletMatrix;
    ///
    /// let a = vec![5u64, 1, 7, 3, 3, 9, 0, 6];
    /// let wm = WaveletMatrix::new(a.clone());
    ///
    /// // 全体: 値でソートしたときの元インデックスは [6, 1, 3, 4, 0, 7, 2, 5]
    /// assert_eq!(wm.kth(.., 0), 6); // 値=0 の位置
    /// assert_eq!(wm.kth(.., 1), 1); // 値=1 の位置
    /// assert_eq!(wm.kth(.., 2), 3); // 値=3（1個目）の位置
    /// assert_eq!(wm.kth(.., 3), 4); // 値=3（2個目）の位置
    /// assert_eq!(wm.kth(.., 4), 0); // 値=5 の位置
    /// assert_eq!(wm.kth(.., 5), 7); // 値=6 の位置
    /// assert_eq!(wm.kth(.., 6), 2); // 値=7 の位置
    /// assert_eq!(wm.kth(.., 7), 5); // 値=9 の位置
    ///
    /// // 部分区間 [2,7) の値: [7, 3, 3, 9, 0]
    /// // 値で並べると [0(6), 3(3), 3(4), 7(2), 9(5)] → インデックス [6, 3, 4, 2, 5]
    /// assert_eq!(wm.kth(2..7, 0), 6);
    /// assert_eq!(wm.kth(2..7, 1), 3);
    /// assert_eq!(wm.kth(2..7, 2), 4);
    /// assert_eq!(wm.kth(2..7, 3), 2);
    /// assert_eq!(wm.kth(2..7, 4), 5);
    /// ```
    pub fn kth(&self, range: impl RangeBounds<usize>, mut k: usize) -> usize {
        let (mut l, mut r) = self.bounds_to_lr(range);
        assert!(k < r - l);

        // 値そのものは不要。葉配列位置 l..r を追跡し、残った k をオフセットとして使う。
        for level in (0..self.max_log).rev() {
            let br = &self.bitmaps[self.max_log - 1 - level];
            let l1 = br.rank1(l);
            let r1 = br.rank1(r);
            let zeros = (r - l) - (r1 - l1);

            if k < zeros {
                // 0 側へ
                l -= l1;
                r -= r1;
            } else {
                // 1 側へ
                k -= zeros;
                l = self.mids[level] + l1;
                r = self.mids[level] + r1;
            }
        }

        // 葉配列（値順）での位置は l + k → 元インデックスは leaf_ids[その位置]
        self.leaf_ids[l + k]
    }

    /// number of x in [l, r) with lower <= x < upper. O(logσ)
    ///
    /// # Examples
    ///
    /// ```
    /// use cp_lib_rs::data_structures::WaveletMatrix;
    ///
    /// let a = vec![5u64, 1, 7, 3, 3, 9, 0, 6];
    /// let wm = WaveletMatrix::new(a.clone());
    ///
    ///
    /// // range_freq: in [0,8) count in [3,7)  => [3, 3, 5, 6] => 4
    /// assert_eq!(wm.range_freq(0..8, 3u64, 7u64), 4);
    ///
    /// // [2,7) = [7, 3, 3, 9, 0] sorted => [0, 3, 3, 7, 9]
    /// assert_eq!(wm.range_freq(2..7, 3u64, 8u64), 3);
    /// ```
    pub fn range_freq(&self, range: impl RangeBounds<usize> + Clone, lower: T, upper: T) -> usize {
        if lower >= upper {
            return 0;
        }

        self.freq_lt(range.clone(), upper) - self.freq_lt(range, lower)
    }

    /// count x in [l, r) with x < upper
    fn freq_lt(&self, range: impl RangeBounds<usize>, upper: T) -> usize {
        let upper = upper.into();
        let (mut l, mut r) = self.bounds_to_lr(range);

        if l == r {
            return 0;
        }

        let mut cnt = 0usize;

        for level in (0..self.max_log).rev() {
            let br = &self.bitmaps[self.max_log - 1 - level];
            let l1 = br.rank1(l);
            let r1 = br.rank1(r);
            let zeros = (r - l) - (r1 - l1);
            let bit = ((upper >> level) & 1) != 0;

            if bit {
                // all zeros go in (they are < upper at this bit)
                cnt += zeros;
                // proceed to ones range
                l = self.mids[level] + l1;
                r = self.mids[level] + r1;
            } else {
                // stay in zeros
                l -= l1;
                r -= r1;
            }
        }

        cnt
    }

    /// 区間 `range` において、値が `[lower, upper)` の要素の **元インデックス**を
    /// **ヒープ確保なし**で列挙し、コールバック関数を呼び出します。
    ///
    /// 発見順は値順ブロック寄り・安定ではありません。
    /// `f` は該当インデックスごとに1回呼ばれます。
    ///
    /// # Examples
    ///
    /// ```
    /// use cp_lib_rs::data_structures::WaveletMatrix;
    ///
    /// let a = vec![5u64, 1, 7, 3, 3, 9, 0, 6];
    /// let wm = WaveletMatrix::new(a.clone());
    ///
    /// // [0,8) ∩ 値 [3,7) に入る元インデックスを列挙
    /// let mut result = Vec::new();
    /// wm.range_report_each(0..8, 3u64, 7u64, |idx| result.push(idx));
    ///
    /// result.sort_unstable();
    /// assert_eq!(result, vec![0, 3, 4, 7])
    /// ```
    pub fn range_report_each(
        &self,
        range: impl RangeBounds<usize>,
        lower: T,
        upper: T,
        mut f: impl FnMut(usize),
    ) {
        let lower = lower.into();
        let upper = upper.into();
        if lower >= upper {
            return;
        }

        let (l, r) = self.bounds_to_lr(range);
        if l == r {
            return;
        }

        // ルート（高さ = max_log, 値プレフィクス base=0）から開始。
        self.report_between_rec(l, r, self.max_log, 0u64, lower, upper, &mut f);
    }

    // ノード（高さ h, 値レンジ [base, base+2^h)）と、現在の添字範囲 [l,r) について、
    // 区間 [lower, upper) に入る要素を列挙。完全内包なら子へ一気に降りて leaf_ids を吐く。
    #[allow(clippy::too_many_arguments)]
    fn report_between_rec<F: FnMut(usize)>(
        &self,
        l: usize,
        r: usize,
        h: usize,
        base: u64,
        lower: u64,
        upper: u64,
        f: &mut F,
    ) {
        if l >= r {
            return;
        }

        // 値レンジの完全外/完全内判定
        let lo = base;
        let len = 1u64 << h;
        let hi = lo + len;

        if hi <= lower || upper <= lo {
            // 完全に外れる
            return;
        }

        if lower <= lo && hi <= upper {
            // 完全に中に入る → 葉へ降りて一括列挙
            self.emit_all_rec(l, r, h, f);
            return;
        }

        // 部分的に重なる → 子に分割して再帰
        if h == 0 {
            // 1点値 (= base) のはず。ここに来たなら lower <= base < upper。
            // 現在の [l,r) は最終レベルの連続区間になっているので、そのまま吐く。
            for &idx in &self.leaf_ids[l..r] {
                f(idx);
            }
            return;
        }

        let bi = self.max_log - h; // bitmap index（高ビット→低ビットで 0..）
        let bitpos = h - 1; // いま見るビット位置（0-based, LSB側が0）
        let br = &self.bitmaps[bi];
        let l1 = br.rank1(l);
        let r1 = br.rank1(r);

        let zl = l - l1;
        let zr = r - r1; // 0 側
        let ol = self.mids[bitpos] + l1;
        let or_ = self.mids[bitpos] + r1; // 1 側

        // 0 子: base そのまま
        self.report_between_rec(zl, zr, h - 1, base, lower, upper, f);
        // 1 子: base に (1<<bitpos) を立てる
        self.report_between_rec(ol, or_, h - 1, base | (1u64 << bitpos), lower, upper, f);
    }

    // 「このノード配下のすべて」を列挙：葉に降りて leaf_ids のスライスをそのまま吐く
    fn emit_all_rec<F: FnMut(usize)>(&self, l: usize, r: usize, h: usize, f: &mut F) {
        if l >= r {
            return;
        }

        if h == 0 {
            for &idx in &self.leaf_ids[l..r] {
                f(idx);
            }

            return;
        }

        let bi = self.max_log - h;
        let bitpos = h - 1;
        let br = &self.bitmaps[bi];
        let l1 = br.rank1(l);
        let r1 = br.rank1(r);

        let zl = l - l1;
        let zr = r - r1;
        let ol = self.mids[bitpos] + l1;
        let or_ = self.mids[bitpos] + r1;

        // 0→1 の順で降りる（順序は用途に応じて変更可）
        self.emit_all_rec(zl, zr, h - 1, f);
        self.emit_all_rec(ol, or_, h - 1, f);
    }

    #[inline]
    fn bounds_to_lr<R: RangeBounds<usize>>(&self, range: R) -> (usize, usize) {
        use Bound::*;

        let l = match range.start_bound() {
            Unbounded => 0,
            Included(&x) => x,
            Excluded(&x) => x.saturating_add(1),
        };
        let r = match range.end_bound() {
            Unbounded => self.n,
            Included(&x) => x.saturating_add(1),
            Excluded(&x) => x,
        };

        assert!(l <= r, "range start must be <= end");
        assert!(r <= self.n, "range end must be <= len");
        (l, r)
    }
}

impl<T> Index<usize> for WaveletMatrix<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.raw_data[index]
    }
}

#[derive(Clone)]
struct BitRank {
    n: usize,
    words: Vec<u64>,
    // prefix_pop[i] = popcount of words[0..i)
    prefix_pop: Vec<u32>,
}

impl BitRank {
    fn from_bools(bits: &[bool]) -> Self {
        let n = bits.len();
        let w = (n + 63) >> 6;
        let mut words = vec![0u64; w];

        for (i, &b) in bits.iter().enumerate() {
            if b {
                words[i >> 6] |= 1u64 << (i & 63);
            }
        }

        let mut prefix_pop = Vec::with_capacity(w + 1);
        prefix_pop.push(0);
        let mut acc: u32 = 0;

        for &x in &words {
            acc += x.count_ones();
            prefix_pop.push(acc);
        }

        Self {
            n,
            words,
            prefix_pop,
        }
    }

    #[inline]
    #[allow(dead_code)]
    fn get(&self, i: usize) -> bool {
        debug_assert!(i < self.n);
        ((self.words[i >> 6] >> (i & 63)) & 1) != 0
    }

    /// rank1(pos): number of 1s in [0, pos)
    #[inline]
    fn rank1(&self, pos: usize) -> usize {
        let pos = pos.min(self.n);
        let w = pos >> 6;
        let m = pos & 63;
        let mut sum = self.prefix_pop[w] as usize;

        if m != 0 {
            // m ∈ [1, 63] が保証されるので 64 シフトの心配はない
            let mask = u64::MAX >> (64 - m);
            sum += (self.words[w] & mask).count_ones() as usize;
        }

        sum
    }
}

/// ローリングハッシュ
///
/// - 初期化: O(N)
/// - ハッシュ値の取得: O(1)
///
///
/// # Examples
///
/// ```
/// use cp_lib_rs::data_structures::RollingHash;
///
/// let s = "mississippi";
/// let rolling_hash = RollingHash::new(s.chars());
///
/// assert!(rolling_hash.hash(2..5) == rolling_hash.hash(5..8));
/// assert!(rolling_hash.hash(0..3) != rolling_hash.hash(1..4));
/// ```
#[derive(Debug, Clone)]
pub struct RollingHash {
    len: usize,
    hashes: Vec<u64>,
    pow: Vec<u64>,
}

impl RollingHash {
    const MOD: u64 = (1 << 61) - 1;

    pub fn new(values: impl IntoIterator<Item = impl Into<u64>>) -> Self {
        let values = values.into_iter();
        let base = rng().random_range(1 << 60..Self::MOD);
        let mut pow = vec![1];
        let mut hashes = vec![0];
        let len = values.size_hint().0;
        pow.reserve(len);
        hashes.reserve(len);

        for (i, v) in values.enumerate() {
            pow.push(Self::mul_mod(pow[i], base));

            // 0が来ると"0"と"00"の区別が付かなくて困るので+1しておく
            let mut hash = Self::mul_mod(hashes[i], base) as u64 + v.into() + 1;
            if hash >= Self::MOD {
                hash -= Self::MOD;
            }
            hashes.push(hash);
        }

        let len = hashes.len() - 1;
        Self { pow, hashes, len }
    }

    pub fn hash(&self, range: impl RangeBounds<usize>) -> u64 {
        let (l, r) = as_half_open_range(range, self.len);
        let hash = Self::MOD + self.hashes[r] - Self::mul_mod(self.hashes[l], self.pow[r - l]);

        if hash >= Self::MOD {
            hash - Self::MOD
        } else {
            hash
        }
    }

    fn mul_mod(a: u64, b: u64) -> u64 {
        // https://qiita.com/keymoon/items/11fac5627672a6d6a9f6#%E4%BD%99%E8%AB%87128bit%E7%92%B0%E5%A2%83%E3%81%AB%E3%81%8A%E3%81%91%E3%82%8B%E5%AE%9F%E8%A3%85%E3%81%AE%E7%B0%A1%E6%98%93%E5%8C%96%E3%81%AE%E8%A9%B1
        let t = a as u128 * b as u128;
        let t = (t >> 61) as u64 + ((t as u64) & Self::MOD);

        if t >= Self::MOD { t - Self::MOD } else { t }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Queue<T> {
    data: Vec<T>,
    pos: usize,
}

impl<T> Queue<T> {
    pub fn reserve(&mut self, n: usize) {
        if n > self.data.len() {
            self.data.reserve(n - self.data.len());
        }
    }

    pub fn size(&self) -> usize {
        self.data.len() - self.pos
    }

    pub fn empty(&self) -> bool {
        self.pos == self.data.len()
    }

    pub fn push(&mut self, t: T) {
        self.data.push(t);
    }

    pub fn front(&self) -> Option<&T> {
        if self.pos < self.data.len() {
            Some(&self.data[self.pos])
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.pos = 0;
    }

    pub fn pop(&mut self) -> Option<&T> {
        if self.pos < self.data.len() {
            self.pos += 1;
            Some(&self.data[self.pos - 1])
        } else {
            None
        }
    }
}

/// Implicit Treap
///
/// 配列を平衡二分木として保持し、ランダムアクセスや区間操作を O(log N) で扱う。
///
/// # Examples
///
/// ```
/// use cp_lib_rs::data_structures::ImplicitTreap;
///
/// let mut treap = ImplicitTreap::from_iter([1, 2, 3, 4]);
/// treap.insert(2, 10);
/// treap.reverse(1..4);
/// treap.rotate_left(1..5, 2);
///
/// assert_eq!(treap.into_vec(), vec![1, 2, 4, 3, 10]);
/// ```
#[derive(Debug, Clone)]
pub struct ImplicitTreap<T> {
    root: Option<Box<ImplicitTreapNode<T>>>,
    random_state: u64,
}

#[derive(Debug, Clone)]
struct ImplicitTreapNode<T> {
    value: T,
    priority: u64,
    /// 部分木サイズ。implicit index の計算に使う。
    len: usize,
    /// この部分木全体を反転する遅延フラグ。
    rev: bool,
    left: Option<Box<ImplicitTreapNode<T>>>,
    right: Option<Box<ImplicitTreapNode<T>>>,
}

impl<T> Default for ImplicitTreap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ImplicitTreap<T> {
    /// 空の treap を作る。
    pub fn new() -> Self {
        Self {
            root: None,
            random_state: rng().random(),
        }
    }

    /// 要素数を返す。
    pub fn len(&self) -> usize {
        ImplicitTreapNode::len(&self.root)
    }

    /// 空なら true を返す。
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// `index` 番目の要素への参照を返す。
    pub fn get(&self, index: usize) -> Option<&T> {
        self.root.as_ref()?.get(index, false)
    }

    /// `index` 番目の要素への可変参照を返す。
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len() {
            return None;
        }

        self.root.as_mut()?.get_mut(index)
    }

    /// 末尾に要素を追加する。
    pub fn push_back(&mut self, value: T) {
        let node = Some(Box::new(ImplicitTreapNode::new(
            value,
            self.next_priority(),
        )));
        self.root = ImplicitTreapNode::merge(self.root.take(), node);
    }

    /// 先頭に要素を追加する。
    pub fn push_front(&mut self, value: T) {
        let node = Some(Box::new(ImplicitTreapNode::new(
            value,
            self.next_priority(),
        )));
        self.root = ImplicitTreapNode::merge(node, self.root.take());
    }

    /// 末尾の要素を削除して返す。
    pub fn pop_back(&mut self) -> Option<T> {
        self.remove(self.len().checked_sub(1)?)
    }

    /// 先頭の要素を削除して返す。
    pub fn pop_front(&mut self) -> Option<T> {
        self.remove(0)
    }

    /// `index` の位置に要素を挿入する。
    pub fn insert(&mut self, index: usize, value: T) {
        assert!(index <= self.len());

        // [0, index), [index, n) に分けて 1 ノードだけ挟む。
        let (left, right) = ImplicitTreapNode::split(self.root.take(), index);
        let middle = Some(Box::new(ImplicitTreapNode::new(
            value,
            self.next_priority(),
        )));
        self.root = ImplicitTreapNode::merge(ImplicitTreapNode::merge(left, middle), right);
    }

    /// `index` 番目の要素を削除して返す。
    pub fn remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len() {
            return None;
        }

        let (left, right) = ImplicitTreapNode::split(self.root.take(), index);
        let (middle, right) = ImplicitTreapNode::split(right, 1);
        self.root = ImplicitTreapNode::merge(left, right);
        middle.map(|node| node.value)
    }

    /// 先頭 `index` 個と残りに分割する。
    pub fn split(mut self, index: usize) -> (Self, Self) {
        assert!(index <= self.len());

        let (left, right) = ImplicitTreapNode::split(self.root.take(), index);

        (
            Self {
                root: left,
                random_state: self.random_state,
            },
            Self {
                root: right,
                // split 後の 2 本が同じ priority 列を生成しないよう右側の状態をずらす。
                // この定数自体に強い意味はなく、既知の 64-bit 定数を便宜的に使っている。
                random_state: self.random_state ^ 0x9e37_79b9_7f4a_7c15,
            },
        )
    }

    /// 左右 2 つの treap を連結する。
    pub fn merge(mut left: Self, mut right: Self) -> Self {
        Self {
            root: ImplicitTreapNode::merge(left.root.take(), right.root.take()),
            random_state: left.random_state ^ right.random_state.rotate_left(7),
        }
    }

    /// 区間 `[l, r)` を反転する。
    pub fn reverse(&mut self, range: impl RangeBounds<usize>) {
        let (l, r) = as_half_open_range(range, self.len());
        assert!(l <= r && r <= self.len());

        // 対象区間だけ切り出して遅延反転フラグを立てる。
        let (left, middle_right) = ImplicitTreapNode::split(self.root.take(), l);
        let (mut middle, right) = ImplicitTreapNode::split(middle_right, r - l);
        ImplicitTreapNode::toggle_rev_subtree(&mut middle);
        self.root = ImplicitTreapNode::merge(ImplicitTreapNode::merge(left, middle), right);
    }

    /// 区間 `[l, r)` を左に `k` 回巡回シフトする。
    pub fn rotate_left(&mut self, range: impl RangeBounds<usize>, k: usize) {
        let (l, r) = as_half_open_range(range, self.len());
        assert!(l <= r && r <= self.len());

        let (left, middle_right) = ImplicitTreapNode::split(self.root.take(), l);
        let (middle, right) = ImplicitTreapNode::split(middle_right, r - l);
        let len = ImplicitTreapNode::len(&middle);

        let middle = if len == 0 {
            middle
        } else {
            let k = k % len;
            // [a, b) を [b, a) に並べ替えて rotate を実現する。
            let (first, second) = ImplicitTreapNode::split(middle, k);
            ImplicitTreapNode::merge(second, first)
        };

        self.root = ImplicitTreapNode::merge(ImplicitTreapNode::merge(left, middle), right);
    }

    /// 区間 `[l, r)` を右に `k` 回巡回シフトする。
    pub fn rotate_right(&mut self, range: impl RangeBounds<usize>, k: usize) {
        let (l, r) = as_half_open_range(range, self.len());
        assert!(l <= r && r <= self.len());

        let len = r - l;
        if len == 0 {
            return;
        }

        self.rotate_left(l..r, len - (k % len));
    }

    /// 中身を順序付きの `Vec` として取り出す。
    pub fn into_vec(mut self) -> Vec<T> {
        let mut values = Vec::with_capacity(self.len());
        if let Some(node) = self.root.take() {
            node.into_vec(&mut values);
        }
        values
    }

    fn next_priority(&mut self) -> u64 {
        // SplitMix64。treap の priority 用に十分速くて偏りが少ない。
        // 0x9e37_79b9_7f4a_7c15 は 64-bit の黄金比由来の加算定数。
        self.random_state = self.random_state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.random_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

impl<T> Index<usize> for ImplicitTreap<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<T> IndexMut<usize> for ImplicitTreap<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

impl<T> FromIterator<T> for ImplicitTreap<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut treap = Self::new();
        for value in iter {
            treap.push_back(value);
        }
        treap
    }
}

impl<T> ImplicitTreapNode<T> {
    fn new(value: T, priority: u64) -> Self {
        Self {
            value,
            priority,
            len: 1,
            rev: false,
            left: None,
            right: None,
        }
    }

    fn len(node: &Option<Box<Self>>) -> usize {
        node.as_ref().map_or(0, |node| node.len)
    }

    fn update(&mut self) {
        self.len = 1 + Self::len(&self.left) + Self::len(&self.right);
    }

    fn toggle_rev(&mut self) {
        self.rev ^= true;
    }

    fn toggle_rev_subtree(node: &mut Option<Box<Self>>) {
        if let Some(node) = node {
            node.toggle_rev();
        }
    }

    fn push(&mut self) {
        if !self.rev {
            return;
        }

        // 反転は「左右の子を入れ替え、子にも反転フラグを伝播」で処理する。
        self.rev = false;
        std::mem::swap(&mut self.left, &mut self.right);
        Self::toggle_rev_subtree(&mut self.left);
        Self::toggle_rev_subtree(&mut self.right);
    }

    fn split(root: Option<Box<Self>>, left_len: usize) -> (Option<Box<Self>>, Option<Box<Self>>) {
        match root {
            None => (None, None),
            Some(mut node) => {
                node.push();
                let node_left_len = Self::len(&node.left);

                // 左部分木サイズを implicit index とみなして分割位置を決める。
                if left_len <= node_left_len {
                    let (left, new_left) = Self::split(node.left.take(), left_len);
                    node.left = new_left;
                    node.update();
                    (left, Some(node))
                } else {
                    let (new_right, right) =
                        Self::split(node.right.take(), left_len - node_left_len - 1);
                    node.right = new_right;
                    node.update();
                    (Some(node), right)
                }
            }
        }
    }

    fn merge(left: Option<Box<Self>>, right: Option<Box<Self>>) -> Option<Box<Self>> {
        match (left, right) {
            (None, right) => right,
            (left, None) => left,
            (Some(mut left), Some(mut right)) => {
                // heap 条件を priority で保ちながら左右を繋ぎ直す。
                if left.priority > right.priority {
                    left.push();
                    left.right = Self::merge(left.right.take(), Some(right));
                    left.update();
                    Some(left)
                } else {
                    right.push();
                    right.left = Self::merge(Some(left), right.left.take());
                    right.update();
                    Some(right)
                }
            }
        }
    }

    fn get(&self, index: usize, reversed: bool) -> Option<&T> {
        let reversed = reversed ^ self.rev;
        // 祖先から見た反転状態だけを引き回せば、不変参照のまま辿れる。
        let left_len = if reversed {
            Self::len(&self.right)
        } else {
            Self::len(&self.left)
        };

        if index < left_len {
            if reversed {
                self.right.as_ref()?.get(index, reversed)
            } else {
                self.left.as_ref()?.get(index, reversed)
            }
        } else if index == left_len {
            Some(&self.value)
        } else if reversed {
            self.left.as_ref()?.get(index - left_len - 1, reversed)
        } else {
            self.right.as_ref()?.get(index - left_len - 1, reversed)
        }
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.push();
        let left_len = Self::len(&self.left);

        if index < left_len {
            self.left.as_mut()?.get_mut(index)
        } else if index == left_len {
            Some(&mut self.value)
        } else {
            self.right.as_mut()?.get_mut(index - left_len - 1)
        }
    }

    fn into_vec(mut self: Box<Self>, values: &mut Vec<T>) {
        self.push();
        if let Some(left) = self.left.take() {
            left.into_vec(values);
        }
        values.push(self.value);
        if let Some(right) = self.right.take() {
            right.into_vec(values);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ac_library::Additive;
    use itertools::Itertools;

    #[test]
    fn index_set() {
        let mut set = IndexSet::new(10);
        set.add(1);
        set.add(5);
        set.add(2);
        assert_eq!(3, set.len());
        assert!(set.contains(1));
        assert!(!set.contains(0));
        assert_eq!(set.iter().copied().sorted().collect_vec(), vec![1, 2, 5]);
        assert_eq!(
            set.as_slice().iter().copied().sorted().collect_vec(),
            vec![1, 2, 5]
        );

        set.add(1);
        assert_eq!(3, set.len());
        assert!(set.contains(1));
        assert_eq!(set.iter().copied().sorted().collect_vec(), vec![1, 2, 5]);

        set.remove(5);
        set.remove(2);
        assert_eq!(1, set.len());
        assert!(set.contains(1));
        assert!(!set.contains(5));
        assert!(!set.contains(2));
        assert_eq!(set.iter().copied().sorted().collect_vec(), vec![1]);

        set.remove(1);
        set.remove(2);
        assert_eq!(0, set.len());
        assert!(!set.contains(1));
        assert_eq!(set.iter().copied().sorted().collect_vec(), vec![]);
    }

    #[test]
    fn fast_clear_array() {
        let mut array = FastClearArray::new(5);
        assert!(!array.get(0));

        array.set_true(0);
        assert!(array.get(0));
        assert!(!array.get(1));

        array.clear();
        assert!(!array.get(0));

        array.set_true(0);
        assert!(array.get(0));
    }

    #[test]
    fn dst_add() {
        let v = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
        let n = v.len();
        let dst = DisjointSparseTable::<Additive<_>>::new(v.clone());

        for l in 0..=n {
            for r in l..=n {
                let expected = v[l..r].iter().copied().sum::<i32>();
                let actual = dst.prod(l..r);
                assert_eq!(expected, actual);
            }
        }
    }

    #[test]
    fn rolling_hash() {
        let s = "mississippi".chars().collect_vec();
        let rolling_hash = RollingHash::new(s.iter().copied());

        for l0 in 0..=s.len() {
            for r0 in l0..=s.len() {
                let hash0 = rolling_hash.hash(l0..r0);
                let s0 = s[l0..r0].iter().collect::<String>();

                for l1 in 0..=s.len() {
                    for r1 in l1..=s.len() {
                        let hash1 = rolling_hash.hash(l1..r1);
                        let s1 = s[l1..r1].iter().collect::<String>();

                        assert_eq!(hash0 == hash1, s0 == s1);
                    }
                }
            }
        }
    }

    #[test]
    fn implicit_treap_basic_operations() {
        let mut treap = ImplicitTreap::new();
        assert!(treap.is_empty());

        treap.push_back(2);
        treap.push_front(1);
        treap.push_back(4);
        treap.insert(2, 3);
        assert_eq!(treap.len(), 4);
        assert_eq!(treap[0], 1);
        assert_eq!(treap[3], 4);

        *treap.get_mut(1).unwrap() = 10;
        assert_eq!(treap.remove(1), Some(10));
        assert_eq!(treap.pop_front(), Some(1));
        assert_eq!(treap.pop_back(), Some(4));
        assert_eq!(treap.into_vec(), vec![3]);
    }

    #[test]
    fn implicit_treap_index_mut() {
        let mut treap = ImplicitTreap::from_iter([1, 2, 3]);
        treap[1] = 10;
        assert_eq!(treap.into_vec(), vec![1, 10, 3]);
    }

    #[test]
    fn implicit_treap_split_merge_reverse_rotate() {
        let mut treap = ImplicitTreap::from_iter(0..8);
        treap.reverse(2..7);
        assert_eq!(treap.clone().into_vec(), vec![0, 1, 6, 5, 4, 3, 2, 7]);

        treap.rotate_left(1..7, 2);
        assert_eq!(treap.clone().into_vec(), vec![0, 5, 4, 3, 2, 1, 6, 7]);

        treap.rotate_right(1..7, 3);
        assert_eq!(treap.clone().into_vec(), vec![0, 2, 1, 6, 5, 4, 3, 7]);

        let (left, right) = treap.split(3);
        assert_eq!(left.into_vec(), vec![0, 2, 1]);
        assert_eq!(right.clone().into_vec(), vec![6, 5, 4, 3, 7]);

        let merged = ImplicitTreap::merge(ImplicitTreap::from_iter([0, 1]), right);
        assert_eq!(merged.into_vec(), vec![0, 1, 6, 5, 4, 3, 7]);
    }

    #[test]
    fn implicit_treap_matches_vec() {
        let mut treap = ImplicitTreap::new();
        let mut vec = Vec::new();

        for i in 0..20 {
            if i % 2 == 0 {
                treap.push_back(i);
                vec.push(i);
            } else {
                treap.push_front(i);
                vec.insert(0, i);
            }
        }

        treap.insert(5, 100);
        vec.insert(5, 100);
        treap.insert(vec.len(), 200);
        vec.push(200);

        treap.reverse(3..15);
        vec[3..15].reverse();
        treap.rotate_left(2..18, 5);
        vec[2..18].rotate_left(5);
        treap.rotate_right(.., 7);
        vec.rotate_right(7);

        assert_eq!(treap.clone().into_vec(), vec);

        for i in (0..5).rev() {
            assert_eq!(treap.remove(i * 3), Some(vec.remove(i * 3)));
        }

        assert_eq!(treap.into_vec(), vec);
    }
}
