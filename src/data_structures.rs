use ac_library::Monoid;
use itertools::Itertools;
use rand::prelude::*;
use rand::thread_rng;
use std::ops::Index;
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

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn iter(&self) -> Iter<usize> {
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
    gen: u64,
}

impl FastClearArray {
    pub fn new(len: usize) -> Self {
        Self {
            values: vec![0; len],
            gen: 1,
        }
    }

    pub fn clear(&mut self) {
        self.gen += 1;
    }

    pub fn set_true(&mut self, index: usize) {
        self.values[index] = self.gen;
    }

    pub fn get(&self, index: usize) -> bool {
        self.values[index] == self.gen
    }

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
/// let dst = DisjointSparseTable::<Additive<_>>::new(&v);
///
/// assert_eq!(dst.prod(0..3), 8);
/// ```
#[derive(Debug, Clone)]
pub struct DisjointSparseTable<M: Monoid> {
    n: usize,
    data: Vec<Vec<M::S>>,
}

impl<M: Monoid> DisjointSparseTable<M> {
    pub fn new(v: &[M::S]) -> Self {
        let n = v.len();
        let ceil_n = 1 << ((n - 1).ilog2() + 1);
        let mut data = vec![];
        let mut v = v.to_vec();
        v.resize(ceil_n, M::identity());

        let mut double_len = 2;
        let mut len = double_len >> 1;

        while double_len <= ceil_n {
            let mut data_k = vec![M::identity(); ceil_n];

            for center in (len..n).step_by(double_len) {
                // 左側
                data_k[center - 1] = v[center - 1].clone();

                for i in (center - len..center - 1).rev() {
                    data_k[i] = M::binary_operation(&v[i], &data_k[i + 1]);
                }

                // 右側
                data_k[center] = v[center].clone();

                for i in center + 1..center + len {
                    data_k[i] = M::binary_operation(&v[i], &data_k[i - 1]);
                }
            }

            data.push(data_k);
            double_len <<= 1;
            len <<= 1;
        }

        Self { n, data }
    }

    pub fn prod(&self, range: impl RangeBounds<usize>) -> M::S {
        let (l, r) = as_half_open_range(range, self.n);

        assert!(l <= r && r <= self.n);

        if r - l == 0 {
            return M::identity();
        } else if r - l == 1 {
            return self.data[0][l].clone();
        }

        // 閉区間にする
        let r = r - 1;

        // MSB (Most Significant Bit) の取得
        let k = (l ^ r).ilog2() as usize;

        M::binary_operation(&self.data[k][l], &self.data[k][r])
    }
}

fn as_half_open_range(range: impl RangeBounds<usize>, n: usize) -> (usize, usize) {
    // 半開区間で受け取る
    let l = match range.start_bound() {
        Bound::Included(l) => *l,
        Bound::Excluded(l) => l + 1,
        Bound::Unbounded => 0,
    };

    let r = match range.end_bound() {
        Bound::Included(r) => r + 1,
        Bound::Excluded(r) => *r,
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
                l = l - l1;
                r = r - r1;
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
                l = l - l1;
                r = r - r1;
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
                l = l - l1;
                r = r - r1;
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
        let base = thread_rng().gen_range(1 << 60..Self::MOD);
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

        if t >= Self::MOD {
            t - Self::MOD
        } else {
            t
        }
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
        assert_eq!(array.get(0), false);

        array.set_true(0);
        assert_eq!(array.get(0), true);
        assert_eq!(array.get(1), false);

        array.clear();
        assert_eq!(array.get(0), false);

        array.set_true(0);
        assert_eq!(array.get(0), true);
    }

    #[test]
    fn dst_add() {
        let v = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
        let n = v.len();
        let dst = DisjointSparseTable::<Additive<_>>::new(&v);

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
}
