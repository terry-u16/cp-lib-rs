use ac_library::Monoid;
use rand::prelude::*;
use rand::thread_rng;
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
