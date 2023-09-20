use std::slice::Iter;

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

#[cfg(test)]
mod test {
    use super::{FastClearArray, IndexSet};
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
}
