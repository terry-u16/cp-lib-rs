use std::{
    fmt::Display,
    ops::{Add, AddAssign, Index, IndexMut},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Coord {
    row: u8,
    col: u8,
}

impl Coord {
    pub const fn new(row: usize, col: usize) -> Self {
        Self {
            row: row as u8,
            col: col as u8,
        }
    }

    pub const fn row(&self) -> usize {
        self.row as usize
    }

    pub const fn col(&self) -> usize {
        self.col as usize
    }

    pub fn in_map(&self, size: usize) -> bool {
        self.row < size as u8 && self.col < size as u8
    }

    pub const fn to_index(&self, size: usize) -> CoordIndex {
        CoordIndex(self.row as usize * size + self.col as usize)
    }

    pub const fn dist(&self, other: &Self) -> usize {
        Self::dist_1d(self.row, other.row) + Self::dist_1d(self.col, other.col)
    }

    const fn dist_1d(x0: u8, x1: u8) -> usize {
        x0.abs_diff(x1) as usize
    }
}

impl Display for Coord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.row, self.col)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct CoordIndex(pub usize);

impl CoordIndex {
    pub const fn to_coord(&self, size: usize) -> Coord {
        Coord::new(self.0 / size, self.0 % size)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct CoordDiff {
    dr: i8,
    dc: i8,
}

impl CoordDiff {
    pub const fn new(dr: i32, dc: i32) -> Self {
        Self {
            dr: dr as i8,
            dc: dc as i8,
        }
    }

    pub const fn invert(&self) -> Self {
        Self {
            dr: -self.dr,
            dc: -self.dc,
        }
    }

    pub const fn dr(&self) -> i32 {
        self.dr as i32
    }

    pub const fn dc(&self) -> i32 {
        self.dc as i32
    }
}

impl Display for CoordDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.dr, self.dc)
    }
}

impl Add<CoordDiff> for Coord {
    type Output = Coord;

    fn add(self, rhs: CoordDiff) -> Self::Output {
        Coord {
            row: self.row.wrapping_add(rhs.dr as u8),
            col: self.col.wrapping_add(rhs.dc as u8),
        }
    }
}

impl AddAssign<CoordDiff> for Coord {
    fn add_assign(&mut self, rhs: CoordDiff) {
        self.row = self.row.wrapping_add(rhs.dr as u8);
        self.col = self.col.wrapping_add(rhs.dc as u8);
    }
}

pub const ADJACENTS: [CoordDiff; 4] = [
    CoordDiff::new(-1, 0),
    CoordDiff::new(0, 1),
    CoordDiff::new(1, 0),
    CoordDiff::new(0, -1),
];

pub const U: usize = 0;
pub const R: usize = 1;
pub const D: usize = 2;
pub const L: usize = 3;
pub const DIRECTIONS: [char; 4] = ['U', 'R', 'D', 'L'];

#[derive(Debug, Clone)]
pub struct Map2d<T> {
    size: usize,
    map: Vec<T>,
}

impl<T> Map2d<T> {
    pub fn new(map: Vec<T>, size: usize) -> Self {
        debug_assert!(size * size == map.len());
        Self { size, map }
    }

    pub fn from_fn(mut f: impl FnMut(Coord) -> T, size: usize) -> Self {
        let mut map = Vec::with_capacity(size * size);

        for row in 0..size {
            for col in 0..size {
                map.push(f(Coord::new(row, col)));
            }
        }

        Self { size, map }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.map.iter()
    }
}

impl<T: Default + Clone> Map2d<T> {
    pub fn with_default(size: usize) -> Self {
        let map = vec![T::default(); size * size];
        Self { size, map }
    }
}

impl<T> Index<Coord> for Map2d<T> {
    type Output = T;

    #[inline]
    fn index(&self, coordinate: Coord) -> &Self::Output {
        &self.map[coordinate.to_index(self.size).0]
    }
}

impl<T> IndexMut<Coord> for Map2d<T> {
    #[inline]
    fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
        &mut self.map[coordinate.to_index(self.size).0]
    }
}

impl<T> Index<&Coord> for Map2d<T> {
    type Output = T;

    #[inline]
    fn index(&self, coordinate: &Coord) -> &Self::Output {
        &self.map[coordinate.to_index(self.size).0]
    }
}

impl<T> IndexMut<&Coord> for Map2d<T> {
    #[inline]
    fn index_mut(&mut self, coordinate: &Coord) -> &mut Self::Output {
        &mut self.map[coordinate.to_index(self.size).0]
    }
}

impl<T> Index<usize> for Map2d<T> {
    type Output = [T];

    #[inline]
    fn index(&self, row: usize) -> &Self::Output {
        let begin = row * self.size;
        let end = begin + self.size;
        &self.map[begin..end]
    }
}

impl<T> IndexMut<usize> for Map2d<T> {
    #[inline]
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        let begin = row * self.size;
        let end = begin + self.size;
        &mut self.map[begin..end]
    }
}

impl<T> Index<CoordIndex> for Map2d<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: CoordIndex) -> &Self::Output {
        &self.map[index.0]
    }
}

impl<T> IndexMut<CoordIndex> for Map2d<T> {
    #[inline]
    fn index_mut(&mut self, index: CoordIndex) -> &mut Self::Output {
        &mut self.map[index.0]
    }
}

#[derive(Debug, Clone)]
pub struct ConstMap2d<T, const N: usize> {
    map: Vec<T>,
}

impl<T, const N: usize> ConstMap2d<T, N> {
    pub fn new(map: Vec<T>) -> Self {
        assert_eq!(map.len(), N * N);
        Self { map }
    }

    pub fn from_fn(mut f: impl FnMut(Coord) -> T) -> Self {
        let mut map = Vec::with_capacity(N * N);

        for row in 0..N {
            for col in 0..N {
                map.push(f(Coord::new(row, col)));
            }
        }

        Self { map }
    }
}

impl<T: Default + Clone, const N: usize> ConstMap2d<T, N> {
    pub fn with_default() -> Self {
        let map = vec![T::default(); N * N];
        Self { map }
    }
}

impl<T, const N: usize> Index<Coord> for ConstMap2d<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, coordinate: Coord) -> &Self::Output {
        &self.map[coordinate.to_index(N).0]
    }
}

impl<T, const N: usize> IndexMut<Coord> for ConstMap2d<T, N> {
    #[inline]
    fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
        &mut self.map[coordinate.to_index(N).0]
    }
}

impl<T, const N: usize> Index<&Coord> for ConstMap2d<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, coordinate: &Coord) -> &Self::Output {
        &self.map[coordinate.to_index(N).0]
    }
}

impl<T, const N: usize> IndexMut<&Coord> for ConstMap2d<T, N> {
    #[inline]
    fn index_mut(&mut self, coordinate: &Coord) -> &mut Self::Output {
        &mut self.map[coordinate.to_index(N).0]
    }
}

impl<T, const N: usize> Index<usize> for ConstMap2d<T, N> {
    type Output = [T];

    #[inline]
    fn index(&self, row: usize) -> &Self::Output {
        let begin = row * N;
        let end = begin + N;
        &self.map[begin..end]
    }
}

impl<T, const N: usize> IndexMut<usize> for ConstMap2d<T, N> {
    #[inline]
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        let begin = row * N;
        let end = begin + N;
        &mut self.map[begin..end]
    }
}

impl<T, const N: usize> Index<CoordIndex> for ConstMap2d<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, index: CoordIndex) -> &Self::Output {
        &self.map[index.0]
    }
}

impl<T, const N: usize> IndexMut<CoordIndex> for ConstMap2d<T, N> {
    #[inline]
    fn index_mut(&mut self, index: CoordIndex) -> &mut Self::Output {
        &mut self.map[index.0]
    }
}

#[cfg(test)]
mod test {
    use super::{ConstMap2d, Coord, CoordDiff, Map2d};

    #[test]
    fn coord_add() {
        let c = Coord::new(2, 4);
        let d = CoordDiff::new(-3, 5);
        let actual = c + d;

        let expected = Coord::new(!0, 9);
        assert_eq!(expected, actual);
    }

    #[test]
    fn coord_add_assign() {
        let mut c = Coord::new(2, 4);
        let d = CoordDiff::new(-3, 5);
        c += d;

        let expected = Coord::new(!0, 9);
        assert_eq!(expected, c);
    }

    #[test]
    fn map_new() {
        let map = Map2d::new(vec![0, 1, 2, 3], 2);
        let actual = map[Coord::new(1, 0)];
        let expected = 2;
        assert_eq!(expected, actual);
    }

    #[test]
    fn map_default() {
        let map = Map2d::with_default(2);
        let actual = map[Coord::new(1, 0)];
        let expected = 0;
        assert_eq!(expected, actual);
    }

    #[test]
    fn const_map_new() {
        let map = ConstMap2d::<_, 2>::new(vec![0, 1, 2, 3]);
        let actual = map[Coord::new(1, 0)];
        let expected = 2;
        assert_eq!(expected, actual);
    }

    #[test]
    fn const_map_default() {
        let map = ConstMap2d::<_, 2>::with_default();
        let actual = map[Coord::new(1, 0)];
        let expected = 0;
        assert_eq!(expected, actual);
    }
}
