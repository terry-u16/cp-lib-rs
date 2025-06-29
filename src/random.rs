use num::{FromPrimitive, PrimInt};
use rand::Rng;
use std::ops::{Range, RangeInclusive};

/// 高速な乱数生成の拡張トレイト
pub trait RandExtension {
    fn fast_gen_range_u64x1<T: PrimInt + FromPrimitive, R: BoundedRange<T>>(
        &mut self,
        range: R,
    ) -> T;

    fn fast_gen_range_u32x1<T: PrimInt + FromPrimitive, R: BoundedRange<T>>(
        &mut self,
        range: R,
    ) -> T;

    fn fast_gen_range_u32x2<
        T0: PrimInt + FromPrimitive,
        T1: PrimInt + FromPrimitive,
        R0: BoundedRange<T0>,
        R1: BoundedRange<T1>,
    >(
        &mut self,
        range0: R0,
        range1: R1,
    ) -> (T0, T1);

    fn fast_gen_range_u16x1<T: PrimInt + FromPrimitive, R: BoundedRange<T>>(
        &mut self,
        range: R,
    ) -> T;

    fn fast_gen_range_u16x2<
        T0: PrimInt + FromPrimitive,
        T1: PrimInt + FromPrimitive,
        R0: BoundedRange<T0>,
        R1: BoundedRange<T1>,
    >(
        &mut self,
        range0: R0,
        range1: R1,
    ) -> (T0, T1);

    fn fast_gen_range_u16x3<
        T0: PrimInt + FromPrimitive,
        T1: PrimInt + FromPrimitive,
        T2: PrimInt + FromPrimitive,
        R0: BoundedRange<T0>,
        R1: BoundedRange<T1>,
        R2: BoundedRange<T2>,
    >(
        &mut self,
        range0: R0,
        range1: R1,
        range2: R2,
    ) -> (T0, T1, T2);

    fn fast_gen_range_u16x4<
        T0: PrimInt + FromPrimitive,
        T1: PrimInt + FromPrimitive,
        T2: PrimInt + FromPrimitive,
        T3: PrimInt + FromPrimitive,
        R0: BoundedRange<T0>,
        R1: BoundedRange<T1>,
        R2: BoundedRange<T2>,
        R3: BoundedRange<T3>,
    >(
        &mut self,
        range0: R0,
        range1: R1,
        range2: R2,
        range3: R3,
    ) -> (T0, T1, T2, T3);
}

impl<G: Rng> RandExtension for G {
    fn fast_gen_range_u64x1<T: PrimInt + FromPrimitive, R: BoundedRange<T>>(
        &mut self,
        range: R,
    ) -> T {
        let rand_value = self.next_u64();
        gen_range_u64(range, rand_value)
    }

    fn fast_gen_range_u32x1<T: PrimInt + FromPrimitive, R: BoundedRange<T>>(
        &mut self,
        range: R,
    ) -> T {
        let rand_value = self.next_u64();
        gen_range_u32(range, rand_value as u32)
    }

    fn fast_gen_range_u32x2<
        T0: PrimInt + FromPrimitive,
        T1: PrimInt + FromPrimitive,
        R0: BoundedRange<T0>,
        R1: BoundedRange<T1>,
    >(
        &mut self,
        range0: R0,
        range1: R1,
    ) -> (T0, T1) {
        let rand_value = self.next_u64();
        let v0 = gen_range_u32(range0, (rand_value >> 0) as u32);
        let v1 = gen_range_u32(range1, (rand_value >> 32) as u32);
        (v0, v1)
    }

    fn fast_gen_range_u16x1<T: PrimInt + FromPrimitive, R: BoundedRange<T>>(
        &mut self,
        range: R,
    ) -> T {
        let rand_value = self.next_u64();
        gen_range_u16(range, rand_value as u16)
    }

    fn fast_gen_range_u16x2<
        T0: PrimInt + FromPrimitive,
        T1: PrimInt + FromPrimitive,
        R0: BoundedRange<T0>,
        R1: BoundedRange<T1>,
    >(
        &mut self,
        range0: R0,
        range1: R1,
    ) -> (T0, T1) {
        let rand_value = self.next_u64();
        let v0 = gen_range_u16(range0, (rand_value >> 0) as u16);
        let v1 = gen_range_u16(range1, (rand_value >> 16) as u16);
        (v0, v1)
    }

    fn fast_gen_range_u16x3<
        T0: PrimInt + FromPrimitive,
        T1: PrimInt + FromPrimitive,
        T2: PrimInt + FromPrimitive,
        R0: BoundedRange<T0>,
        R1: BoundedRange<T1>,
        R2: BoundedRange<T2>,
    >(
        &mut self,
        range0: R0,
        range1: R1,
        range2: R2,
    ) -> (T0, T1, T2) {
        let rand_value = self.next_u64();
        let v0 = gen_range_u16(range0, (rand_value >> 0) as u16);
        let v1 = gen_range_u16(range1, (rand_value >> 16) as u16);
        let v2 = gen_range_u16(range2, (rand_value >> 32) as u16);
        (v0, v1, v2)
    }

    fn fast_gen_range_u16x4<
        T0: PrimInt + FromPrimitive,
        T1: PrimInt + FromPrimitive,
        T2: PrimInt + FromPrimitive,
        T3: PrimInt + FromPrimitive,
        R0: BoundedRange<T0>,
        R1: BoundedRange<T1>,
        R2: BoundedRange<T2>,
        R3: BoundedRange<T3>,
    >(
        &mut self,
        range0: R0,
        range1: R1,
        range2: R2,
        range3: R3,
    ) -> (T0, T1, T2, T3) {
        let rand_value = self.next_u64();
        let v0 = gen_range_u16(range0, (rand_value >> 0) as u16);
        let v1 = gen_range_u16(range1, (rand_value >> 16) as u16);
        let v2 = gen_range_u16(range2, (rand_value >> 32) as u16);
        let v3 = gen_range_u16(range3, (rand_value >> 48) as u16);
        (v0, v1, v2, v3)
    }
}

fn gen_range_u64<T: PrimInt + FromPrimitive, R: BoundedRange<T>>(range: R, rand_value: u64) -> T {
    assert!(!range.is_empty(), "cannot sample empty range");
    let width = range.width().to_u64().expect("width must fit in u64");
    let start = range.start();
    let value = (((width as u128) * (rand_value as u128)) >> 64) as u64;
    T::from_u64(value).expect("value must fit in T") + start
}

fn gen_range_u32<T: PrimInt + FromPrimitive, R: BoundedRange<T>>(range: R, rand_value: u32) -> T {
    assert!(!range.is_empty(), "cannot sample empty range");
    let width = range.width().to_u32().expect("width must fit in u32");
    let start = range.start();
    let value = (((width as u64) * (rand_value as u64)) >> 32) as u32;
    T::from_u32(value).expect("value must fit in T") + start
}

fn gen_range_u16<T: PrimInt + FromPrimitive, R: BoundedRange<T>>(range: R, rand_value: u16) -> T {
    assert!(!range.is_empty(), "cannot sample empty range");
    let width = range.width().to_u16().expect("width must fit in u16");
    let start = range.start();
    let value = (((width as u32) * (rand_value as u32)) >> 16) as u16;
    T::from_u16(value).expect("value must fit in T") + start
}

pub trait BoundedRange<T> {
    fn start(&self) -> T;
    fn width(&self) -> T;
    fn is_empty(&self) -> bool;
}

impl<T: PrimInt> BoundedRange<T> for Range<T> {
    fn start(&self) -> T {
        self.start
    }

    fn width(&self) -> T {
        self.end - self.start
    }

    fn is_empty(&self) -> bool {
        !(self.start < self.end)
    }
}

impl<T: PrimInt> BoundedRange<T> for RangeInclusive<T> {
    fn start(&self) -> T {
        *self.start()
    }

    fn width(&self) -> T {
        *self.end() - *self.start() + T::one()
    }

    fn is_empty(&self) -> bool {
        !(self.start() <= self.end())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::HashSet;

    fn check_range<R: BoundedRange<usize> + Clone, F: Fn(&mut StdRng, R) -> usize>(
        range: R,
        gen: F,
    ) {
        let mut rng = StdRng::seed_from_u64(42);
        let mut values = HashSet::new();
        let start = range.start();
        let width = range.width();
        let end = start + width;

        for _ in 0..10000 {
            let v = gen(&mut rng, range.clone());
            assert!(start <= v && v < end, "value {:?} out of range", v);
            values.insert(v);
        }

        for v in start..end {
            assert!(values.contains(&v), "value {:?} not generated", v);
        }
    }

    #[test]
    fn test_fast_gen_range_u64x1() {
        check_range(0usize..10usize, |rng, r| rng.fast_gen_range_u64x1(r));
    }

    #[test]
    #[should_panic(expected = "cannot sample empty range")]
    fn test_fast_gen_range_u64x1_empty_range() {
        let mut rng = StdRng::seed_from_u64(42);
        let _ = rng.fast_gen_range_u64x1(5usize..5usize);
    }

    #[test]
    fn test_fast_gen_range_u32x1() {
        check_range(0usize..10usize, |rng, r| rng.fast_gen_range_u32x1(r));
    }

    #[test]
    fn test_fast_gen_range_u32x2() {
        let mut rng = StdRng::seed_from_u64(123);
        let mut set0 = HashSet::new();
        let mut set1 = HashSet::new();

        for _ in 0..10000 {
            let (v0, v1) = rng.fast_gen_range_u32x2(0u32..5u32, 10u32..=15u32);
            assert!((0..5).contains(&v0));
            assert!((10..=15).contains(&v1));
            set0.insert(v0);
            set1.insert(v1);
        }

        for v in 0u32..5u32 {
            assert!(set0.contains(&v));
        }

        for v in 10u32..=15u32 {
            assert!(set1.contains(&v));
        }
    }

    #[test]
    fn test_fast_gen_range_u16x1() {
        check_range(0usize..10usize, |rng, r| rng.fast_gen_range_u16x1(r));
    }

    #[test]
    fn test_fast_gen_range_u16x2() {
        let mut rng = StdRng::seed_from_u64(456);
        let mut set0 = HashSet::new();
        let mut set1 = HashSet::new();

        for _ in 0..10000 {
            let (v0, v1) = rng.fast_gen_range_u16x2(0u16..5u16, 10u16..15u16);
            assert!((0..5).contains(&v0));
            assert!((10..15).contains(&v1));
            set0.insert(v0);
            set1.insert(v1);
        }

        for v in 0u16..5u16 {
            assert!(set0.contains(&v));
        }

        for v in 10u16..15u16 {
            assert!(set1.contains(&v));
        }
    }

    #[test]
    fn test_fast_gen_range_u16x3() {
        let mut rng = StdRng::seed_from_u64(789);
        let mut set0 = HashSet::new();
        let mut set1 = HashSet::new();
        let mut set2 = HashSet::new();

        for _ in 0..10000 {
            let (v0, v1, v2) = rng.fast_gen_range_u16x3(0u16..3u16, 10u16..13u16, 20u16..=23u16);
            assert!((0..3).contains(&v0));
            assert!((10..13).contains(&v1));
            assert!((20..=23).contains(&v2));
            set0.insert(v0);
            set1.insert(v1);
            set2.insert(v2);
        }

        for v in 0u16..3u16 {
            assert!(set0.contains(&v));
        }

        for v in 10u16..13u16 {
            assert!(set1.contains(&v));
        }

        for v in 20u16..=23u16 {
            assert!(set2.contains(&v));
        }
    }

    #[test]
    fn test_fast_gen_range_u16x4() {
        let mut rng = StdRng::seed_from_u64(321);
        let mut set0 = HashSet::new();
        let mut set1 = HashSet::new();
        let mut set2 = HashSet::new();
        let mut set3 = HashSet::new();

        for _ in 0..10000 {
            let (v0, v1, v2, v3) =
                rng.fast_gen_range_u16x4(0u16..2u16, 10u16..12u16, 20u16..=22u16, 30u16..32u16);
            assert!((0..2).contains(&v0));
            assert!((10..12).contains(&v1));
            assert!((20..=22).contains(&v2));
            assert!((30..32).contains(&v3));
            set0.insert(v0);
            set1.insert(v1);
            set2.insert(v2);
            set3.insert(v3);
        }

        for v in 0u16..2u16 {
            assert!(set0.contains(&v));
        }

        for v in 10u16..12u16 {
            assert!(set1.contains(&v));
        }

        for v in 20u16..=22u16 {
            assert!(set2.contains(&v));
        }

        for v in 30u16..32u16 {
            assert!(set3.contains(&v));
        }
    }
}
