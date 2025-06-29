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
