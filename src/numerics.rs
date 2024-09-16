use num::PrimInt;
use num_traits::NumAssign;

/// 最大公約数を求める
///
/// # Examples
///
/// ```
/// use cp_lib_rs::numerics::gcd;
///
/// let x = gcd(12, 18);
/// assert_eq!(x, 6);
/// ```
pub fn gcd<T: PrimInt>(mut a: T, mut b: T) -> T {
    assert!(a >= T::zero());
    assert!(b >= T::zero());

    if a < b {
        std::mem::swap(&mut a, &mut b);
    }

    while b != T::zero() {
        let r = a % b;
        a = b;
        b = r;
    }

    a
}

/// 最小公倍数を求める
///
/// # Examples
///
/// ```
/// use cp_lib_rs::numerics::lcm;
///
/// let x = lcm(12, 18);
/// assert_eq!(x, 36);
/// ```
pub fn lcm<T: PrimInt>(a: T, b: T) -> T {
    a / gcd(a, b) * b
}

/// 約数を列挙する
///
/// 約数を列挙するイテレータを返す
/// 約数は順不同であることに注意
///
/// # Examples
///
/// ```
/// use cp_lib_rs::numerics::calc_divisiors;
/// use itertools::*;
///
/// let div = calc_divisiors(12).sorted().collect_vec();
/// assert_eq!(div, vec![1, 2, 3, 4, 6, 12]);
/// ```
pub fn calc_divisiors<T: PrimInt + NumAssign>(n: T) -> impl Iterator<Item = T> {
    Divisiors::new(n)
}

pub struct Divisiors<T: PrimInt> {
    n: T,
    i: T,
    pair: Option<T>,
}

impl<T: PrimInt> Divisiors<T> {
    fn new(n: T) -> Self {
        assert!(n > T::zero());

        Self {
            n,
            i: T::one(),
            pair: None,
        }
    }
}

impl<T: PrimInt + NumAssign> Iterator for Divisiors<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v) = self.pair.take() {
            return Some(v);
        }

        while self.i * self.i <= self.n {
            let i = self.i;
            self.i += T::one();

            if self.n % i == T::zero() {
                let pair = self.n / i;

                if pair != i {
                    self.pair = Some(pair);
                }

                return Some(i);
            }
        }

        None
    }
}

/// 素因数分解を行う
///
/// # Examples
///
/// ```
/// use cp_lib_rs::numerics::prime_factorize;
/// use itertools::*;
///
/// let factors = prime_factorize(12).sorted().collect_vec();
/// assert_eq!(factors, vec![(2, 2), (3, 1)]);
/// ```
pub fn prime_factorize<T: PrimInt + NumAssign>(n: T) -> impl Iterator<Item = (T, usize)> {
    PrimeFactorize::new(n)
}

struct PrimeFactorize<T: PrimInt + NumAssign> {
    n: T,
    i: T,
}

impl<T: PrimInt + NumAssign> PrimeFactorize<T> {
    fn new(n: T) -> Self {
        assert!(n > T::zero());

        Self {
            n,
            i: T::one() + T::one(),
        }
    }
}

impl<T: PrimInt + NumAssign> Iterator for PrimeFactorize<T> {
    type Item = (T, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while self.i * self.i <= self.n {
            let i = self.i;
            self.i += T::one();

            if self.n % i == T::zero() {
                let mut count = 0;

                while self.n % i == T::zero() {
                    count += 1;
                    self.n /= i;
                }

                return Some((i, count));
            }
        }

        if self.n != T::one() {
            let n = self.n;
            self.n = T::one();
            return Some((n, 1));
        }

        None
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use super::*;

    #[test]
    fn gcd_ok() {
        assert_eq!(gcd(12, 18), 6);
        assert_eq!(gcd(18, 12), 6);
        assert_eq!(gcd(3, 2), 1);
    }

    #[test]
    #[should_panic]
    fn gcd_ng() {
        gcd(-1, 1);
    }

    #[test]
    fn lcm_ok() {
        assert_eq!(lcm(12, 18), 36);
        assert_eq!(lcm(18, 12), 36);
        assert_eq!(lcm(3, 2), 6);
    }

    #[test]
    #[should_panic]
    fn lcm_ng() {
        lcm(-1, 1);
    }

    #[test]
    fn divisiors() {
        let div = calc_divisiors(12).sorted().collect_vec();
        assert_eq!(div, vec![1, 2, 3, 4, 6, 12]);

        let div = calc_divisiors(16).sorted().collect_vec();
        assert_eq!(div, vec![1, 2, 4, 8, 16]);
    }

    #[test]
    fn prime_factorize_ok() {
        let factors = prime_factorize(12).sorted().collect_vec();
        assert_eq!(factors, vec![(2, 2), (3, 1)]);

        let factors = prime_factorize(16).sorted().collect_vec();
        assert_eq!(factors, vec![(2, 4)]);
    }

    #[test]
    #[should_panic]
    fn prime_factorize_ng() {
        prime_factorize(0).count();
    }
}
