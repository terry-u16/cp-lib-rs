use itertools::Itertools;
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
/// 素因数分解を行うイテレータを返す
/// 素因数は昇順で返される
///
/// # Examples
///
/// ```
/// use cp_lib_rs::numerics::prime_factorize;
/// use itertools::*;
///
/// let factors = prime_factorize(12).collect_vec();
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

/// エラトステネスの篩
///
/// # Examples
///
/// ```
/// use cp_lib_rs::numerics::Erathosthenes;
/// use itertools::*;
///
/// let erathosthenes = Erathosthenes::new(1000);
/// assert!(erathosthenes.is_prime(2));
///
/// let factors = erathosthenes.prime_factorize(12).collect_vec();
/// assert_eq!(factors, vec![(2, 2), (3, 1)]);
/// ```
#[derive(Debug, Clone)]
pub struct Erathosthenes {
    max: usize,
    smallest_prime_factor: Vec<usize>,
}

impl Erathosthenes {
    /// エラトステネスの篩を初期化する
    ///
    /// # Arguments
    ///
    /// * `max` - 素数判定を行う最大値
    pub fn new(max: u32) -> Self {
        let mut smallest_prime_factor = (0..=max as usize).collect_vec();
        let mut i = 2;

        while i * i <= max as usize {
            if smallest_prime_factor[i] == i {
                for j in (i * 2..=max as usize).step_by(i) {
                    if smallest_prime_factor[j] == j {
                        smallest_prime_factor[j] = i;
                    }
                }
            }

            i += 1;
        }

        Self {
            max: max as usize,
            smallest_prime_factor,
        }
    }

    /// 素数判定を行う
    ///
    /// # Arguments
    ///
    /// * `n` - 素数判定を行う値
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// use cp_lib_rs::numerics::Erathosthenes;
    ///
    /// let erathosthenes = Erathosthenes::new(1000);
    ///
    /// assert!(erathosthenes.is_prime(2));
    /// assert!(!erathosthenes.is_prime(4));
    pub fn is_prime(&self, n: u32) -> bool {
        let n = n as usize;
        assert!(n <= self.max);
        n >= 2 && self.smallest_prime_factor[n] == n
    }

    /// 素因数分解を行う
    ///
    /// # Arguments
    ///
    /// * `n` - 素因数分解を行う値
    pub fn prime_factorize<'a>(&'a self, n: u32) -> ErathosthenesIter<'a> {
        let n = n as usize;
        assert!(n <= self.max);
        ErathosthenesIter {
            n,
            erathosthenes: self,
        }
    }
}

pub struct ErathosthenesIter<'a> {
    n: usize,
    erathosthenes: &'a Erathosthenes,
}

impl<'a> Iterator for ErathosthenesIter<'a> {
    type Item = (u32, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.n > 1 {
            let mut count = 0;
            let spf = self.erathosthenes.smallest_prime_factor[self.n];

            while self.n > 1 && self.erathosthenes.smallest_prime_factor[self.n] == spf {
                count += 1;
                self.n /= spf;
            }

            return Some((spf as u32, count));
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

    #[test]
    fn erathosthenes_is_prime() {
        let erathosthenes = Erathosthenes::new(1000);
        assert!(erathosthenes.is_prime(2));
        assert!(erathosthenes.is_prime(3));
        assert!(erathosthenes.is_prime(5));
        assert!(erathosthenes.is_prime(7));
        assert!(erathosthenes.is_prime(11));
        assert!(!erathosthenes.is_prime(1));
        assert!(!erathosthenes.is_prime(4));
        assert!(!erathosthenes.is_prime(6));
        assert!(!erathosthenes.is_prime(8));
        assert!(!erathosthenes.is_prime(9));
        assert!(!erathosthenes.is_prime(10));
    }

    #[test]
    fn erathosthenes_prime_factorize() {
        let erathosthenes = Erathosthenes::new(1000);
        let factors = erathosthenes.prime_factorize(12).collect_vec();

        assert_eq!(factors, vec![(2, 2), (3, 1)]);
    }
}
