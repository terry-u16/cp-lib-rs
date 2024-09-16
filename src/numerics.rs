use num::PrimInt;

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

#[cfg(test)]
mod test {
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
}
