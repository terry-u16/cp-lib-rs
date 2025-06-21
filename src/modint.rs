use ac_library::modint::ModIntBase;

pub struct Comb<T: ModIntBase> {
    fact: Vec<T>,
    inv_fact: Vec<T>,
}

impl<T: ModIntBase> Comb<T> {
    pub fn new(n: usize) -> Self {
        let mut fact = vec![T::raw(1); n + 1];
        let mut inv_fact = vec![T::raw(1); n + 1];

        for i in 1..=n {
            fact[i] = fact[i - 1] * T::raw(i as u32);
        }

        inv_fact[n] = fact[n].inv();

        for i in (0..n).rev() {
            inv_fact[i] = inv_fact[i + 1] * T::raw((i + 1) as u32);
        }

        Self { fact, inv_fact }
    }

    pub fn fact(&self, n: usize) -> T {
        assert!(n < self.fact.len());
        self.fact[n]
    }

    pub fn inv_fact(&self, n: usize) -> T {
        assert!(n < self.inv_fact.len());
        self.inv_fact[n]
    }

    pub fn comb(&self, n: usize, k: usize) -> T {
        assert!(n < self.fact.len() && k <= n);
        self.fact[n] * self.inv_fact[k] * self.inv_fact[n - k]
    }

    pub fn perm(&self, n: usize, k: usize) -> T {
        assert!(n < self.fact.len() && k <= n);
        self.fact[n] * self.inv_fact[n - k]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ac_library::modint::ModInt998244353;

    #[test]
    fn test_comb_basic() {
        let comb = Comb::<ModInt998244353>::new(10);
        // 5C2 = 10
        assert_eq!(comb.comb(5, 2).val(), 10);
        // 10C0 = 1
        assert_eq!(comb.comb(10, 0).val(), 1);
        // 10C10 = 1
        assert_eq!(comb.comb(10, 10).val(), 1);
        // 10C1 = 10
        assert_eq!(comb.comb(10, 1).val(), 10);
        // 10C2 = 45
        assert_eq!(comb.comb(10, 2).val(), 45);
    }

    #[test]
    fn test_perm_basic() {
        let comb = Comb::<ModInt998244353>::new(10);
        // 5P2 = 20
        assert_eq!(comb.perm(5, 2).val(), 20);
        // 10P0 = 1
        assert_eq!(comb.perm(10, 0).val(), 1);
        // 10P10 = 3628800
        assert_eq!(comb.perm(10, 10).val(), 3628800);
    }

    #[test]
    fn test_fact_inv_fact() {
        let comb = Comb::<ModInt998244353>::new(5);
        // 5! = 120
        assert_eq!(comb.fact(5).val(), 120);
        // 0! = 1
        assert_eq!(comb.fact(0).val(), 1);
        // inv_fact[5] * fact[5] = 1
        assert_eq!((comb.inv_fact(5) * comb.fact(5)).val(), 1);
    }
}
