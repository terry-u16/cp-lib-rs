struct Comb<T: ModIntBase> {
    fact: Vec<T>,
    inv_fact: Vec<T>,
}

impl<T: ModIntBase> Comb<T> {
    fn new(n: usize) -> Self {
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
}
