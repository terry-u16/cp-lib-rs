#![macro_use]
//! 焼きなましライブラリ
//!
use itertools::Itertools;
use rand::{distributions::Distribution, Rng as _};
use rand_distr::WeightedAliasIndex;
use rand_pcg::Pcg64Mcg;
use std::{
    cell::RefCell,
    collections::BTreeMap,
    fmt::{Debug, Display},
    rc::Rc,
    time::Instant,
};

pub type AnnealingRng = Pcg64Mcg;

/// 焼きなましの状態
pub trait State {
    type Env;
    type Score: Score;

    /// 生スコア（大きいほど良い）
    fn score(&self, env: &Self::Env) -> Self::Score;
}

pub trait Score {
    /// 焼きなまし用スコア（大きいほど良い）
    /// デフォルトでは生スコアをそのまま返す
    #[allow(unused_variables)]
    fn annealing_score(&self, progress: f64) -> f64 {
        self.raw_score() as f64
    }

    /// 生スコア
    fn raw_score(&self) -> f64;
}

/// 単一の値からなるスコア
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SingleScore(pub i64);

impl Score for SingleScore {
    fn raw_score(&self) -> f64 {
        self.0 as f64
    }
}

/// 焼きなましの近傍
///
/// * 受理パターンの流れ: `preprocess()` -> `eval()` -> `postprocess()`
/// * 却下パターンの流れ: `preprocess()` -> `eval()` -> `rollback()`
pub trait Neighbor {
    type Env;
    type State: State<Env = Self::Env>;

    fn generate(
        env: &Self::Env,
        state: &Self::State,
        rng: &mut AnnealingRng,
        progress: f64,
    ) -> Option<Box<dyn Neighbor<Env = Self::Env, State = Self::State>>>
    where
        Self: Sized;

    /// `eval()` 前の変形操作を行う
    fn preprocess(&mut self, env: &Self::Env, state: &mut Self::State);

    /// 変形後の状態の評価を行う
    ///
    /// # Arguments
    ///
    /// * `env` - 環境
    /// * `state` - 状態
    /// * `progress` - 焼きなましの進捗（[0, 1]の範囲をとる）
    /// * `threshold` - 近傍採用の閾値。新しいスコアがこの値を下回る場合はrejectされる
    ///
    /// # Returns
    ///
    /// 現在の状態のスコア。スコアが `threshold` を下回ることが明らかな場合は `None` を返すことで評価の打ち切りを行うことができる。
    ///
    /// 評価の打ち切りについては[焼きなまし法での評価関数の打ち切り](https://qiita.com/not522/items/cd20b87157d15850d31c)を参照。
    #[allow(unused_variables)]
    fn eval(
        &mut self,
        env: &Self::Env,
        state: &Self::State,
        progress: f64,
        threshold: f64,
    ) -> Option<<Self::State as State>::Score> {
        Some(state.score(env))
    }

    /// `eval()` 後の変形操作を行う（2-optの区間reverse処理など）
    fn postprocess(self: Box<Self>, env: &Self::Env, state: &mut Self::State);

    /// `preprocess()` で変形した `state` をロールバックする
    fn rollback(self: Box<Self>, env: &Self::Env, state: &mut Self::State);

    /// 近傍の名前
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// 焼きなましの近傍を生成する構造体
pub trait NeighborGenerator {
    type Env;
    type State: State;

    /// 近傍を生成する
    fn generate(
        &self,
        env: &Self::Env,
        state: &Self::State,
        rng: &mut AnnealingRng,
        progress: f64,
    ) -> Option<Box<dyn Neighbor<Env = Self::Env, State = Self::State>>>;
}

/// WeightedNeighborGenerator用の簡易マクロ
#[macro_export]
macro_rules! weighted_neighbor {
    ( $( $ty:ident => $weight:expr ),+ $(,)? ) => {
        $crate::annealing::WeightedNeighborGenerator::new(vec![
            $(
                ({
                    // 無理矢理useする
                    use crate::annealing::Neighbor as _;
                    Box::new(|env, state, rng, progress| $ty::generate(env, state, rng, progress))
                }, $weight),
            )+
        ])
    };
}

/// 複数の近傍生成器を重み付きで選択する近傍生成器
///
/// `weighted_neighbor! { NeighborA => 1.0, NeighborB => 2.0 }` のように使用する。
pub struct WeightedNeighborGenerator<E, S: State<Env = E>> {
    weights: WeightedAliasIndex<f64>,
    generators: Vec<
        Box<
            dyn Fn(&E, &S, &mut AnnealingRng, f64) -> Option<Box<dyn Neighbor<Env = E, State = S>>>,
        >,
    >,
}

impl<E, S: State<Env = E>> WeightedNeighborGenerator<E, S> {
    pub fn new(
        candidates: Vec<(
            Box<
                dyn Fn(
                    &E,
                    &S,
                    &mut AnnealingRng,
                    f64,
                ) -> Option<Box<dyn Neighbor<Env = E, State = S>>>,
            >,
            f64,
        )>,
    ) -> Self {
        let weights: Vec<f64> = candidates.iter().map(|c| c.1).collect();
        let weights = WeightedAliasIndex::new(weights)
            .expect("weights must be non-negative and not all zero");
        let generators = candidates.into_iter().map(|(gen, _)| gen).collect_vec();
        Self {
            weights,
            generators,
        }
    }
}

impl<E, S> NeighborGenerator for WeightedNeighborGenerator<E, S>
where
    S: State<Env = E>,
{
    type Env = E;
    type State = S;

    fn generate(
        &self,
        env: &Self::Env,
        state: &Self::State,
        rng: &mut AnnealingRng,
        progress: f64,
    ) -> Option<Box<dyn Neighbor<Env = Self::Env, State = Self::State>>> {
        let idx = self.weights.sample(rng);
        (self.generators[idx])(env, state, rng, progress)
    }
}

/// 焼きなましの統計データ
#[derive(Debug, Clone)]
pub struct AnnealingStatistics {
    all_iter: usize,
    valid_iter: usize,
    accepted_count: usize,
    updated_count: usize,
    init_score: f64,
    final_score: f64,
    selected: BTreeMap<&'static str, usize>,
    accepted: BTreeMap<&'static str, usize>,
}

impl AnnealingStatistics {
    fn new(init_score: f64) -> Self {
        Self {
            all_iter: 0,
            valid_iter: 0,
            accepted_count: 0,
            updated_count: 0,
            init_score,
            final_score: init_score,
            selected: BTreeMap::new(),
            accepted: BTreeMap::new(),
        }
    }

    fn select(&mut self, name: &'static str) {
        self.valid_iter += 1;
        self.selected
            .entry(name)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }

    fn accept(&mut self, name: &'static str) {
        self.accepted_count += 1;
        self.accepted
            .entry(name)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }
}

impl Display for AnnealingStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "===== annealing =====")?;
        writeln!(f, "init score : {}", self.init_score)?;
        writeln!(f, "score      : {}", self.final_score)?;
        writeln!(f, "all iter   : {}", self.all_iter)?;
        writeln!(f, "valid iter : {}", self.valid_iter)?;
        writeln!(f, "accepted   : {}", self.accepted_count)?;
        writeln!(f, "updated    : {}", self.updated_count)?;

        for (name, &selected) in self.selected.iter() {
            let accepted = self.accepted.get(name).copied().unwrap_or(0);
            let precent = accepted as f64 / selected as f64 * 100.0;
            let name = name.split("::").last().unwrap_or(name);

            writeln!(
                f,
                "{:11}: {} / {} ({:.2}%)",
                name, accepted, selected, precent
            )?;
        }

        Ok(())
    }
}

/// 焼きなましを行う構造体
///
/// `I` は焼きなましの進捗を更新する間隔を指定する。例えば `I = 1024` とすると、1024回に1回の頻度で進捗を更新する。
#[derive(Debug, Clone)]
pub struct Annealer<const I: usize> {
    /// 開始温度
    start_temp: f64,
    /// 終了温度
    end_temp: f64,
    /// 乱数シード
    seed: u128,
}

impl<const I: usize> Annealer<I> {
    pub fn new(start_temp: f64, end_temp: f64, seed: u128) -> Self {
        Self {
            start_temp,
            end_temp,
            seed,
        }
    }

    pub fn run<E, S: State<Env = E> + Clone, G: NeighborGenerator<Env = E, State = S>>(
        &self,
        env: &E,
        mut state: S,
        neighbor_generator: &G,
        duration_sec: f64,
    ) -> (S, AnnealingStatistics) {
        let mut best_state = state.clone();
        let mut current_score = state.score(&env);
        let mut best_score = current_score.annealing_score(1.0);

        let mut diagnostics = AnnealingStatistics::new(current_score.raw_score());
        let mut rng = Pcg64Mcg::new(self.seed);
        let threshold_generator = ThresholdGenerator::get_singleton();
        let mut threshold_generator = threshold_generator.borrow_mut();
        threshold_generator.set_pos(rng.gen());

        let duration_inv = 1.0 / duration_sec;
        let since = Instant::now();

        let mut progress = 0.0;
        let mut temperature = self.start_temp;

        loop {
            if diagnostics.all_iter % I == 0 {
                progress = (Instant::now() - since).as_secs_f64() * duration_inv;
                temperature =
                    f64::powf(self.start_temp, 1.0 - progress) * f64::powf(self.end_temp, progress);

                if progress >= 1.0 {
                    break;
                }
            }

            diagnostics.all_iter += 1;

            // 変形
            let Some(mut neighbor) = neighbor_generator.generate(env, &state, &mut rng, progress)
            else {
                continue;
            };

            diagnostics.select(neighbor.name());
            neighbor.preprocess(env, &mut state);

            // スコア計算
            let threshold =
                threshold_generator.next(current_score.annealing_score(progress), temperature);
            let Some(new_score) = neighbor.eval(env, &state, progress, threshold) else {
                // 明らかに閾値に届かない場合はreject
                neighbor.rollback(env, &mut state);
                continue;
            };

            if new_score.annealing_score(progress) >= threshold {
                diagnostics.accept(neighbor.name());
                current_score = new_score;
                neighbor.postprocess(env, &mut state);

                let new_score = current_score.annealing_score(1.0);

                if best_score < new_score {
                    best_score = new_score;
                    best_state = state.clone();
                    diagnostics.updated_count += 1;
                }
            } else {
                neighbor.rollback(env, &mut state);
            }
        }

        diagnostics.final_score = best_state.score(&env).raw_score();

        (best_state, diagnostics)
    }
}

/// 焼きなましにおける評価関数の打ち切り基準となる次の閾値を返す構造体
///
/// 参考: [焼きなまし法での評価関数の打ち切り](https://qiita.com/not522/items/cd20b87157d15850d31c)
struct ThresholdGenerator {
    iter: usize,
    log_randoms: Vec<f64>,
}

impl ThresholdGenerator {
    const LEN: usize = 1 << 16;
    thread_local! {
        static THRESHOLD_GENERATOR: Rc<RefCell<ThresholdGenerator>> = Rc::new(RefCell::new(ThresholdGenerator::new(42)));
    }

    fn new(seed: u128) -> Self {
        let mut rng = Pcg64Mcg::new(seed);
        let log_randoms = (0..Self::LEN)
            .map(|_| rng.gen_range(0.0f64..1.0).ln())
            .collect_vec();

        Self {
            iter: 0,
            log_randoms,
        }
    }

    /// 評価関数の打ち切り基準となる次の閾値を返す
    fn next(&mut self, prev_score: f64, temperature: f64) -> f64 {
        let threshold = prev_score + temperature * self.log_randoms[self.iter % Self::LEN];
        self.iter += 1;
        threshold
    }

    fn set_pos(&mut self, pos: usize) {
        self.iter = pos % Self::LEN;
    }

    fn get_singleton() -> Rc<RefCell<Self>> {
        Self::THRESHOLD_GENERATOR.with(|cell| cell.clone())
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use rand::Rng;

    #[derive(Debug, Clone)]
    struct Input {
        n: usize,
        distances: Vec<Vec<i32>>,
    }

    impl Input {
        fn gen_testcase() -> Self {
            let n = 4;
            let distances = vec![
                vec![0, 2, 3, 10],
                vec![2, 0, 1, 3],
                vec![3, 1, 0, 2],
                vec![10, 3, 2, 0],
            ];

            Self { n, distances }
        }
    }

    #[derive(Debug, Clone)]
    struct State {
        order: Vec<usize>,
        dist: i32,
    }

    impl State {
        fn new(input: &Input) -> Self {
            let mut order = (0..input.n).collect_vec();
            order.push(0);
            let dist = order
                .iter()
                .tuple_windows()
                .map(|(&prev, &next)| input.distances[prev][next])
                .sum();

            Self { order, dist }
        }
    }

    impl super::State for State {
        type Env = Input;
        type Score = Dist;

        fn score(&self, _env: &Self::Env) -> Self::Score {
            Dist(self.dist)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Dist(i32);

    impl super::Score for Dist {
        fn annealing_score(&self, _progress: f64) -> f64 {
            // 大きい方が良いとするため符号を反転
            -self.0 as f64
        }

        fn raw_score(&self) -> f64 {
            self.0 as f64
        }
    }

    struct NoOp;

    impl super::Neighbor for NoOp {
        type Env = Input;
        type State = State;

        fn generate(
            _env: &Self::Env,
            _state: &Self::State,
            _rng: &mut super::AnnealingRng,
            _progress: f64,
        ) -> Option<Box<dyn super::Neighbor<Env = Self::Env, State = Self::State>>>
        where
            Self: Sized,
        {
            Some(Box::new(NoOp))
        }

        fn preprocess(&mut self, _env: &Self::Env, _state: &mut Self::State) {
            // do nothing
        }

        fn postprocess(self: Box<Self>, _env: &Self::Env, _state: &mut Self::State) {
            // do nothing
        }

        fn rollback(self: Box<Self>, _env: &Self::Env, _state: &mut Self::State) {
            // do nothing
        }
    }

    struct TwoOpt {
        begin: usize,
        end: usize,
        new_dist: Option<i32>,
    }

    impl TwoOpt {
        fn new(begin: usize, end: usize) -> Self {
            Self {
                begin,
                end,
                new_dist: None,
            }
        }
    }

    impl super::Neighbor for TwoOpt {
        type Env = Input;
        type State = State;

        fn generate(
            _env: &Self::Env,
            state: &Self::State,
            rng: &mut super::AnnealingRng,
            _progress: f64,
        ) -> Option<Box<dyn super::Neighbor<Env = Self::Env, State = Self::State>>>
        where
            Self: Sized,
        {
            loop {
                let begin = rng.gen_range(1..state.order.len());
                let end = rng.gen_range(1..state.order.len());

                if begin + 2 <= end {
                    return Some(Box::new(TwoOpt::new(begin, end)));
                }
            }
        }

        fn preprocess(&mut self, _env: &Self::Env, _state: &mut Self::State) {
            // do nothing
        }

        fn eval(
            &mut self,
            env: &Self::Env,
            state: &Self::State,
            _progress: f64,
            _threshold: f64,
        ) -> Option<<Self::State as super::State>::Score> {
            let v0 = state.order[self.begin - 1];
            let v1 = state.order[self.begin];
            let v2 = state.order[self.end - 1];
            let v3 = state.order[self.end];

            let d00 = env.distances[v0][v1];
            let d01 = env.distances[v0][v2];
            let d10 = env.distances[v2][v3];
            let d11 = env.distances[v1][v3];

            let new_dist = state.dist - d00 - d10 + d01 + d11;
            self.new_dist = Some(new_dist);

            Some(Dist(new_dist))
        }

        fn postprocess(self: Box<Self>, _env: &Self::Env, state: &mut Self::State) {
            state.order[self.begin..self.end].reverse();
            state.dist = self
                .new_dist
                .expect("postprocess()を呼ぶ前にeval()を呼んでください。");
        }

        fn rollback(self: Box<Self>, _env: &Self::Env, _state: &mut Self::State) {
            // do nothing
        }
    }

    #[test]
    fn annealing_tsp_test() {
        let input = Input::gen_testcase();
        let state = State::new(&input);
        let annealer = super::Annealer::<1024>::new(1e1, 1e-1, 42);
        let neighbor_generator = crate::weighted_neighbor! {
            NoOp => 1.0,
            TwoOpt => 2.0,
        };

        let (state, diagnostics) = annealer.run(&input, state, &neighbor_generator, 0.1);

        eprintln!("{}", diagnostics);

        eprintln!("score: {}", state.dist);
        eprintln!("state.dist: {:?}", state.order);
        assert_eq!(state.dist, 10);
        assert!(state.order == vec![0, 1, 3, 2, 0] || state.order == vec![0, 2, 3, 1, 0]);
    }
}
