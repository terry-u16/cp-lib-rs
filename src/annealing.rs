//! 焼きなましライブラリ
#![macro_use]
use itertools::{izip, Itertools};
use rand::{distributions::Distribution, Rng as _};
use rand_distr::WeightedAliasIndex;
use rand_pcg::Pcg64Mcg;
use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    rc::Rc,
    time::{Duration, Instant},
};

pub type AnnealingRng = Pcg64Mcg;

/// 焼きなましの状態
pub trait State: Clone {
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
        self.raw_score()
    }

    /// 生スコア
    fn raw_score(&self) -> f64;
}

/// 単一の値からなるスコア
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SingleScore(pub f64);

impl Score for SingleScore {
    fn raw_score(&self) -> f64 {
        self.0
    }
}

/// 焼きなましの近傍
///
/// * 受理パターンの流れ: `preprocess()` -> `eval()` -> `postprocess()`
/// * 却下パターンの流れ: `preprocess()` -> `eval()` -> `rollback()`
pub trait Neighbor: Sized {
    type Env;
    type State: State<Env = Self::Env>;

    fn generate(
        env: &Self::Env,
        state: &Self::State,
        rng: &mut AnnealingRng,
        progress: f64,
    ) -> Option<Self>;

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
    fn postprocess(self, env: &Self::Env, state: &mut Self::State);

    /// `preprocess()` で変形した `state` をロールバックする
    fn rollback(self, env: &Self::Env, state: &mut Self::State);
}

pub trait NeighborEnum: Sized {
    type Env;
    type State: State<Env = Self::Env>;

    fn get_neighbor_weights() -> &'static [f64];

    fn get_neighbor_names() -> &'static [&'static str];

    fn generate(
        env: &Self::Env,
        state: &Self::State,
        rng: &mut AnnealingRng,
        progress: f64,
        neighbor_index: usize,
    ) -> Option<Self>;

    fn preprocess(&mut self, env: &Self::Env, state: &mut Self::State);

    fn eval(
        &mut self,
        env: &Self::Env,
        state: &Self::State,
        progress: f64,
        threshold: f64,
    ) -> Option<<Self::State as State>::Score>;

    fn postprocess(self, env: &Self::Env, state: &mut Self::State);

    fn rollback(self, env: &Self::Env, state: &mut Self::State);
}

/// 近傍をまとめたenumを生成するマクロ
///
/// `enum_name` は生成するenumの名前、`env` は環境の型、`state` は状態の型を指定する。
/// 各近傍は `variant => weight` の形式で指定する。`weight` はその近傍が選ばれる確率の重みを表す。
///
/// # Usage
///
/// ```ignore
/// neighbors! {
///     Neighbors, Env, State, [
///         TwoOpt => 1.0,
///         Swap => 2.0,
///         Insert => 0.5,
///     ]
/// }
///
/// let (state, stats) = run_annealing::<Neighbors, 128>(&input, state, 1e1, 1e-1, Duration::from_millis(1000), 42);
/// ```
#[macro_export]
macro_rules! neighbors {
    (
        $enum_name:ident, $env:ty, $state:ty, [
            $( $variant:ident => $weight:expr ),+ $(,)?
        ]
    ) => {
        use crate::annealing::Neighbor as _;

        // generate()内のmatch式で使用するNEIGHBOR_INDEXを定義する
        neighbors! {
            @index 0; $($variant),+
        }

        // 本体
        neighbors! {
            @body
            $enum_name, $env, $state, [
                $( $variant => $weight ),+
            ]
        }
    };

    (
        @body
        $enum_name:ident, $env:ty, $state:ty, [
            $( $variant:ident => $weight:expr ),+ $(,)?
        ]
    ) => {
        enum $enum_name {
            $(
                $variant($variant),
            )*
        }

        impl $enum_name {
            const WEIGHTS: &'static [f64] = &[$($weight, )+];
            const NAMES: &'static [&'static str] = &[$(stringify!($variant), )+];
        }


        impl crate::annealing::NeighborEnum for $enum_name
        {
            type Env = $env;
            type State = $state;

            fn get_neighbor_weights() -> &'static [f64] {
                Self::WEIGHTS
            }

            fn get_neighbor_names() -> &'static [&'static str] {
                Self::NAMES
            }

            fn generate(
                env: &Self::Env,
                state: &Self::State,
                rng: &mut crate::annealing::AnnealingRng,
                progress: f64,
                neighbor_index: usize,
            ) -> Option<Self> {
                match neighbor_index {
                    $($variant::NEIGHBOR_INDEX => Some($enum_name::$variant($variant::generate(env, state, rng, progress)?))),*,
                    _ => panic!("Invalid neighbor index: {}", neighbor_index),
                }
            }

            fn preprocess(&mut self, env: &Self::Env, state: &mut Self::State) {
                match self {
                    $(Self::$variant(inner) => inner.preprocess(env, state),)+
                }
            }

            fn eval(
                &mut self,
                env: &Self::Env,
                state: &Self::State,
                progress: f64,
                threshold: f64,
            ) -> Option<<Self::State as crate::annealing::State>::Score> {
                match self {
                    $(Self::$variant(inner) => inner.eval(env, state, progress, threshold),)+
                }
            }

            fn postprocess(self, env: &Self::Env, state: &mut Self::State) {
                match self {
                    $(Self::$variant(inner) => inner.postprocess(env, state),)+
                }
            }

            fn rollback(self, env: &Self::Env, state: &mut Self::State) {
                match self {
                    $(Self::$variant(inner) => inner.rollback(env, state),)+
                }
            }
        }
    };

    // 近傍にNEIGHBOR_INDEXを定義する補助マクロ
    // バリアントが複数残っているとき
    (@index $idx:expr; $head:ident, $($tail:ident),+) => {
        impl $head {
            const NEIGHBOR_INDEX: usize = $idx;
        }

        neighbors! {
            @index ($idx + 1); $($tail),+
        }
    };

    // 最後のバリアント
    (@index $idx:expr; $last:ident) => {
        impl $last {
            const NEIGHBOR_INDEX: usize = $idx;
        }
    };
}

/// 焼きなましの統計データ
#[derive(Debug, Clone)]
pub struct AnnealingStatistics<N: NeighborEnum> {
    all_iter: usize,
    valid_iter: usize,
    accepted_count: usize,
    updated_count: usize,
    init_score: f64,
    final_score: f64,
    selected: Vec<u64>,
    accepted: Vec<u64>,
    _phantom: std::marker::PhantomData<N>,
}

impl<N: NeighborEnum> AnnealingStatistics<N> {
    fn new(init_score: f64) -> Self {
        Self {
            all_iter: 0,
            valid_iter: 0,
            accepted_count: 0,
            updated_count: 0,
            init_score,
            final_score: init_score,
            selected: vec![0; N::get_neighbor_names().len()],
            accepted: vec![0; N::get_neighbor_names().len()],
            _phantom: std::marker::PhantomData,
        }
    }

    fn select(&mut self, neighbor_index: usize) {
        self.valid_iter += 1;
        self.selected[neighbor_index] += 1;
    }

    fn accept(&mut self, neighbor_index: usize) {
        self.accepted_count += 1;
        self.accepted[neighbor_index] += 1;
    }
}

impl<N: NeighborEnum> Display for AnnealingStatistics<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "===== annealing =====")?;
        writeln!(f, "init score : {}", self.init_score)?;
        writeln!(f, "score      : {}", self.final_score)?;
        writeln!(f, "all iter   : {}", self.all_iter)?;
        writeln!(f, "valid iter : {}", self.valid_iter)?;
        writeln!(f, "accepted   : {}", self.accepted_count)?;
        writeln!(f, "updated    : {}", self.updated_count)?;

        for (name, &selected, &accepted) in
            izip!(N::get_neighbor_names(), &self.selected, &self.accepted)
        {
            let percent = accepted as f64 / selected as f64 * 100.0;

            writeln!(
                f,
                "{:11}: {} / {} ({:.2}%)",
                name, accepted, selected, percent
            )?;
        }

        Ok(())
    }
}

/// 焼きなましを実行する関数
///
/// `N` は近傍のセットを表すenumで、`NeighborEnum` トレイトを実装している必要がある。
/// `I` は焼きなましの進捗を更新する間隔を指定する。例えば `I = 1024` とすると、1024回に1回の頻度で進捗を更新する。
///
/// # Usage
///
/// ```ignore
/// neighbors! {
///     Neighbors, Env, State, [
///         TwoOpt => 1.0,
///         Swap => 2.0,
///         Insert => 0.5,
///     ]
/// }
///
/// let (state, stats) = run_annealing::<Neighbors, 128>(&input, state, 1e1, 1e-1, Duration::from_millis(1000), 42);
/// ```
pub fn run_annealing<N: NeighborEnum, const I: usize>(
    env: &N::Env,
    mut state: N::State,
    start_temp: f64,
    end_temp: f64,
    duration: Duration,
    seed: u128,
) -> (N::State, AnnealingStatistics<N>) {
    let mut best_state = state.clone();
    let mut current_score = state.score(&env);
    let mut best_score = current_score.annealing_score(1.0);

    let mut stats = AnnealingStatistics::new(current_score.raw_score());
    let mut rng = AnnealingRng::new(seed);
    let neighbor_weights = WeightedAliasIndex::new(N::get_neighbor_weights().to_vec())
        .expect("weights must be non-negative and not all zero");
    let threshold_generator = ThresholdGenerator::get_singleton();
    let mut threshold_generator = threshold_generator.borrow_mut();
    threshold_generator.set_pos(rng.gen());

    let since = Instant::now();
    let duration_inv = 1.0 / duration.as_secs_f64();

    let mut progress = 0.0;
    let mut temperature = start_temp;

    loop {
        if stats.all_iter % I == 0 {
            progress = since.elapsed().as_secs_f64() * duration_inv;
            temperature = start_temp.powf(1.0 - progress) * end_temp.powf(progress);

            if progress >= 1.0 {
                break;
            }
        }

        stats.all_iter += 1;

        // 変形
        let neighbor_index = neighbor_weights.sample(&mut rng);
        let Some(mut neighbor) = N::generate(env, &state, &mut rng, progress, neighbor_index)
        else {
            continue;
        };

        stats.select(neighbor_index);
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
            stats.accept(neighbor_index);
            current_score = new_score;
            neighbor.postprocess(env, &mut state);

            let new_score = current_score.annealing_score(1.0);

            if best_score < new_score {
                best_score = new_score;
                best_state = state.clone();
                stats.updated_count += 1;
            }
        } else {
            neighbor.rollback(env, &mut state);
        }
    }

    stats.final_score = best_state.score(&env).raw_score();

    (best_state, stats)
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
    use crate::{annealing::run_annealing, random::RandExtension};
    use itertools::Itertools;
    use std::time::Duration;

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
        ) -> Option<Self> {
            Some(NoOp)
        }

        fn preprocess(&mut self, _env: &Self::Env, _state: &mut Self::State) {
            // do nothing
        }

        fn postprocess(self, _env: &Self::Env, _state: &mut Self::State) {
            // do nothing
        }

        fn rollback(self, _env: &Self::Env, _state: &mut Self::State) {
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
        ) -> Option<Self> {
            loop {
                let (i, j) = rng.fast_gen_range_u16x2(1..state.order.len(), 1..state.order.len());
                let begin = i.min(j);
                let end = i.max(j);

                if begin + 2 <= end {
                    return Some(TwoOpt::new(begin, end));
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

        fn postprocess(self, _env: &Self::Env, state: &mut Self::State) {
            state.order[self.begin..self.end].reverse();
            state.dist = self
                .new_dist
                .expect("postprocess()を呼ぶ前にeval()を呼んでください。");
        }

        fn rollback(self, _env: &Self::Env, _state: &mut Self::State) {
            // do nothing
        }
    }

    // 近傍セットを表すenumの生成
    crate::neighbors! {
        Neighbors, Input, State, [
            NoOp => 1.0,
            TwoOpt => 2.0,
        ]
    }

    #[test]
    fn annealing_tsp_test() {
        let input = Input::gen_testcase();
        let state = State::new(&input);

        let (state, diagnostics) = run_annealing::<Neighbors, 1024>(
            &input,
            state,
            1e1,
            1e-1,
            Duration::from_millis(100),
            42,
        );

        eprintln!("{}", diagnostics);

        eprintln!("score: {}", state.dist);
        eprintln!("state.dist: {:?}", state.order);
        assert_eq!(state.dist, 10);
        assert!(state.order == vec![0, 1, 3, 2, 0] || state.order == vec![0, 2, 3, 1, 0]);
    }
}
