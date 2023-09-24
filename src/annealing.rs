//! 焼きなましライブラリ

use itertools::Itertools;
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use std::{fmt::Debug, time::Instant};

/// 焼きなましの状態
pub trait State {
    type Score: Score + Clone + PartialEq + Debug;

    /// 生スコア（大きいほど良い）
    fn score(&self) -> Self::Score;
}

pub trait Score {
    /// 焼きなまし用スコア（大きいほど良い）
    /// デフォルトでは生スコアをそのまま返す
    fn annealing_score(&self, _progress: f64) -> f64 {
        self.raw_score() as f64
    }

    /// 生スコア
    fn raw_score(&self) -> i64;
}

/// 単一の値からなるスコア
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SingleScore(i64);

impl Score for SingleScore {
    fn raw_score(&self) -> i64 {
        self.0
    }
}

/// 焼きなましの近傍
///
/// * 受理パターンの流れ: `preprocess()` -> `eval()` -> `postprocess()`
/// * 却下パターンの流れ: `preprocess()` -> `eval()` -> `rollback()`
pub trait Neighbor {
    type Env;
    type State: State;

    /// `eval()` 前の変形操作を行う
    fn preprocess(&mut self, _env: &Self::Env, _state: &mut Self::State);

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
    fn eval(
        &mut self,
        _env: &Self::Env,
        state: &Self::State,
        _progress: f64,
        _threshold: f64,
    ) -> Option<<Self::State as State>::Score> {
        Some(state.score())
    }

    /// `eval()` 後の変形操作を行う（2-optの区間reverse処理など）
    fn postprocess(&mut self, _env: &Self::Env, _state: &mut Self::State);

    /// `preprocess()` で変形した `state` をロールバックする
    fn rollback(&mut self, _env: &Self::Env, _state: &mut Self::State);
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
        rng: &mut impl Rng,
    ) -> Box<dyn Neighbor<Env = Self::Env, State = Self::State>>;
}

#[derive(Debug, Clone)]
pub struct Annealer {
    /// 開始温度
    start_temp: f64,
    /// 終了温度
    end_temp: f64,
    /// 乱数シード
    seed: u128,
    /// 時間計測を行うインターバル
    clock_interval: usize,
}

impl Annealer {
    pub fn new(start_temp: f64, end_temp: f64, seed: u128, clock_interval: usize) -> Self {
        Self {
            start_temp,
            end_temp,
            seed,
            clock_interval,
        }
    }

    pub fn run<E, S: State + Clone, G: NeighborGenerator<Env = E, State = S>>(
        &self,
        env: &E,
        mut state: S,
        neighbor_generator: &G,
        duration_sec: f64,
    ) -> S {
        let mut best_state = state.clone();
        let mut current_score = state.score();
        let mut best_score = current_score.annealing_score(1.0);
        let init_score = best_score;

        let mut all_iter = 0;
        let mut accepted_count = 0;
        let mut update_count = 0;
        let mut rng = Pcg64Mcg::new(self.seed);
        let mut threshold_generator = ThresholdGenerator::new(rng.gen());

        let duration_inv = 1.0 / duration_sec;
        let since = Instant::now();

        let mut progress = 0.0;
        let mut temperature = self.start_temp;

        loop {
            all_iter += 1;
            if all_iter % self.clock_interval == 0 {
                progress = (Instant::now() - since).as_secs_f64() * duration_inv;
                temperature =
                    f64::powf(self.start_temp, 1.0 - progress) * f64::powf(self.end_temp, progress);

                if progress >= 1.0 {
                    break;
                }
            }

            // 変形
            let mut neighbor = neighbor_generator.generate(env, &state, &mut rng);
            neighbor.preprocess(env, &mut state);

            // スコア計算
            let threshold =
                threshold_generator.next(current_score.annealing_score(progress), temperature);
            let Some(new_score) = neighbor.eval(env, &state, progress, threshold) else {
                // 明らかに閾値に届かない場合はreject
                neighbor.rollback(env, &mut state);
                debug_assert_eq!(state.score(), current_score);
                continue;
            };

            if new_score.annealing_score(progress) >= threshold {
                // 解の更新
                neighbor.postprocess(env, &mut state);
                debug_assert_eq!(state.score(), new_score);

                current_score = new_score;
                accepted_count += 1;

                let new_score = current_score.annealing_score(1.0);

                if best_score < new_score {
                    best_score = new_score;
                    best_state = state.clone();
                    update_count += 1;
                }
            } else {
                neighbor.rollback(env, &mut state);
                debug_assert_eq!(state.score(), current_score);
            }
        }

        eprintln!("===== annealing =====");
        eprintln!("init score : {}", init_score);
        eprintln!("score      : {}", best_score);
        eprintln!("all iter   : {}", all_iter);
        eprintln!("accepted   : {}", accepted_count);
        eprintln!("updated    : {}", update_count);
        eprintln!("");

        best_state
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
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use rand::Rng;

    use super::{Annealer, Neighbor, Score};

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
        type Score = Dist;

        fn score(&self) -> Self::Score {
            Dist(self.dist)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Dist(i32);

    impl Score for Dist {
        fn annealing_score(&self, _progress: f64) -> f64 {
            // 大きい方が良いとするため符号を反転
            -self.0 as f64
        }

        fn raw_score(&self) -> i64 {
            self.0 as i64
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

    impl Neighbor for TwoOpt {
        type Env = Input;
        type State = State;

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

        fn postprocess(&mut self, _env: &Self::Env, state: &mut Self::State) {
            state.order[self.begin..self.end].reverse();
            state.dist = self
                .new_dist
                .expect("postprocess()を呼ぶ前にeval()を呼んでください。");
        }

        fn rollback(&mut self, _env: &Self::Env, _state: &mut Self::State) {
            // do nothing
        }
    }

    struct NeighborGenerator;

    impl super::NeighborGenerator for NeighborGenerator {
        type Env = Input;
        type State = State;

        fn generate(
            &self,
            _env: &Self::Env,
            state: &Self::State,
            rng: &mut impl Rng,
        ) -> Box<dyn Neighbor<Env = Self::Env, State = Self::State>> {
            loop {
                let begin = rng.gen_range(1..state.order.len());
                let end = rng.gen_range(1..state.order.len());

                if begin + 2 <= end {
                    return Box::new(TwoOpt::new(begin, end));
                }
            }
        }
    }

    #[test]
    fn annealing_tsp_test() {
        let input = Input::gen_testcase();
        let state = State::new(&input);
        let annealer = Annealer::new(1e1, 1e-1, 42, 1000);
        let neighbor_generator = NeighborGenerator;

        let state = annealer.run(&input, state, &neighbor_generator, 0.1);

        eprintln!("score: {}", state.dist);
        eprintln!("state.dist: {:?}", state.order);
        assert_eq!(state.dist, 10);
        assert!(state.order == vec![0, 1, 3, 2, 0] || state.order == vec![0, 2, 3, 1, 0]);
    }
}
