use ac_library::Monoid;
use num_traits::bounds::LowerBounded;
use std::{
    convert::Infallible, error::Error, fmt::Display, marker::PhantomData, num::NonZeroUsize,
    time::Instant,
};

pub trait BeamWidthSuggester {
    fn suggest(&mut self) -> usize;
    fn max_width(&self) -> usize;
}

pub struct FixedBeamWidthSuggester {
    beam_width: usize,
}

impl FixedBeamWidthSuggester {
    pub fn new(beam_width: usize) -> Self {
        assert!(beam_width > 0, "Beam width must be greater than 0.");
        Self { beam_width }
    }
}

impl BeamWidthSuggester for FixedBeamWidthSuggester {
    fn suggest(&mut self) -> usize {
        self.beam_width
    }

    fn max_width(&self) -> usize {
        self.beam_width
    }
}

/// ベイズ推定+カルマンフィルタにより適切なビーム幅を計算するBeamWidthSuggester。
/// 1ターンあたりの実行時間が正規分布に従うと仮定し、+3σ分の余裕を持ってビーム幅を決める。
///
/// ## モデル
///
/// カルマンフィルタを適用するにあたって、以下のモデルを考える。
///
/// - `i` ターン目のビーム幅1あたりの所要時間の平均値 `t_i` が正規分布 `N(μ_i, σ_i^2)` に従うと仮定する。
///   - 各ターンに観測される所要時間が `N(μ_i, σ_i^2)` に従うのではなく、所要時間の**平均値**が `N(μ_i, σ_i^2)` に従うとしている点に注意。
///     - すなわち `μ_i` は所要時間の平均値の平均値であり、所要時間の平均値が `μ_i` を中心とした確率分布を形成しているものとしている。ややこしい。
///   - この `μ_i` , `σ_i^2` をベイズ推定によって求めたい。
/// - 所要時間 `t_i` は `t_{i+1}=t_i+N(0, α^2)` により更新されるものとする。
///   - `N(0, α^2)` は標準偏差 `α` のノイズを意味する。お気持ちとしては「実行時間がターン経過に伴ってちょっとずつ変わっていくことがあるよ」という感じ。
///   - `α` は既知の定数とし、適当に決める。
///   - 本来は問題に合わせたちゃんとした更新式にすべき（ターン経過に伴って線形に増加するなど）なのだが、事前情報がないため大胆に仮定する。
/// - 所要時間の観測値 `τ_i` は `τ_i=t_i+N(0, β^2)` により得られるものとする。
///   - `β` は既知の定数とし、適当に決める。
///   - 本来この `β` も推定できると嬉しいのだが、取扱いが煩雑になるためこちらも大胆に仮定する。
///
/// ## モデルの初期化
///
/// - `μ_0` は実行時間制限を `T` 、標準ビーム幅を `W` 、実行ターン数を `M` として、 `μ_0=T/WM` などとすればよい。
/// - `σ_0` は適当に `σ_0=0.1μ_0` とする。ここは標準ビーム幅にどのくらい自信があるかによる。
/// - `α` は適当に `α=0.01μ_0` とする。定数は本当に勘。多分問題に合わせてちゃんと考えた方が良い。
/// - `β` は `σ_0=0.05μ_0` とする。適当なベンチマーク問題で標準偏差を取ったらそのくらいだったため。
///
/// ## モデルの更新
///
/// 以下のように更新をかけていく。
///
/// 1. `t_0=N(μ_0, σ_0^2)` と初期化する。
/// 2. `t_1=t_0+N(0, α^2)` とし、事前分布 `t_1=N(μ_1, σ_1^2)=N(μ_0, σ_0^2+α^2)` を得る。ここはベイズ更新ではなく単純な正規分布の合成でよい。
/// 3. `τ_1` が観測されるので、ベイズ更新して事後分布 `N(μ_1', σ_1^2')` を得る。
/// 4. 同様に `t_2=N(μ_2, σ_2^2)` を得る。
/// 5. `τ_2` を用いてベイズ更新。以下同様。
///
/// ## 適切なビーム幅の推定
///
/// - 余裕を持って、99.8%程度の確率（+3σ）で実行時間制限に収まるようなビーム幅にしたい。
/// - ここで、 `t_i=t_{i+1}=･･･=t_M=N(μ_i, σ_i^2)` と大胆仮定する。
///   - `α` によって `t_i` がどんどん変わってしまうと考えるのは保守的すぎるため。
/// - すると残りターン数 `M_i=M-i` として、 `Στ_i=N(M_i*μ_i, M_i*σ_i^2)` となる。
/// - したがって、残り時間を `T_i` として `W(M_i*μ_i+3(σ_i√M_i))≦T_i` となる最大の `W` を求めればよく、 `W=floor(T_i/(M_i*μ_i+3(σ_i√M_i)))` となる。
/// - 最後に、念のため適当な `W_min` , `W_max` でclampしておく。
pub struct BayesianBeamWidthSuggester {
    /// ビーム幅1あたりの所要時間の平均値の平均値μ_i（逐次更新される）
    mean_sec: f64,
    /// ビーム幅1あたりの所要時間の平均値の分散σ_i^2（逐次更新される）
    variance_sec: f64,
    /// 1ターンごとに状態に作用するノイズの大きさを表す分散α^2（定数）
    variance_state_sec: f64,
    /// 観測時に乗るノイズの大きさを表す分散β^2（定数）
    variance_observe_sec: f64,
    /// 問題の実行時間制限T
    time_limit_sec: f64,
    /// 現在のターン数i
    current_turn: usize,
    /// 最大ターン数M
    max_turn: usize,
    /// ウォームアップターン数（最初のXターン分の情報は採用せずに捨てる）
    warmup_turn: usize,
    /// 最小ビーム幅W_min
    min_beam_width: usize,
    /// 最大ビーム幅W_max
    max_beam_width: usize,
    /// 現在のビーム幅W_i
    current_beam_width: usize,
    /// ログの出力インターバル
    verbose_interval: Option<NonZeroUsize>,
    /// ビーム開始時刻
    start_time: Instant,
    /// 前回の計測時刻
    last_time: Instant,
}

impl BayesianBeamWidthSuggester {
    pub fn new(
        max_turn: usize,
        warmup_turn: usize,
        time_limit_sec: f64,
        standard_beam_width: usize,
        min_beam_width: usize,
        max_beam_width: usize,
        verbose_interval: Option<NonZeroUsize>,
    ) -> Self {
        assert!(
            max_turn * standard_beam_width > 0,
            "ターン数とビーム幅設定が不正です。"
        );
        assert!(
            min_beam_width > 0,
            "最小のビーム幅は正の値でなければなりません。"
        );
        assert!(
            min_beam_width <= max_beam_width,
            "最大のビーム幅は最小のビーム幅以上でなければなりません。"
        );

        let mean_sec = time_limit_sec / (max_turn * standard_beam_width) as f64;

        // 雑にσ=10%ズレると仮定
        let stddev_sec = 0.1 * mean_sec;
        let variance_sec = stddev_sec * stddev_sec;
        let stddev_state_sec = 0.01 * mean_sec;
        let variance_state_sec = stddev_state_sec * stddev_state_sec;
        let stddev_observe_sec = 0.05 * mean_sec;
        let variance_observe_sec = stddev_observe_sec * stddev_observe_sec;

        Self {
            mean_sec,
            variance_sec,
            time_limit_sec,
            variance_state_sec,
            variance_observe_sec,
            current_turn: 0,
            min_beam_width,
            max_beam_width,
            verbose_interval,
            max_turn,
            warmup_turn,
            current_beam_width: 0,
            start_time: Instant::now(),
            last_time: Instant::now(),
        }
    }

    fn update_state(&mut self) {
        // N(0, α^2)のノイズが乗る
        self.variance_sec += self.variance_state_sec;
    }

    fn update_distribution(&mut self, duration_sec: f64) {
        let old_mean = self.mean_sec;
        let old_variance = self.variance_sec;
        let noise_variance = self.variance_observe_sec;

        self.mean_sec = (old_mean * noise_variance + old_variance * duration_sec)
            / (noise_variance + old_variance);
        self.variance_sec = old_variance * noise_variance / (old_variance + noise_variance);
    }

    fn calc_safe_beam_width(&self) -> usize {
        let remaining_turn = (self.max_turn - self.current_turn) as f64;
        let elapsed_time = (Instant::now() - self.start_time).as_secs_f64();
        let remaining_time = self.time_limit_sec - elapsed_time;

        // 平均値の分散σ^2と観測ノイズβ^2が乗ってくると考える
        let variance_total = self.variance_sec + self.variance_observe_sec;

        // N(ξ, η^2)からのサンプリングをK回繰り返すとN(Kξ, Kη^2)となる（はず）
        let mean = remaining_turn * self.mean_sec;
        let variance = remaining_turn * variance_total;
        let stddev = variance.sqrt();

        // 3σの余裕を持たせる
        const SIGMA_COEF: f64 = 3.0;
        let needed_time_per_width = mean + SIGMA_COEF * stddev;
        let beam_width = ((remaining_time / needed_time_per_width) as usize)
            .max(self.min_beam_width)
            .min(self.max_beam_width);

        if let Some(verbose_interval) = self.verbose_interval {
            if self.current_turn % verbose_interval.get() == 0 {
                let stddev_per_run = (self.max_turn as f64 * variance_total).sqrt();
                let stddev_per_turn = variance_total.sqrt();

                eprintln!(
                    "turn: {:4}, beam width: {:4}, pase: {:.3}±{:.3}ms/run, iter time: {:.3}±{:.3}ms",
                    self.current_turn,
                    beam_width,
                    self.mean_sec * (beam_width * self.max_turn) as f64 * 1e3,
                    stddev_per_run * beam_width as f64 * 1e3,
                    self.mean_sec * beam_width as f64 * 1e3,
                    stddev_per_turn * beam_width as f64 * 1e3
                );
            }
        }

        beam_width
    }
}

impl BeamWidthSuggester for BayesianBeamWidthSuggester {
    fn suggest(&mut self) -> usize {
        assert!(
            self.current_turn < self.max_turn,
            "規定ターン終了後にsuggest()が呼び出されました。"
        );

        if self.current_turn > 0 && self.current_turn >= self.warmup_turn {
            let elapsed = (Instant::now() - self.last_time).as_secs_f64();
            let elapsed_per_beam = elapsed / self.current_beam_width as f64;
            self.update_state();
            self.update_distribution(elapsed_per_beam);
        }

        self.last_time = Instant::now();
        let beam_width = self.calc_safe_beam_width();
        self.current_beam_width = beam_width;
        self.current_turn += 1;
        beam_width
    }

    fn max_width(&self) -> usize {
        self.max_beam_width
    }
}

pub(super) struct MaxCostIndex<T>(Infallible, PhantomData<fn() -> T>);

impl<T> Monoid for MaxCostIndex<T>
where
    T: Copy + PartialOrd + LowerBounded + Default,
{
    type S = (T, usize);

    fn identity() -> Self::S {
        (T::min_value(), !0)
    }

    fn binary_operation(a: &Self::S, b: &Self::S) -> Self::S {
        // multi_turnではセグ木容量を2冪で広めに取り、空きslotをidentityで埋める。
        // 実候補のコストがmin_value()のときでも空きslot(index == !0)を選ばない。
        match (a.1 == !0, b.1 == !0) {
            (true, _) => *b,
            (_, true) => *a,
            _ => {
                if a.0 >= b.0 {
                    *a
                } else {
                    *b
                }
            }
        }
    }
}

#[derive(Debug)]
pub enum BeamError {
    NoCandidates(NoCandidatesError),
}

#[derive(Debug)]
pub struct NoCandidatesError {
    turn: usize,
}

impl NoCandidatesError {
    pub(super) fn new(turn: usize) -> Self {
        Self { turn }
    }
}

impl Display for BeamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BeamError::NoCandidates(err) => {
                write!(f, "[turn {}] No candidates found.", err.turn)
            }
        }
    }
}

impl Error for BeamError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayesian_suggester_with_zero_warmup_keeps_distribution_finite_after_first_suggest() {
        let mut suggester = BayesianBeamWidthSuggester::new(10, 0, 1.0, 10, 1, 100, None);

        let beam_width = suggester.suggest();

        assert!((1..=100).contains(&beam_width));
        assert!(suggester.mean_sec.is_finite());
        assert!(suggester.variance_sec.is_finite());
    }
}
