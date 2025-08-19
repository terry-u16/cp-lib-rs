use std::{
    borrow::Cow,
    cell::RefCell,
    collections::BTreeMap,
    fmt::Display,
    io::{self, IsTerminal as _},
    rc::Rc,
    time::{Duration, Instant},
};

/// 実行時間を計測・集計する構造体
///
/// # Examples
///
/// ```
/// use cp_lib_rs::diagnostics::Perf;
///
/// // 計測グループを作成する
/// // dropされるときに計測結果を出力する
/// let mut perf = Perf::new("Group");
/// let mut _sum = 0u64;
///
/// for i in 0..100000 {
///     // start-stop間の処理時間を計測する
///     let sw = perf.start("sum");
///     _sum += i;
///     sw.stop();
///
///     // 名前は&'strでもStringでもOK
///     let sw = perf.start(format!("no-op"));
///     sw.stop();
/// }
///
/// let mut _sum_sqrt = 0f64;
///
/// for i in 0..100000 {
///     // 明示的にstop()を呼ばなくても、スコープを抜ける際に自動でstopする
///     let _sw = perf.start("sum sqrt");
///     _sum_sqrt += (i as f64).sqrt();
/// }
///
/// let mut _sum_sq = 0u64;
///
/// for i in 0..100000 {
///     // perfのインスタンス化が面倒な場合はシングルトンを使う
///     let sw = Perf::start_singleton("sum sq");
///     _sum_sq += i * i;
///     sw.stop();
/// }
/// ```
pub struct Perf {
    name: Option<Cow<'static, str>>,
    measures: BTreeMap<Cow<'static, str>, Measure>,
}

#[allow(dead_code)]
impl Perf {
    thread_local!(static SINGLETON: Rc<RefCell<Perf>> = Rc::new(RefCell::new(Perf::new("Singleton"))));

    /// 新しい Perf インスタンスを作成する
    pub fn new(name: impl Into<Cow<'static, str>>) -> Self {
        Self {
            name: Some(name.into()),
            measures: BTreeMap::new(),
        }
    }

    /// 匿名の Perf インスタンスを作成する
    pub fn new_anonymous() -> Self {
        Self {
            name: None,
            measures: BTreeMap::new(),
        }
    }

    /// 計測を開始する
    pub fn start<'a>(&'a mut self, name: impl Into<Cow<'static, str>>) -> StopWatch<&'a mut Perf> {
        let name = name.into();

        StopWatch {
            start: Instant::now(),
            perf: self,
            name,
        }
    }

    /// 計測を開始する（シングルトン）
    pub fn start_singleton(name: impl Into<Cow<'static, str>>) -> StopWatch<Rc<RefCell<Perf>>> {
        Self::SINGLETON.with(|perf| StopWatch {
            start: Instant::now(),
            perf: perf.clone(),
            name: name.into(),
        })
    }
}

impl Drop for Perf {
    fn drop(&mut self) {
        if !self.measures.is_empty() {
            // ターミナルかどうかで色を付けるかどうか分岐
            let is_tty = io::stderr().is_terminal();
            let name = self.name.as_deref().unwrap_or("Anonymous");

            if is_tty {
                // コンソール → 色付き
                eprintln!("\x1b[35m[{}] Performance measures\x1b[0m", name);
            } else {
                // リダイレクト → 色なし
                eprintln!("[{}] Performance measures", name);
            }

            for (name, measure) in &self.measures {
                eprintln!("{}: {}", name, measure);
            }
        }
    }
}

pub struct StopWatch<M: WithMut<Perf>> {
    start: Instant,
    perf: M,
    name: Cow<'static, str>,
}

impl<M: WithMut<Perf>> StopWatch<M> {
    pub fn stop(self) {
        std::mem::drop(self);
    }
}

impl<M: WithMut<Perf>> Drop for StopWatch<M> {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        let name = std::mem::take(&mut self.name);

        self.perf.with_mut(|perf| {
            perf.measures
                .entry(name)
                .or_insert_with(Measure::new)
                .add_measure(&duration)
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Measure {
    sum: f64,
    sum_sq: f64,
    cnt: usize,
}

impl Measure {
    fn new() -> Self {
        Self::default()
    }

    fn add_measure(&mut self, duration: &Duration) {
        let sec = duration.as_secs_f64();
        self.sum += sec;
        self.sum_sq += sec * sec;
        self.cnt += 1;
    }

    fn mean(&self) -> Duration {
        if self.cnt == 0 {
            Duration::from_millis(0)
        } else {
            Duration::from_secs_f64(self.sum / self.cnt as f64)
        }
    }

    fn std_dev(&self) -> Duration {
        if self.cnt == 0 {
            Duration::from_millis(0)
        } else {
            let mean = self.mean().as_secs_f64();
            let variance = (self.sum_sq / self.cnt as f64) - (mean * mean);
            Duration::from_secs_f64(variance.sqrt())
        }
    }
}

impl Display for Measure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} ± {:?} ({} samples)",
            self.mean(),
            self.std_dev(),
            self.cnt
        )
    }
}

/// &mut T と Rc<RefCell<T>> を統一的に扱って処理を行うためのトレイト
pub trait WithMut<T> {
    fn with_mut<R>(&mut self, f: impl FnOnce(&mut T) -> R) -> R;
}

impl<T> WithMut<T> for &mut T {
    fn with_mut<R>(&mut self, f: impl FnOnce(&mut T) -> R) -> R {
        f(*self)
    }
}

impl<T> WithMut<T> for Rc<RefCell<T>> {
    fn with_mut<R>(&mut self, f: impl FnOnce(&mut T) -> R) -> R {
        let mut guard = self.borrow_mut();
        f(&mut *guard)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_measure() {
        let mut perf = Perf::new("group");
        let mut _sum = 0u64;

        for i in 0..100000 {
            // start-stop間の処理時間を計測する
            let sw = perf.start("sum");
            _sum += i;
            sw.stop();

            // 名前は&'strでもStringでもOK
            let sw = perf.start(format!("no-op"));
            sw.stop();
        }

        let mut _sum_sqrt = 0f64;

        for i in 0..100000 {
            // 明示的にstop()を呼ばなくても、スコープを抜ける際に自動でstopする
            let _sw = perf.start("sum sqrt".to_string());
            _sum_sqrt += (i as f64).sqrt();
        }

        let mut _sum_sq = 0u64;

        for i in 0..100000 {
            // perfのインスタンス化が面倒な場合はシングルトンを使う
            let sw = Perf::start_singleton("sum sqrt");
            _sum_sq += i * i;
            sw.stop();
        }
    }
}
