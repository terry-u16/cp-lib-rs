//! よく使われるユーティリティ関数をまとめたモジュール

use std::{
    fmt::Display,
    io::{self, BufWriter, Write as _},
};

use num::PrimInt;

/// 最小値と最大値を更新するトレイト
///
/// # Examples
///
/// ```
/// use cp_lib_rs::util::ChangeMinMax;
///
/// let mut x = 10;
/// assert!(x.change_min(3));
/// assert_eq!(x, 3);
/// ```
pub trait ChangeMinMax {
    fn change_min(&mut self, v: Self) -> bool;
    fn change_max(&mut self, v: Self) -> bool;
}

impl<T: PartialOrd> ChangeMinMax for T {
    fn change_min(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }

    fn change_max(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

/// 条件に従ってYes/Noを出力する
///
/// # Examples
///
/// ```
/// use cp_lib_rs::yesno;
///
/// let n = 3;
/// yesno!(3 % 2 == 0)
/// ```
#[macro_export]
macro_rules! yesno {
    ($p:expr) => {
        if $p {
            println!("Yes");
        } else {
            println!("No");
        }
    };
}

/// 標準出力・標準エラー出力に出力するトレイト
///
/// # Examples
///
/// ```
/// use cp_lib_rs::util::PrintLine as _;
///
/// let x = 5;
/// x.println();
/// x.eprintln();
///
/// let y = [1, 2, 3];
/// y.println();
/// y.eprintln();
/// ```
pub trait PrintLine {
    fn println(&self);
    fn eprintln(&self);
}

/// 単体値版
impl<T: Display> PrintLine for T {
    fn println(&self) {
        println!("{self}");
    }

    fn eprintln(&self) {
        eprintln!("{self}");
    }
}

/// スライス版
impl<T: Display> PrintLine for [T] {
    fn println(&self) {
        let stdout = io::stdout();
        let mut out = BufWriter::new(stdout.lock());

        let mut first = true;
        for x in self {
            if !first {
                out.write_all(b" ").unwrap();
            }
            write!(out, "{x}").unwrap();
            first = false;
        }

        out.write_all(b"\n").unwrap();
        // drop(out)でflush
    }

    fn eprintln(&self) {
        let stderr = io::stderr();
        let mut out = BufWriter::new(stderr.lock());

        let mut first = true;
        for x in self {
            if !first {
                out.write_all(b" ").unwrap();
            }
            write!(out, "{x}").unwrap();
            first = false;
        }

        out.write_all(b"\n").unwrap();
    }
}

/// 多次元配列を作成する
///
/// # Examples
///
/// ```
/// use cp_lib_rs::mat;
///
/// let a = mat![0; 4; 3];
/// assert_eq!(a, vec![vec![0; 3]; 4]);
/// ```
#[macro_export]
macro_rules! mat {
	($($e:expr),*) => { vec![$($e),*] };
	($($e:expr,)*) => { vec![$($e),*] };
	($e:expr; $d:expr) => { vec![$e; $d] };
	($e:expr; $d:expr $(; $ds:expr)+) => { vec![mat![$e $(; $ds)*]; $d] };
}

/// 整数の二分探索を行う
///
/// # Examples
///
/// ```
/// use cp_lib_rs::util::binary_search;
///
/// let result = binary_search(0, 10, |x| x * x <= 5);
/// assert_eq!(result, 2);
/// ```
pub fn binary_search<T: PrimInt>(ok: T, ng: T, f: impl Fn(T) -> bool) -> T {
    let mut ok = ok;
    let mut ng = ng;

    while ok.max(ng) - ok.min(ng) > T::one() {
        let mid = (ok + ng) / (T::one() << 1);

        if f(mid) {
            ok = mid;
        } else {
            ng = mid;
        }
    }

    ok
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn binary_search_test() {
        assert_eq!(binary_search(0, 10, |x| x * x <= 5), 2);
        assert_eq!(binary_search(10, 0, |x| x * x >= 5), 3);
    }
}
