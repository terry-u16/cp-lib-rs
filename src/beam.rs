//! ビームサーチライブラリ
//!
//! eijirouさんのライブラリをベースにRust化
//! <https://eijirou-kyopro.hatenablog.com/entry/2024/02/01/115639>
//!
//! # Traits
//!
//! 実装すべきトレイトは以下の通り
//! `State`・`Evaluator`・`Action`・`BeamHash` は `single_turn` と `multi_turn` で
//! それぞれ別のトレイトなので、使用するモジュールのものを選ぶ。
//!
//! - `State`: 状態を表すトレイト
//!   - `make_initial_node`: 初期の `Evaluator` と `BeamHash` を生成する関数
//!   - `expand`: 状態を展開し、 `candidate_set` に候補を追加する関数
//! - `Evaluator`: 評価値を表すトレイト
//!   - `evaluate`: コストを計算する関数（コストは小さい方が良い）
//! - `Action`: 行動を表すトレイト
//!   - `apply`: 行動を適用する関数
//!   - `rollback`: 行動を元に戻す関数
//! - `BeamHash`: 状態の重複判定に使う値のマーカートレイト
//!   - `u8`・`u16`・`u32`・`u64` は実装済み
//!   - 値は `make_initial_node` と `expand` で利用者が生成する
//!
//! 候補の重複判定には内部で `FxHashMap` を使う。
//! ハッシュテーブルのサイズや素数の指定は不要。
//!
//! # Usage
//!
//! ```
//! use cp_lib_rs::beam::common::{BeamError, FixedBeamWidthSuggester};
//! use cp_lib_rs::beam::multi_turn::{BeamSearch, State};
//!
//! // 問題に合わせて State を実装した状態を渡す。
//! fn solve<S: State>(state: S) -> Result<Vec<S::Action>, BeamError> {
//!     let beam_width_suggester = FixedBeamWidthSuggester::new(4500);
//!     let beam_search = BeamSearch::new(beam_width_suggester, 10000);
//!     beam_search.run_default(state)
//! }
//! ```
//!
//! # Modules
//!
//! - `common`: ビームサーチの共通部分
//! - `single_turn`: 1ターン先にしか遷移できない代わりに高速なビームサーチ
//!   - 内部実装としてはオイラーツアーの辺を配列として保持
//! - `multi_turn`: 複数ターン先に遷移できるビームサーチ
//!   - 内部実装としては二重連鎖木のノードを保持
pub mod common;
pub mod multi_turn;
pub mod single_turn;
