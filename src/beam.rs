//! ビームサーチライブラリ
//!
//! eijirouさんのライブラリをベースにRust化
//! https://eijirou-kyopro.hatenablog.com/entry/2024/02/01/115639
//!
//! # Traits
//!
//! 実装すべきトレイトは以下の通り
//!
//! - `State`: 状態を表すトレイト
//!   - `make_initial_node`: 初期の `Evaluator` と `BeamHash` を生成する関数
//!   - `expand`: 状態を展開し、 `candidate_set` に候補を追加する関数
//! - `Evaluator`: 評価値を表すトレイト
//!   - `evaluate`: コストを計算する関数（コストは小さい方が良い）
//! - `Action`: 行動を表すトレイト
//!   - `apply`: 行動を適用する関数
//!   - `rollback`: 行動を元に戻す関数
//! - `BeamHash`: ビームサーチのハッシュ値を表すトレイト
//!
//! # Usage
//!
//! ```ignore
//! use cp_lib_rs::beam::common::FixedBeamWidthSuggester;
//! use cp_lib_rs::beam::multi_turn::BeamSearch;
//!
//! let state = State::new();
//! let beam_width_suggester = FixedBeamWidthSuggester::new(4500);
//! // 71119は内部のハッシュテーブルのサイズで、ビーム幅の16倍程度の素数を指定する
//! let beam_search = BeamSearch::<_, 71999>::new(beam_width_suggester, 10000);
//! let result = beam_search.run_default(state).unwrap();
//! ```
//!
//! # Crates
//!
//! - `common`: ビームサーチの共通部分
//! - `single_turn`: 1ターン先にしか遷移できない代わりに高速なビームサーチ
//!   - 内部実装としてはオイラーツアーの辺を配列として保持
//! - `multi_turn`: 複数ターン先に遷移できるビームサーチ
//!   - 内部実装としては二重連鎖木のノードを保持
pub mod common;
pub mod multi_turn;
pub mod single_turn;
