//! オイラーツアーの辺を保持するビームサーチ
//!
//! 1ターン先にしか遷移できない代わりに高速
use super::common::{BeamError, BeamWidthSuggester, MaxCostIndex, NoCandidatesError};
use ac_library::Segtree;
use num_traits::bounds::LowerBounded;
use rustc_hash::FxHashMap;
use std::{
    collections::hash_map::Entry,
    fmt::{Debug, Display},
    hash::Hash,
    marker::PhantomData,
    time::Instant,
};

/// 状態遷移を行うために必要な情報
/// メモリ使用量をできるだけ小さくしてください
pub trait Action: Clone + Eq {
    type State: State<Action = Self>;

    fn apply(&self, state: &mut Self::State);

    fn rollback(&self, state: &mut Self::State);
}

/// 状態のコストを評価するための構造体
/// メモリ使用量をできるだけ小さくしてください
pub trait Evaluator: Clone {
    type Cost: Copy + Ord + LowerBounded + Default + Display;

    fn evaluate(&self) -> Self::Cost;
}

pub trait BeamHash: Copy + Eq + Hash {}
impl BeamHash for u8 {}
impl BeamHash for u16 {}
impl BeamHash for u32 {}
impl BeamHash for u64 {}

pub trait State {
    type Evaluator: Evaluator;
    type Hash: BeamHash;
    type Action: Action<State = Self>;

    fn make_initial_node(&self) -> (Self::Evaluator, Self::Hash);

    fn expand(
        &mut self,
        evaluator: &Self::Evaluator,
        hash: Self::Hash,
        candidate_set: &mut impl CandidateSet<
            Action = Self::Action,
            Evaluator = Self::Evaluator,
            Hash = Self::Hash,
        >,
    );
}

#[derive(Clone)]
pub struct Candidate<A: Action, E: Evaluator, H: BeamHash> {
    action: A,
    evaluator: E,
    hash: H,
    parent_id: ParentId,
}

impl<A: Action, E: Evaluator, H: BeamHash> Candidate<A, E, H> {
    #[inline]
    fn new(action: A, evaluator: E, hash: H, parent_id: ParentId) -> Self {
        Self {
            action,
            evaluator,
            hash,
            parent_id,
        }
    }

    pub fn action(&self) -> &A {
        &self.action
    }

    pub fn evaluator(&self) -> &E {
        &self.evaluator
    }

    pub fn hash(&self) -> &H {
        &self.hash
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ParentId(u32);

impl ParentId {
    fn index(&self) -> usize {
        self.0 as usize
    }
}

pub trait CandidateSet {
    type Action: Action;
    type Evaluator: Evaluator;
    type Hash: BeamHash;

    fn push(
        &mut self,
        action: Self::Action,
        evaluator: Self::Evaluator,
        hash: Self::Hash,
        is_finished: bool,
    );
}

struct CandidateSetImpl<'a, A: Action, E: Evaluator, H: BeamHash> {
    selector: &'a mut NodeSelector<A, E, H>,
    parent_id: ParentId,
}

impl<'a, A: Action, E: Evaluator, H: BeamHash> CandidateSetImpl<'a, A, E, H> {
    fn new(selector: &'a mut NodeSelector<A, E, H>, parent_id: ParentId) -> Self {
        Self {
            selector,
            parent_id,
        }
    }
}

impl<'a, A: Action, E: Evaluator, H: BeamHash> CandidateSet for CandidateSetImpl<'a, A, E, H> {
    type Action = A;
    type Evaluator = E;
    type Hash = H;

    #[inline]
    fn push(
        &mut self,
        action: Self::Action,
        evaluator: Self::Evaluator,
        hash: Self::Hash,
        is_finished: bool,
    ) {
        self.selector.push(
            Candidate::new(action, evaluator, hash, self.parent_id),
            is_finished,
        );
    }
}

/// ノードの候補から実際に追加するものを選ぶ構造体
/// ビーム幅の個数だけ、評価がよいものを選ぶ
/// ハッシュ値が一致したものについては、評価がよいほうのみを残す
struct NodeSelector<A: Action, E: Evaluator, H: BeamHash> {
    beam_width: usize,
    candidates: Vec<Candidate<A, E, H>>,
    hash_to_index: FxHashMap<H, usize>,
    costs: Vec<(E::Cost, usize)>,
    /// セグメント木を削除可能な優先度付きキューとして使う
    cost_segtree: Option<Segtree<MaxCostIndex<E::Cost>>>,
    finished_candidates: Vec<Candidate<A, E, H>>,
}

impl<A: Action, E: Evaluator, H: BeamHash> NodeSelector<A, E, H> {
    fn new(max_beam_width: usize) -> Self {
        let candidates = Vec::with_capacity(max_beam_width);
        let costs = Vec::with_capacity(max_beam_width);

        Self {
            beam_width: max_beam_width,
            candidates,
            hash_to_index: FxHashMap::default(),
            costs,
            cost_segtree: None,
            finished_candidates: Vec::new(),
        }
    }

    fn push(&mut self, candidate: Candidate<A, E, H>, is_finished: bool) {
        if is_finished {
            self.finished_candidates.push(candidate);
            return;
        }

        let cost = candidate.evaluator.evaluate();

        // 保持しているどの候補よりもコストが小さくないとき
        if let Some(segtree) = &self.cost_segtree {
            if cost >= segtree.all_prod().0 {
                return;
            }
        }

        match self.hash_to_index.entry(candidate.hash) {
            // ハッシュ値が等しいものが存在している場合
            Entry::Occupied(entry) => {
                let index_c = *entry.get();
                let old_cand = &self.candidates[index_c];

                // 同じハッシュの状態は1つしか保持しないため、その場合は上書き処理となる
                if candidate.hash == old_cand.hash {
                    // セグ木が構築されているかどうかで場合分け
                    match &mut self.cost_segtree {
                        Some(segtree) => {
                            if cost < segtree.get(index_c).0 {
                                self.candidates[index_c] = candidate;
                                segtree.set(index_c, (cost, index_c));
                            }
                        }
                        None => {
                            if cost < self.costs[index_c].0 {
                                self.candidates[index_c] = candidate;
                                self.costs[index_c] = (cost, index_c);
                            }
                        }
                    }
                }
            }
            // ハッシュ値が等しいものが存在していない場合
            Entry::Vacant(vacant_entry) => {
                // セグ木が構築されているかで場合分け
                match &mut self.cost_segtree {
                    Some(segtree) => {
                        let index_c = segtree.all_prod().1;
                        let old_hash = self.candidates[index_c].hash;
                        vacant_entry.insert(index_c);
                        self.hash_to_index.remove(&old_hash);
                        self.candidates[index_c] = candidate;
                        segtree.set(index_c, (cost, index_c));
                    }
                    None => {
                        let index_c = self.candidates.len();
                        vacant_entry.insert(index_c);
                        self.candidates.push(candidate);
                        self.costs.push((cost, index_c));

                        // 保持している候補がビーム幅分になったときにセグ木を構築する
                        if self.candidates.len() >= self.beam_width {
                            // ビーム幅の変動に備え、少し多めに確保
                            let mut costs = Vec::with_capacity(self.beam_width * 12 / 10);
                            std::mem::swap(&mut self.costs, &mut costs);
                            let segtree = Segtree::from(costs);
                            self.cost_segtree = Some(segtree);
                        }
                    }
                }
            }
        }
    }

    fn iter_candidates_and_clear<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = Candidate<A, E, H>> + 'a {
        self.costs.clear();
        self.hash_to_index.clear();
        self.cost_segtree = None;

        self.candidates.drain(..)
    }

    fn calculate_best_candidate(&self) -> Option<&Candidate<A, E, H>> {
        match &self.cost_segtree {
            Some(segtree) => {
                let mut best_index = 0;
                let (mut best_cost, _) = segtree.get(0);

                for i in 1..self.beam_width {
                    let cost = segtree.get(i).0;

                    if cost < best_cost {
                        best_cost = cost;
                        best_index = i;
                    }
                }

                Some(&self.candidates[best_index])
            }
            None => {
                if self.candidates.is_empty() {
                    return None;
                }

                let mut best_index = 0;
                let (mut best_cost, _) = self.costs[0];

                for i in 1..self.costs.len() {
                    let cost = self.costs[i].0;

                    if cost < best_cost {
                        best_cost = cost;
                        best_index = i;
                    }
                }

                Some(&self.candidates[best_index])
            }
        }
    }

    pub(super) fn set_beam_width(&mut self, beam_width: usize) {
        self.beam_width = beam_width;
    }
}

struct BeamTree<E: Evaluator, H: BeamHash, S: State, A: Action> {
    state: S,
    current_tour: Vec<(TourStep, A)>,
    next_tour: Vec<(TourStep, A)>,
    leaves: Vec<(E, H)>,
    buckets: Vec<Vec<(A, E, H)>>,
    direct_road: Vec<A>,
}

impl<E: Evaluator, H: BeamHash, S: State<Evaluator = E, Hash = H, Action = A>, A: Action<State = S>>
    BeamTree<E, H, S, A>
{
    fn new(state: S, max_beam_width: usize) -> Self {
        Self {
            state,
            current_tour: vec![],
            next_tour: vec![],
            leaves: vec![],
            buckets: vec![vec![]; max_beam_width],
            direct_road: vec![],
        }
    }

    /// 状態を更新しながら深さ優先探索を行い、次のノードの候補を全てselectorに追加する
    fn dfs(&mut self, selector: &mut NodeSelector<A, E, H>) {
        if self.current_tour.is_empty() {
            // 最初のターン
            let (evaluator, hash) = self.state.make_initial_node();
            let mut cand_set = CandidateSetImpl::new(selector, ParentId(0));
            self.state.expand(&evaluator, hash, &mut cand_set);
            return;
        }

        for (tour_step, action) in self.current_tour.iter() {
            match *tour_step {
                TourStep::FORWARD_EDGE => {
                    action.apply(&mut self.state);
                }
                TourStep::BACKWARD_EDGE => {
                    action.rollback(&mut self.state);
                }
                TourStep(leaf_index) => {
                    let leaf_index = leaf_index as usize;
                    action.apply(&mut self.state);
                    let (evaluator, hash) = &self.leaves[leaf_index];
                    let mut cand_set = CandidateSetImpl::new(selector, ParentId(leaf_index as u32));
                    self.state.expand(evaluator, *hash, &mut cand_set);
                    action.rollback(&mut self.state);
                }
            }
        }
    }

    fn update(&mut self, candidates: impl Iterator<Item = Candidate<A, E, H>>) {
        self.leaves.clear();

        if self.current_tour.is_empty() {
            // 最初のターン
            for Candidate {
                action,
                evaluator,
                hash,
                ..
            } in candidates
            {
                let step = TourStep::new_leaf(self.leaves.len() as u32);
                self.current_tour.push((step, action));
                self.leaves.push((evaluator, hash));
            }

            return;
        }

        for Candidate {
            action,
            evaluator,
            hash,
            parent_id,
        } in candidates
        {
            self.buckets[parent_id.index()].push((action, evaluator, hash));
        }

        let mut tour_index = 0;
        let tour = &mut self.current_tour;

        // 一本道を反復しないようにする
        while tour[tour_index].0 == TourStep::FORWARD_EDGE
            && tour[tour_index].1 == tour.last().unwrap().1
        {
            let action = tour[tour_index].1.clone();
            tour_index += 1;
            action.apply(&mut self.state);
            self.direct_road.push(action);
            tour.pop();
        }

        // 葉の追加や不要な辺の削除をする
        for (tour_step, action) in tour.drain(tour_index..) {
            match tour_step {
                TourStep::FORWARD_EDGE => {
                    self.next_tour.push((TourStep::FORWARD_EDGE, action));
                }
                TourStep::BACKWARD_EDGE => {
                    let (prev_step, _) = self.next_tour.last().unwrap();

                    match *prev_step {
                        TourStep::FORWARD_EDGE => {
                            // 行ってすぐ戻る場合は削除
                            self.next_tour.pop();
                        }
                        _ => {
                            self.next_tour.push((TourStep::BACKWARD_EDGE, action));
                        }
                    }
                }
                TourStep(leaf_index) => {
                    let leaf_index = leaf_index as usize;
                    let bucket = &mut self.buckets[leaf_index];

                    if bucket.is_empty() {
                        continue;
                    }

                    self.next_tour
                        .push((TourStep::FORWARD_EDGE, action.clone()));

                    for (action, evaluator, hash) in bucket.drain(..) {
                        let leaf_index = self.leaves.len() as u32;
                        self.next_tour
                            .push((TourStep::new_leaf(leaf_index), action));
                        self.leaves.push((evaluator, hash));
                    }

                    self.next_tour.push((TourStep::BACKWARD_EDGE, action));
                }
            }
        }

        std::mem::swap(&mut self.current_tour, &mut self.next_tour);
        self.next_tour.clear();
    }

    fn restore_path(&self, parent_id: ParentId) -> Vec<A> {
        let mut result = self.direct_road.clone();

        for (tour_step, action) in self.current_tour.iter() {
            match *tour_step {
                TourStep::FORWARD_EDGE => {
                    result.push(action.clone());
                }
                TourStep::BACKWARD_EDGE => {
                    result.pop();
                }
                TourStep(leaf_index) => {
                    if leaf_index == parent_id.0 {
                        result.push(action.clone());
                        return result;
                    }
                }
            }
        }

        unreachable!("There is no path to the parent node.");
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TourStep(u32);

impl TourStep {
    const FORWARD_EDGE: Self = Self(u32::MAX);
    const BACKWARD_EDGE: Self = Self(u32::MAX - 1);

    const fn new_leaf(leaf_index: u32) -> Self {
        Self(leaf_index)
    }
}

/// ビームサーチを行う構造体
///
/// 使用には以下の実装が必要になる。
///
/// - `State`: 状態遷移を行うための構造体
/// - `Evaluator`: 状態のコストを評価するための構造体
/// - `BeamHash`: ビームサーチのためのハッシュ値を計算するための構造体
/// - `Action`: 状態遷移を行うためのアクションを表す構造体
/// - `BeamWidthSuggester`: ビーム幅を提案するための構造体（Optional）
pub struct BeamSearch<W: BeamWidthSuggester> {
    beam_width_suggester: W,
    max_turn: usize,
}

impl<W: BeamWidthSuggester> BeamSearch<W> {
    pub fn new(beam_width_suggester: W, max_turn: usize) -> Self {
        assert!(max_turn > 0, "Max turn must be greater than 0.");

        Self {
            beam_width_suggester,
            max_turn,
        }
    }

    /// デフォルトの標準エラー出力用コールバックを使ってビームサーチを実行する
    pub fn run_default<S: State>(self, state: S) -> Result<Vec<S::Action>, BeamError> {
        let callbacks: Vec<Box<dyn BeamCallback<State = S>>> =
            vec![Box::new(DefaultBeamCallback::<S>::new())];
        self.run(state, callbacks)
    }

    /// 任意のコールバックを使ってビームサーチを実行する
    ///
    /// コールバックをカスタムする必要がないときは `run_default` を使うことを推奨
    pub fn run<'a, S: State>(
        mut self,
        state: S,
        mut callbacks: Vec<Box<dyn BeamCallback<State = S> + 'a>>,
    ) -> Result<Vec<S::Action>, BeamError> {
        let mut tree = BeamTree::new(state, self.beam_width_suggester.max_width());
        let mut selector = NodeSelector::new(self.beam_width_suggester.max_width());

        for turn in 0..self.max_turn {
            let beam_width = self.beam_width_suggester.suggest();
            assert!(
                beam_width > 0 && beam_width <= self.beam_width_suggester.max_width(),
                "Beam width must be in 1..=max_width()."
            );

            for callback in &mut callbacks {
                callback.on_turn_start(turn, beam_width);
            }

            selector.set_beam_width(beam_width);

            // Euler Tourでselectorに候補を追加する
            tree.dfs(&mut selector);

            let best_candidate = selector.calculate_best_candidate();

            for callback in &mut callbacks {
                callback.on_expanded(turn, selector.candidates.as_slice(), best_candidate);
            }

            // ターン数最小化型の問題で実行可能解が見つかったとき
            if let Some(cand) = selector
                .finished_candidates
                .iter()
                .min_by_key(|c| c.evaluator.evaluate())
            {
                let mut actions = tree.restore_path(cand.parent_id);
                actions.push(cand.action.clone());
                return Ok(actions);
            }

            if selector.candidates.is_empty() {
                return Err(BeamError::NoCandidates(NoCandidatesError::new(turn)));
            }

            // ターン数固定型の問題で全ターンが終了したとき
            if turn == self.max_turn - 1 {
                break;
            }

            // 木を更新する
            tree.update(selector.iter_candidates_and_clear());

            for callback in &mut callbacks {
                callback.on_turn_end(turn);
            }
        }

        match selector.calculate_best_candidate() {
            None => Err(BeamError::NoCandidates(NoCandidatesError::new(
                self.max_turn,
            ))),
            Some(best_cand) => {
                let mut actions = tree.restore_path(best_cand.parent_id);
                actions.push(best_cand.action.clone());
                Ok(actions)
            }
        }
    }
}

pub type BeamCallbackCandidate<S> =
    Candidate<<S as State>::Action, <S as State>::Evaluator, <S as State>::Hash>;

pub trait BeamCallback {
    type State: State;

    fn on_turn_start(&mut self, turn: usize, beam_width: usize);

    fn on_expanded(
        &mut self,
        turn: usize,
        canidates: &[BeamCallbackCandidate<Self::State>],
        best_candidate: Option<&BeamCallbackCandidate<Self::State>>,
    );

    fn on_turn_end(&mut self, turn: usize);
}

/// デフォルトの標準エラー出力用コールバック
///
/// ターン数・ビーム幅・最良候補の評価値・実行時間を出力する
pub struct DefaultBeamCallback<S: State> {
    since: Instant,
    since_turn: Instant,
    phantom: PhantomData<S>,
}

impl<S: State> DefaultBeamCallback<S> {
    fn new() -> Self {
        Self {
            since: Instant::now(),
            since_turn: Instant::now(),
            phantom: PhantomData,
        }
    }
}

impl<S: State> BeamCallback for DefaultBeamCallback<S> {
    type State = S;

    fn on_turn_start(&mut self, turn: usize, beam_width: usize) {
        self.since_turn = Instant::now();
        eprintln!("[Turn {turn}]");
        eprintln!("beam width = {beam_width}");
    }

    fn on_expanded(
        &mut self,
        _turn: usize,
        _canidates: &[BeamCallbackCandidate<Self::State>],
        best_candidate: Option<&BeamCallbackCandidate<Self::State>>,
    ) {
        match best_candidate {
            None => {
                eprintln!("No candidates found.");
                return;
            }
            Some(best_candidate) => {
                let best_score = best_candidate.evaluator.evaluate();
                eprintln!("best score = {best_score}");
            }
        }
    }

    fn on_turn_end(&mut self, _turn: usize) {
        eprintln!("elapsed = {:?}", self.since_turn.elapsed());
        eprintln!("total elapsed = {:?}", self.since.elapsed());
        eprintln!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beam::common::BeamWidthSuggester;

    #[derive(Clone, Eq, PartialEq)]
    struct TestAction;

    struct TestState;

    #[derive(Clone)]
    struct TestEvaluator(i32);

    struct InvalidBeamWidthSuggester;

    impl BeamWidthSuggester for InvalidBeamWidthSuggester {
        fn suggest(&mut self) -> usize {
            2
        }

        fn max_width(&self) -> usize {
            1
        }
    }

    impl Action for TestAction {
        type State = TestState;

        fn apply(&self, _state: &mut Self::State) {}

        fn rollback(&self, _state: &mut Self::State) {}
    }

    impl Evaluator for TestEvaluator {
        type Cost = i32;

        fn evaluate(&self) -> Self::Cost {
            self.0
        }
    }

    impl State for TestState {
        type Evaluator = TestEvaluator;
        type Hash = u64;
        type Action = TestAction;

        fn make_initial_node(&self) -> (Self::Evaluator, Self::Hash) {
            (TestEvaluator(0), 0)
        }

        fn expand(
            &mut self,
            _evaluator: &Self::Evaluator,
            _hash: Self::Hash,
            _candidate_set: &mut impl CandidateSet<
                Action = Self::Action,
                Evaluator = Self::Evaluator,
                Hash = Self::Hash,
            >,
        ) {
        }
    }

    #[test]
    fn node_selector_removes_stale_hash_when_replacing_worst_candidate() {
        let mut selector: NodeSelector<TestAction, TestEvaluator, u64> = NodeSelector::new(2);

        selector.push(
            Candidate::new(TestAction, TestEvaluator(10), 1, ParentId(0)),
            false,
        );
        selector.push(
            Candidate::new(TestAction, TestEvaluator(20), 2, ParentId(0)),
            false,
        );
        selector.push(
            Candidate::new(TestAction, TestEvaluator(5), 3, ParentId(0)),
            false,
        );
        selector.push(
            Candidate::new(TestAction, TestEvaluator(1), 2, ParentId(0)),
            false,
        );

        let best = selector.calculate_best_candidate().unwrap();
        assert_eq!(*best.hash(), 2);
        assert_eq!(best.evaluator().evaluate(), 1);
    }

    #[test]
    #[should_panic(expected = "Beam width must be in 1..=max_width().")]
    fn beam_search_rejects_invalid_suggested_width() {
        let search = BeamSearch::new(InvalidBeamWidthSuggester, 1);
        let _ = search.run(TestState, vec![]);
    }
}
