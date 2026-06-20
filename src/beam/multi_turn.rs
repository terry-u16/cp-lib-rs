use super::common::{BeamError, BeamWidthSuggester, MaxCostIndex, NoCandidatesError};
use ac_library::Segtree;
use num_traits::bounds::LowerBounded;
use rustc_hash::FxHashMap;
use std::{
    collections::{VecDeque, hash_map::Entry},
    fmt::{Debug, Display},
    hash::Hash,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Index, IndexMut},
    time::Instant,
};

struct ObjectPool<T> {
    data: Vec<MaybeUninit<T>>,
    vacant_indices: Vec<usize>,
}

impl<T> ObjectPool<T> {
    fn new() -> Self {
        Self {
            data: Vec::with_capacity(1024),
            vacant_indices: Vec::with_capacity(1024),
        }
    }

    fn push(&mut self, value: T) -> ObjectPoolIndex {
        let index = if let Some(index) = self.vacant_indices.pop() {
            index
        } else {
            let index = self.data.len();
            self.data.push(MaybeUninit::uninit());
            index
        };

        self.data[index].write(value);
        ObjectPoolIndex(index as u32)
    }

    fn remove(&mut self, index: ObjectPoolIndex) {
        let index = index.0 as usize;
        unsafe {
            self.data[index].assume_init_drop();
        }
        self.vacant_indices.push(index);
    }
}

impl<T> Drop for ObjectPool<T> {
    fn drop(&mut self) {
        self.vacant_indices.sort_unstable();
        let mut vacant_index = 0;

        for (i, data) in self.data.iter_mut().enumerate() {
            if self.vacant_indices.get(vacant_index) == Some(&i) {
                // 二重dropにならないようスキップする
                // 空indexの位置を進める
                vacant_index += 1;
            } else {
                unsafe { data.assume_init_drop() };
            }
        }
    }
}

impl<T> Index<ObjectPoolIndex> for ObjectPool<T> {
    type Output = T;

    fn index(&self, index: ObjectPoolIndex) -> &Self::Output {
        let index = index.0 as usize;
        unsafe { self.data[index].assume_init_ref() }
    }
}

impl<T> IndexMut<ObjectPoolIndex> for ObjectPool<T> {
    fn index_mut(&mut self, index: ObjectPoolIndex) -> &mut Self::Output {
        let index = index.0 as usize;
        unsafe { self.data[index].assume_init_mut() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ObjectPoolIndex(u32);

impl ObjectPoolIndex {
    const NONE: Self = Self(!0);
}

/// 状態遷移を行うために必要な情報
/// メモリ使用量をできるだけ小さくしてください
pub trait Action: Clone + Eq + Default {
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
    parent_id: ObjectPoolIndex,
}

impl<A: Action, E: Evaluator, H: BeamHash> Candidate<A, E, H> {
    #[inline]
    fn new(action: A, evaluator: E, hash: H, parent_id: ObjectPoolIndex) -> Self {
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
        turn_step: usize,
    );
}

struct CandidateSetImpl<'a, A: Action, E: Evaluator, H: BeamHash> {
    selectors: &'a mut MultiSelector<A, E, H>,
    parent_id: ObjectPoolIndex,
}

impl<'a, A: Action, E: Evaluator, H: BeamHash> CandidateSetImpl<'a, A, E, H> {
    fn new(selectors: &'a mut MultiSelector<A, E, H>, parent_id: ObjectPoolIndex) -> Self {
        Self {
            selectors,
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
        turn_step: usize,
    ) {
        assert!(turn_step > 0, "Turn step must be greater than 0.");
        self.selectors.push(
            Candidate::new(action, evaluator, hash, self.parent_id),
            is_finished,
            turn_step,
        );
    }
}

/// ノードの候補から実際に追加するものを選ぶ構造体
/// ビーム幅の個数だけ、評価がよいものを選ぶ
/// ハッシュ値が一致したものについては、評価がよいほうのみを残す
struct NodeSelector<A: Action, E: Evaluator, H: BeamHash> {
    /// 実際に次のターンへ渡す候補数の上限。
    beam_width: usize,
    /// セグ木上で保持する候補容量。beam_width以上の2冪を保つ。
    ///
    /// beam_widthの縮小に合わせてセグ木を縮めると、可変ビーム幅で毎ターン再構築が
    /// 発生しやすい。そのため、Vecのcapacityと同じように内部容量は広めに維持し、
    /// 実際に候補を列挙する直前だけbeam_width個に絞る。
    segtree_width: usize,
    candidates: Vec<Candidate<A, E, H>>,
    hash_to_index: FxHashMap<H, usize>,
    costs: Vec<(E::Cost, usize)>,
    /// セグメント木を削除可能な優先度付きキューとして使う
    cost_segtree: Option<Segtree<MaxCostIndex<E::Cost>>>,
    finished_candidates: Vec<Candidate<A, E, H>>,
}

impl<A: Action, E: Evaluator, H: BeamHash> NodeSelector<A, E, H> {
    fn new(beam_width: usize) -> Self {
        let segtree_width = beam_width.next_power_of_two();
        let candidates = Vec::with_capacity(segtree_width);
        let costs = Vec::with_capacity(segtree_width);

        Self {
            beam_width,
            segtree_width,
            candidates,
            hash_to_index: FxHashMap::default(),
            costs,
            cost_segtree: None,
            finished_candidates: Vec::new(),
        }
    }

    fn push(&mut self, candidate: Candidate<A, E, H>, is_finished: bool) -> bool {
        if is_finished {
            self.finished_candidates.push(candidate);
            return true;
        }

        let cost = candidate.evaluator.evaluate();

        // 保持しているどの候補よりもコストが小さくないとき
        if let Some(segtree) = &self.cost_segtree {
            if self.candidates.len() >= self.segtree_width && cost >= segtree.all_prod().0 {
                return false;
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
                                return true;
                            }
                        }
                        None => {
                            if cost < self.costs[index_c].0 {
                                self.candidates[index_c] = candidate;
                                self.costs[index_c] = (cost, index_c);
                                return true;
                            }
                        }
                    }

                    return false;
                }
            }
            // ハッシュ値が等しいものが存在しない場合
            Entry::Vacant(entry) => {
                // セグ木が構築されているかで場合分け
                match &mut self.cost_segtree {
                    Some(segtree) => {
                        if self.candidates.len() < self.segtree_width {
                            let index_c = self.candidates.len();
                            entry.insert(index_c);
                            self.candidates.push(candidate);
                            segtree.set(index_c, (cost, index_c));
                        } else {
                            let index_c = segtree.all_prod().1;
                            let old_hash = self.candidates[index_c].hash;
                            entry.insert(index_c);
                            self.hash_to_index.remove(&old_hash);
                            self.candidates[index_c] = candidate;
                            segtree.set(index_c, (cost, index_c));
                        }
                    }
                    None => {
                        let index_c = self.candidates.len();
                        entry.insert(index_c);
                        self.candidates.push(candidate);
                        self.costs.push((cost, index_c));

                        // 保持している候補が内部容量分になったときにセグ木を構築する
                        if self.costs.len() >= self.segtree_width {
                            self.rebuild_segtree();
                        }
                    }
                }
            }
        }

        true
    }

    fn iter_candidates_and_clear<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = Candidate<A, E, H>> + 'a {
        // セグ木容量はbeam_widthより大きいことがあるため、木へ追加する直前にだけ
        // 真のビーム幅まで候補を絞る。ホットパスのpushや幅変更時には縮小しない。
        if self.candidates.len() > self.beam_width {
            self.candidates
                .select_nth_unstable_by_key(self.beam_width, |candidate| {
                    candidate.evaluator.evaluate()
                });
            self.candidates.truncate(self.beam_width);
        }

        self.costs.clear();
        self.hash_to_index.clear();
        self.cost_segtree = None;

        self.candidates.drain(..)
    }

    fn rebuild_segtree(&mut self) {
        let mut costs =
            vec![<MaxCostIndex<E::Cost> as ac_library::Monoid>::identity(); self.segtree_width];
        for (index, candidate) in self.candidates.iter().enumerate() {
            costs[index] = (candidate.evaluator.evaluate(), index);
        }

        self.costs.clear();
        self.cost_segtree = Some(Segtree::from(costs));
    }

    fn calculate_best_candidate(&self) -> Option<&Candidate<A, E, H>> {
        self.candidates
            .iter()
            .min_by_key(|candidate| candidate.evaluator.evaluate())
    }

    pub(super) fn set_beam_width(&mut self, beam_width: usize) {
        if self.beam_width == beam_width {
            return;
        }

        // 縮小時は内部容量を保ち、拡大時も現在の2冪容量に収まるなら再構築しない。
        // 容量を超えたときだけ次の2冪へ拡張して、セグ木の深さを安定させる。
        if beam_width > self.segtree_width {
            self.segtree_width = beam_width.next_power_of_two();
            self.candidates.reserve(
                self.segtree_width
                    .saturating_sub(self.candidates.capacity()),
            );
            self.costs
                .reserve(self.segtree_width.saturating_sub(self.costs.capacity()));

            if self.cost_segtree.is_some() {
                self.rebuild_segtree();
            }
        }

        self.beam_width = beam_width;
    }
}

struct MultiSelector<A: Action, E: Evaluator, H: BeamHash> {
    max_step: usize,
    beam_width: usize,
    selectors: VecDeque<NodeSelector<A, E, H>>,
}

impl<A: Action, E: Evaluator, H: BeamHash> MultiSelector<A, E, H> {
    fn new(max_beam_width: usize) -> Self {
        let selectors = VecDeque::new();

        Self {
            max_step: 1,
            beam_width: max_beam_width,
            selectors,
        }
    }

    fn push(&mut self, candidate: Candidate<A, E, H>, is_finished: bool, step: usize) {
        // single_turnではselectorを毎ターン空にしてすぐ使い切るため、幅変更は
        // 次のターンのpush前に反映すれば十分。
        //
        // multi_turnではturn_step>1の候補を未来ターン用selectorへ積むため、
        // selectorが複数ターンにまたがって残る。可変ビーム幅では、その間に
        // 真のbeam_widthだけが変わり得るので、内部セグ木容量を別に持って
        // 必要なときだけ拡張する。
        while self.selectors.len() < step + 1 {
            self.selectors.push_back(NodeSelector::new(self.beam_width));
        }

        if self.selectors[step - 1].push(candidate, is_finished) {
            self.max_step = self.max_step.max(step);
        }
    }

    fn reset_step_max(&mut self) {
        self.max_step = 1;
    }

    fn pop_selector(&mut self) -> NodeSelector<A, E, H> {
        self.selectors.pop_front().expect("No selector to pop.")
    }

    /// selectorを使い回す
    fn push_selector(&mut self, selector: NodeSelector<A, E, H>) {
        self.selectors.push_back(selector);
    }

    fn set_beam_width(&mut self, beam_width: usize) {
        self.beam_width = beam_width;

        for selector in &mut self.selectors {
            selector.set_beam_width(beam_width);
        }
    }
}

/// 二重連鎖木のノード
struct Node<A: Action, E: Evaluator, H: BeamHash> {
    action: A,
    evaluator: E,
    hash: H,
    parent: ObjectPoolIndex,
    child: ObjectPoolIndex,
    left: ObjectPoolIndex,
    right: ObjectPoolIndex,
    is_active: bool,
    /// remove_queueに残っており、未来selectorからparent_idとして参照される可能性がある。
    is_removal_reserved: bool,
}

impl<A: Action, E: Evaluator, H: BeamHash> Node<A, E, H> {
    /// 通常のコンストラクタ
    fn new(candidate: Candidate<A, E, H>, right: ObjectPoolIndex) -> Self {
        let Candidate {
            action,
            evaluator,
            hash,
            parent_id,
            ..
        } = candidate;

        Self {
            action,
            evaluator,
            hash,
            parent: parent_id,
            child: ObjectPoolIndex::NONE,
            left: ObjectPoolIndex::NONE,
            right,
            is_active: true,
            is_removal_reserved: false,
        }
    }

    /// ルートノードのコンストラクタ
    fn new_root(action: A, evaluator: E, hash: H) -> Self {
        Self {
            action,
            evaluator,
            hash,
            parent: ObjectPoolIndex::NONE,
            child: ObjectPoolIndex::NONE,
            left: ObjectPoolIndex::NONE,
            right: ObjectPoolIndex::NONE,
            is_active: true,
            is_removal_reserved: false,
        }
    }
}

struct BeamTree<S: State> {
    state: S,
    nodes: ObjectPool<Node<S::Action, S::Evaluator, S::Hash>>,
    root: ObjectPoolIndex,
    remove_queue: VecDeque<Vec<ObjectPoolIndex>>,
}

impl<S: State> BeamTree<S> {
    fn new(state: S, root: Node<S::Action, S::Evaluator, S::Hash>) -> Self {
        let mut nodes = ObjectPool::new();
        let root = nodes.push(root);
        let remove_queue = VecDeque::new();

        Self {
            state,
            nodes,
            root,
            remove_queue,
        }
    }

    /// 状態を更新しながら深さ優先探索を行い、次のノードの候補を全てselectorに追加する
    fn dfs(
        &mut self,
        selectors: &mut MultiSelector<S::Action, S::Evaluator, S::Hash>,
        turn: usize,
    ) -> Result<(), BeamError> {
        self.remove_useless_nodes(turn)?;
        self.update_root();

        let mut v = self.root;

        if !self.nodes[v].is_active {
            // activeなノードがない場合
            return Ok(());
        }

        loop {
            v = self.move_to_leaf(v);

            selectors.reset_step_max();
            let node = &self.nodes[v];

            let mut cand_set = CandidateSetImpl::new(selectors, v);
            self.state.expand(&node.evaluator, node.hash, &mut cand_set);

            while self.remove_queue.len() < selectors.max_step {
                self.remove_queue.push_back(vec![]);
            }

            self.nodes[v].is_removal_reserved = true;
            self.remove_queue[selectors.max_step - 1].push(v);

            v = self.move_to_ancestor(v);

            if v == self.root {
                break;
            }
        }

        Ok(())
    }

    /// 根からノードvまでのパスを取得する
    fn restore_path(&self, mut v: ObjectPoolIndex) -> Vec<S::Action> {
        let mut path = vec![];

        while self.nodes[v].parent != ObjectPoolIndex::NONE {
            path.push(self.nodes[v].action.clone());
            v = self.nodes[v].parent;
        }

        path.reverse();
        path
    }

    /// 新しいノードを追加する
    fn add_leaf(
        &mut self,
        candidate: Candidate<S::Action, S::Evaluator, S::Hash>,
    ) -> ObjectPoolIndex {
        let parent = candidate.parent_id;
        let sibling = self.nodes[parent].child;
        let v = self.nodes.push(Node::new(candidate, sibling));

        self.nodes[parent].child = v;

        if sibling != ObjectPoolIndex::NONE {
            self.nodes[sibling].left = v;
        }

        // 祖先をactivateする
        let mut u = parent;

        while !self.nodes[u].is_active {
            self.nodes[u].is_active = true;

            if u == self.root {
                break;
            }

            u = self.nodes[u].parent;
        }

        v
    }

    /// 根から一本道の部分は往復しないようにする
    fn update_root(&mut self) {
        let mut child = self.nodes[self.root].child;

        while !self.nodes[self.root].is_removal_reserved
            && child != ObjectPoolIndex::NONE
            && self.nodes[child].right == ObjectPoolIndex::NONE
        {
            self.root = child;
            self.nodes[child].action.apply(&mut self.state);
            child = self.nodes[child].child;
        }
    }

    /// ノードvの子孫で、最も左にある葉に移動する
    fn move_to_leaf(&mut self, mut v: ObjectPoolIndex) -> ObjectPoolIndex {
        let mut child = self.nodes[v].child;

        while child != ObjectPoolIndex::NONE {
            // activeなノードが見つかるまで右に移動
            while !self.nodes[child].is_active {
                child = self.nodes[child].right;
            }

            self.nodes[v].is_active = false;
            v = child;
            let c = &self.nodes[v];
            c.action.apply(&mut self.state);
            child = c.child;
        }

        self.nodes[v].is_active = false;
        v
    }

    /// ノードvの先祖で、右への分岐があるところまで移動する
    fn move_to_ancestor(&mut self, mut v: ObjectPoolIndex) -> ObjectPoolIndex {
        while v != self.root {
            self.nodes[v].action.rollback(&mut self.state);

            // activeなノードが見つかるまで右に移動する
            let mut u = self.nodes[v].right;
            while u != ObjectPoolIndex::NONE {
                if self.nodes[u].is_active {
                    self.nodes[u].action.apply(&mut self.state);
                    return u;
                }

                u = self.nodes[u].right;
            }

            v = self.nodes[v].parent;
        }

        assert_eq!(v, self.root);
        self.root
    }

    /// 不要になったノードを全て削除する
    fn remove_useless_nodes(&mut self, turn: usize) -> Result<(), BeamError> {
        if self.remove_queue.is_empty() {
            return Ok(());
        }

        let mut remove_queue = self.remove_queue.pop_front().unwrap();

        for v in remove_queue.drain(..) {
            self.nodes[v].is_removal_reserved = false;

            if self.nodes[v].child == ObjectPoolIndex::NONE {
                // 子がいないので消してOK
                self.remove_leaf(v, turn)?;
            }
        }

        // キューを再利用する
        self.remove_queue.push_back(remove_queue);

        Ok(())
    }

    /// 不要になった葉を再帰的に削除する
    fn remove_leaf(&mut self, mut v: ObjectPoolIndex, turn: usize) -> Result<(), BeamError> {
        loop {
            // 予約中の祖先は、まだ実体化していない未来候補の親になり得る。
            if self.nodes[v].is_removal_reserved {
                return Ok(());
            }

            let left = self.nodes[v].left;
            let right = self.nodes[v].right;

            match left {
                ObjectPoolIndex::NONE => {
                    let parent = self.nodes[v].parent;

                    if parent == ObjectPoolIndex::NONE {
                        // ルートノードが削除される = 有効手が存在しない
                        return Err(BeamError::NoCandidates(NoCandidatesError::new(turn)));
                    }

                    self.nodes.remove(v);
                    self.nodes[parent].child = right;

                    if right != ObjectPoolIndex::NONE {
                        self.nodes[right].left = ObjectPoolIndex::NONE;
                        return Ok(());
                    }

                    v = parent;
                }
                _ => {
                    self.nodes.remove(v);
                    self.nodes[left].right = right;

                    if right != ObjectPoolIndex::NONE {
                        self.nodes[right].left = left;
                    }

                    return Ok(());
                }
            }
        }
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
        let (evaluator, hash) = state.make_initial_node();
        let mut tree = BeamTree::new(state, Node::new_root(S::Action::default(), evaluator, hash));
        let mut selectors = MultiSelector::new(self.beam_width_suggester.max_width());

        for turn in 0..self.max_turn {
            let beam_width = self.beam_width_suggester.suggest();
            assert!(
                beam_width > 0 && beam_width <= self.beam_width_suggester.max_width(),
                "Beam width must be in 1..=max_width()."
            );

            for callback in &mut callbacks {
                callback.on_turn_start(turn, beam_width);
            }

            selectors.set_beam_width(beam_width);

            // Euler Tourでselectorに候補を追加する
            tree.dfs(&mut selectors, turn)?;

            let mut selector = selectors.pop_selector();
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

            // ターン数固定型の問題で全ターンが終了したとき
            if turn == self.max_turn - 1 {
                return match selector.calculate_best_candidate() {
                    Some(candidate) => {
                        let mut actions = tree.restore_path(candidate.parent_id);
                        actions.push(candidate.action.clone());
                        Ok(actions)
                    }
                    None => Err(BeamError::NoCandidates(NoCandidatesError::new(turn))),
                };
            }

            // 新しいノードを追加する
            for candidate in selector.iter_candidates_and_clear() {
                tree.add_leaf(candidate);
            }

            // selectorを使い回す
            selectors.push_selector(selector);

            for callback in &mut callbacks {
                callback.on_turn_end(turn);
            }
        }

        unreachable!();
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
            Some(best_candidate) => {
                eprintln!("best score = {}", best_candidate.evaluator.evaluate());
            }
            None => {
                eprintln!("best score = (no candidates)");
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
    use crate::beam::common::{BeamWidthSuggester, FixedBeamWidthSuggester};

    #[derive(Clone, Default, Eq, PartialEq, Debug)]
    enum TestAction {
        #[default]
        Root,
        A,
        B,
        Goal,
    }

    #[derive(Clone, Default, Eq, PartialEq, Debug)]
    enum VariableWidthAction {
        #[default]
        Root,
        A,
        B,
        Goal,
    }

    #[derive(Clone, Default, Eq, PartialEq, Debug)]
    enum PendingFutureAction {
        #[default]
        Root,
        Enter,
        ShortDeadEnd,
        FutureChild,
        FutureGoal,
        Finish,
    }

    struct TestState;

    struct VariableWidthState;

    struct PendingFutureState {
        mode: PendingFutureMode,
    }

    enum PendingFutureMode {
        FinishedCandidate,
        NonFinishedCandidate,
    }

    #[derive(Clone)]
    struct TestEvaluator(i32);

    struct VariableBeamWidthSuggester {
        widths: Vec<usize>,
        index: usize,
        max_width: usize,
    }

    struct InvalidBeamWidthSuggester;

    impl VariableBeamWidthSuggester {
        fn new(widths: Vec<usize>, max_width: usize) -> Self {
            Self {
                widths,
                index: 0,
                max_width,
            }
        }
    }

    impl BeamWidthSuggester for VariableBeamWidthSuggester {
        fn suggest(&mut self) -> usize {
            let width = self.widths[self.index];
            self.index += 1;
            width
        }

        fn max_width(&self) -> usize {
            self.max_width
        }
    }

    impl BeamWidthSuggester for InvalidBeamWidthSuggester {
        fn suggest(&mut self) -> usize {
            0
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

    impl Action for VariableWidthAction {
        type State = VariableWidthState;

        fn apply(&self, _state: &mut Self::State) {}

        fn rollback(&self, _state: &mut Self::State) {}
    }

    impl Action for PendingFutureAction {
        type State = PendingFutureState;

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
            evaluator: &Self::Evaluator,
            _hash: Self::Hash,
            candidate_set: &mut impl CandidateSet<
                Action = Self::Action,
                Evaluator = Self::Evaluator,
                Hash = Self::Hash,
            >,
        ) {
            match evaluator.0 {
                0 => {
                    candidate_set.push(TestAction::A, TestEvaluator(1), 1, false, 1);
                    candidate_set.push(TestAction::B, TestEvaluator(2), 2, false, 1);
                }
                1 => {
                    candidate_set.push(TestAction::Goal, TestEvaluator(-1), 3, true, 1);
                }
                _ => {}
            }
        }
    }

    impl State for VariableWidthState {
        type Evaluator = TestEvaluator;
        type Hash = u64;
        type Action = VariableWidthAction;

        fn make_initial_node(&self) -> (Self::Evaluator, Self::Hash) {
            (TestEvaluator(0), 0)
        }

        fn expand(
            &mut self,
            evaluator: &Self::Evaluator,
            _hash: Self::Hash,
            candidate_set: &mut impl CandidateSet<
                Action = Self::Action,
                Evaluator = Self::Evaluator,
                Hash = Self::Hash,
            >,
        ) {
            match evaluator.0 {
                0 => {
                    candidate_set.push(VariableWidthAction::A, TestEvaluator(1), 1, false, 1);
                    candidate_set.push(VariableWidthAction::B, TestEvaluator(2), 2, false, 1);
                }
                2 => {
                    candidate_set.push(VariableWidthAction::Goal, TestEvaluator(-1), 3, true, 1);
                }
                _ => {}
            }
        }
    }

    impl State for PendingFutureState {
        type Evaluator = TestEvaluator;
        type Hash = u64;
        type Action = PendingFutureAction;

        fn make_initial_node(&self) -> (Self::Evaluator, Self::Hash) {
            (TestEvaluator(0), 0)
        }

        fn expand(
            &mut self,
            evaluator: &Self::Evaluator,
            _hash: Self::Hash,
            candidate_set: &mut impl CandidateSet<
                Action = Self::Action,
                Evaluator = Self::Evaluator,
                Hash = Self::Hash,
            >,
        ) {
            match (evaluator.0, &self.mode) {
                (0, PendingFutureMode::FinishedCandidate) => {
                    candidate_set.push(
                        PendingFutureAction::ShortDeadEnd,
                        TestEvaluator(1),
                        1,
                        false,
                        1,
                    );
                    candidate_set.push(
                        PendingFutureAction::FutureGoal,
                        TestEvaluator(-1),
                        2,
                        true,
                        3,
                    );
                }
                (0, PendingFutureMode::NonFinishedCandidate) => {
                    candidate_set.push(PendingFutureAction::Enter, TestEvaluator(1), 1, false, 1);
                }
                (1, PendingFutureMode::NonFinishedCandidate) => {
                    candidate_set.push(
                        PendingFutureAction::ShortDeadEnd,
                        TestEvaluator(2),
                        2,
                        false,
                        1,
                    );
                    candidate_set.push(
                        PendingFutureAction::FutureChild,
                        TestEvaluator(3),
                        3,
                        false,
                        3,
                    );
                }
                (3, PendingFutureMode::NonFinishedCandidate) => {
                    candidate_set.push(PendingFutureAction::Finish, TestEvaluator(-1), 4, true, 1);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn node_selector_removes_stale_hash_when_replacing_worst_candidate() {
        let mut selector: NodeSelector<TestAction, TestEvaluator, u64> = NodeSelector::new(2);

        selector.push(
            Candidate::new(
                TestAction::Root,
                TestEvaluator(10),
                1,
                ObjectPoolIndex::NONE,
            ),
            false,
        );
        selector.push(
            Candidate::new(
                TestAction::Root,
                TestEvaluator(20),
                2,
                ObjectPoolIndex::NONE,
            ),
            false,
        );
        selector.push(
            Candidate::new(TestAction::Root, TestEvaluator(5), 3, ObjectPoolIndex::NONE),
            false,
        );
        selector.push(
            Candidate::new(TestAction::Root, TestEvaluator(1), 2, ObjectPoolIndex::NONE),
            false,
        );

        let best = selector.calculate_best_candidate().unwrap();
        assert_eq!(*best.hash(), 2);
        assert_eq!(best.evaluator().evaluate(), 1);
    }

    #[test]
    fn beam_tree_keeps_root_in_place_when_it_has_multiple_children() {
        let search = BeamSearch::new(FixedBeamWidthSuggester::new(2), 2);
        let result = search.run(TestState, vec![]).unwrap();

        assert_eq!(result, vec![TestAction::A, TestAction::Goal]);
    }

    #[test]
    fn multi_turn_respects_shrunk_beam_width_for_new_selectors() {
        let search = BeamSearch::new(VariableBeamWidthSuggester::new(vec![1, 1], 2), 2);
        let result = search.run(VariableWidthState, vec![]);

        assert!(result.is_err());
    }

    #[test]
    fn pending_finished_candidate_keeps_parent_alive_until_it_matures() {
        let search = BeamSearch::new(FixedBeamWidthSuggester::new(2), 3);
        let result = search
            .run(
                PendingFutureState {
                    mode: PendingFutureMode::FinishedCandidate,
                },
                vec![],
            )
            .unwrap();

        assert_eq!(result, vec![PendingFutureAction::FutureGoal]);
    }

    #[test]
    fn pending_non_finished_candidate_can_be_added_after_short_sibling_dies() {
        let search = BeamSearch::new(FixedBeamWidthSuggester::new(2), 5);
        let result = search
            .run(
                PendingFutureState {
                    mode: PendingFutureMode::NonFinishedCandidate,
                },
                vec![],
            )
            .unwrap();

        assert_eq!(
            result,
            vec![
                PendingFutureAction::Enter,
                PendingFutureAction::FutureChild,
                PendingFutureAction::Finish
            ]
        );
    }

    #[test]
    fn node_selector_keeps_power_of_two_capacity_and_prunes_on_drain() {
        let mut selector: NodeSelector<TestAction, TestEvaluator, u64> = NodeSelector::new(3);
        assert_eq!(selector.segtree_width, 4);

        for (cost, hash) in [(10, 10), (20, 20), (30, 30), (40, 40)] {
            selector.push(
                Candidate::new(
                    TestAction::Root,
                    TestEvaluator(cost),
                    hash,
                    ObjectPoolIndex::NONE,
                ),
                false,
            );
        }

        selector.set_beam_width(5);
        assert_eq!(selector.segtree_width, 8);

        selector.push(
            Candidate::new(
                TestAction::Root,
                TestEvaluator(0),
                50,
                ObjectPoolIndex::NONE,
            ),
            false,
        );
        assert_eq!(selector.candidates.len(), 5);

        selector.set_beam_width(2);
        assert_eq!(selector.segtree_width, 8);
        assert_eq!(selector.candidates.len(), 5);

        let mut costs = selector
            .iter_candidates_and_clear()
            .map(|candidate| candidate.evaluator().evaluate())
            .collect::<Vec<_>>();
        costs.sort();

        assert_eq!(costs, vec![0, 10]);
    }

    #[test]
    #[should_panic(expected = "Beam width must be in 1..=max_width().")]
    fn beam_search_rejects_invalid_suggested_width() {
        let search = BeamSearch::new(InvalidBeamWidthSuggester, 1);
        let _ = search.run(TestState, vec![]);
    }
}
