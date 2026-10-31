use std::sync::Arc;

use modern_chainer_autograd::Variable;
use modern_chainer_core::{Backend, Device, Result, Rng, Tensor};

use crate::key::StateId;
use crate::spec::StateSpec;
use crate::store::{ScalarValue, StateBuilder, StatePlan, StateStore, StateValue};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunMode {
    Train,
    Eval,
    Inference,
}

impl RunMode {
    pub fn is_inference(self) -> bool {
        matches!(self, RunMode::Inference | RunMode::Eval)
    }

    pub fn is_training(self) -> bool {
        matches!(self, RunMode::Train)
    }
}

/// A model that declares explicit state and advances it one step at a time.
pub trait Step<B: Backend>: Send + Sync {
    type Input;
    type Output;
    type Handles: Clone + Send + Sync;

    fn declare_state(&self, builder: &mut StateBuilder<'_, B>) -> Result<Self::Handles>;

    fn step(
        &self,
        ctx: &mut RunContext<'_, B>,
        handles: &Self::Handles,
        input: Self::Input,
    ) -> Result<Self::Output>;
}

/// Access to the state, mode and RNG of the current step.
pub struct RunContext<'a, B: Backend> {
    state: &'a mut StateStore<B>,
    mode: RunMode,
    step: u64,
    rng: &'a mut Rng,
}

impl<'a, B: Backend> RunContext<'a, B> {
    pub fn state(&self) -> &StateStore<B> {
        self.state
    }

    pub fn state_mut(&mut self) -> &mut StateStore<B> {
        self.state
    }

    pub fn mode(&self) -> RunMode {
        self.mode
    }

    pub fn is_inference(&self) -> bool {
        self.mode.is_inference()
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn rng(&self) -> &Rng {
        self.rng
    }

    pub fn rng_mut(&mut self) -> &mut Rng {
        self.rng
    }

    pub fn tensor(&self, id: StateId) -> Result<Tensor<B>> {
        self.state.tensor(id)
    }

    pub fn set_tensor(&mut self, id: StateId, tensor: Tensor<B>) -> Result<()> {
        self.state.set_tensor(id, tensor)
    }

    pub fn variable(&self, id: StateId) -> Result<Variable<B>> {
        self.state.variable(id)
    }

    pub fn set_variable(&mut self, id: StateId, variable: Variable<B>) -> Result<()> {
        self.state.set_variable(id, variable)
    }

    pub fn scalar(&self, id: StateId) -> Result<ScalarValue> {
        self.state.scalar(id)
    }

    pub fn set_scalar(&mut self, id: StateId, value: ScalarValue) -> Result<()> {
        self.state.set_scalar(id, value)
    }

    pub fn value(&self, id: StateId) -> Result<&StateValue<B>> {
        self.state.value(id)
    }

    pub fn set_value(&mut self, id: StateId, value: StateValue<B>) -> Result<()> {
        self.state.set_value(id, value)
    }
}

#[derive(Clone)]
pub struct RunSnapshot<B: Backend> {
    state: StateStore<B>,
    rng: Rng,
    steps: u64,
    mode: RunMode,
}

#[derive(Clone, Copy, Debug)]
pub struct RunMetrics {
    pub steps: u64,
    pub state_slots: usize,
    pub state_bytes: usize,
    pub mode: RunMode,
}

/// Owns a model, its explicit state, RNG and step counter.
pub struct Run<B: Backend, M: Step<B>> {
    model: Arc<M>,
    handles: M::Handles,
    state: StateStore<B>,
    plan: StatePlan,
    mode: RunMode,
    rng: Rng,
    steps: u64,
}

impl<B: Backend, M: Step<B>> Run<B, M> {
    pub fn new(model: M, backend: B, mode: RunMode, seed: u64) -> Result<Self> {
        let mut state = StateStore::new(backend);
        let handles = {
            let mut builder = StateBuilder::new(&mut state);
            model.declare_state(&mut builder)?
        };
        let plan = state.plan();
        Ok(Self {
            model: Arc::new(model),
            handles,
            state,
            plan,
            mode,
            rng: Rng::seed_from_u64(seed),
            steps: 0,
        })
    }

    pub fn step(&mut self, input: M::Input) -> Result<M::Output> {
        let Run {
            model,
            handles,
            state,
            mode,
            rng,
            steps,
            ..
        } = self;
        let mut ctx = RunContext {
            state,
            mode: *mode,
            step: *steps,
            rng,
        };
        let output = model.step(&mut ctx, handles, input)?;
        *steps += 1;
        Ok(output)
    }

    pub fn fork(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            handles: self.handles.clone(),
            state: self.state.clone(),
            plan: self.plan.clone(),
            mode: self.mode,
            rng: self.rng.clone(),
            steps: self.steps,
        }
    }

    pub fn snapshot(&self) -> RunSnapshot<B> {
        RunSnapshot {
            state: self.state.clone(),
            rng: self.rng.clone(),
            steps: self.steps,
            mode: self.mode,
        }
    }

    pub fn restore(&mut self, snapshot: RunSnapshot<B>) {
        self.state = snapshot.state;
        self.rng = snapshot.rng;
        self.steps = snapshot.steps;
        self.mode = snapshot.mode;
    }

    pub fn transaction(&mut self) -> Transaction<'_, B, M> {
        let snapshot = self.snapshot();
        Transaction {
            run: self,
            snapshot: Some(snapshot),
        }
    }

    pub fn state(&self) -> &StateStore<B> {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut StateStore<B> {
        &mut self.state
    }

    pub fn handles(&self) -> &M::Handles {
        &self.handles
    }

    pub fn plan(&self) -> &StatePlan {
        &self.plan
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn mode(&self) -> RunMode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: RunMode) {
        self.mode = mode;
    }

    pub fn rng(&self) -> &Rng {
        &self.rng
    }

    pub fn rng_mut(&mut self) -> &mut Rng {
        &mut self.rng
    }

    pub fn reseed(&mut self, seed: u64) {
        self.rng = Rng::seed_from_u64(seed);
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn device(&self) -> Device {
        self.state.device()
    }

    pub fn metrics(&self) -> RunMetrics {
        RunMetrics {
            steps: self.steps,
            state_slots: self.state.len(),
            state_bytes: self.state.memory_usage(),
            mode: self.mode,
        }
    }
}

impl<B: Backend, M: Step<B>> Clone for Run<B, M> {
    fn clone(&self) -> Self {
        self.fork()
    }
}

pub struct Transaction<'a, B: Backend, M: Step<B>> {
    run: &'a mut Run<B, M>,
    snapshot: Option<RunSnapshot<B>>,
}

impl<'a, B: Backend, M: Step<B>> Transaction<'a, B, M> {
    /// Access the run inside the transaction.  Changes are rolled back unless
    /// `commit` is called.
    pub fn run_mut(&mut self) -> &mut Run<B, M> {
        self.run
    }

    pub fn commit(mut self) {
        self.snapshot = None;
    }

    pub fn rollback(mut self) {
        if let Some(snapshot) = self.snapshot.take() {
            self.run.restore(snapshot);
        }
    }
}

impl<B: Backend, M: Step<B>> Drop for Transaction<'_, B, M> {
    fn drop(&mut self) {
        if let Some(snapshot) = self.snapshot.take() {
            self.run.restore(snapshot);
        }
    }
}

/// Declare one state slot without going through a model.
pub fn declare_dense<B: Backend>(
    builder: &mut StateBuilder<'_, B>,
    key: impl Into<crate::key::StateKey>,
    shape: impl Into<modern_chainer_core::Shape>,
    dtype: modern_chainer_core::DType,
) -> Result<StateId> {
    builder.declare(key, StateSpec::dense(shape, dtype))
}
