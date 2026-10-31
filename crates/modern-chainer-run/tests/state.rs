use modern_chainer_core::{DType, Result, Tensor};
use modern_chainer_cpu::CpuBackend;
use modern_chainer_run::{
    Run, RunContext, RunMode, ScalarValue, StateBuilder, StateId, StateSpec, Step,
};

struct Counter;

#[derive(Clone)]
struct CounterHandles {
    count: StateId,
    last: StateId,
}

impl Step<CpuBackend> for Counter {
    type Input = i64;
    type Output = i64;
    type Handles = CounterHandles;

    fn declare_state(&self, builder: &mut StateBuilder<'_, CpuBackend>) -> Result<Self::Handles> {
        Ok(CounterHandles {
            count: builder.declare("counter.count", StateSpec::scalar(DType::I64))?,
            last: builder.declare("counter.last", StateSpec::scalar(DType::I64))?,
        })
    }

    fn step(
        &self,
        ctx: &mut RunContext<'_, CpuBackend>,
        handles: &Self::Handles,
        input: Self::Input,
    ) -> Result<Self::Output> {
        let current = ctx.scalar(handles.count)?.as_i64()?;
        let next = current + input;
        ctx.set_scalar(handles.count, ScalarValue::I64(next))?;
        ctx.set_scalar(handles.last, ScalarValue::I64(input))?;
        Ok(next)
    }
}

fn run() -> Run<CpuBackend, Counter> {
    Run::new(Counter, CpuBackend::new(), RunMode::Inference, 7).unwrap()
}

#[test]
fn steps_advance_explicit_state() {
    let mut run = run();
    assert_eq!(run.step(1).unwrap(), 1);
    assert_eq!(run.step(2).unwrap(), 3);
    assert_eq!(run.steps(), 2);
    assert_eq!(run.metrics().state_slots, 2);
    let keys: Vec<String> = run.state().keys().map(|k| k.as_str().to_string()).collect();
    assert!(keys.contains(&"counter.count".to_string()));
    assert!(keys.contains(&"counter.last".to_string()));
}

#[test]
fn fork_isolates_state() {
    let mut parent = run();
    parent.step(1).unwrap();
    let mut draft = parent.fork();
    assert_eq!(draft.step(10).unwrap(), 11);
    assert_eq!(parent.step(1).unwrap(), 2);
    assert_eq!(draft.steps(), 2);
    assert_eq!(parent.steps(), 2);
}

#[test]
fn snapshot_and_restore() {
    let mut run = run();
    run.step(5).unwrap();
    let snapshot = run.snapshot();
    run.step(5).unwrap();
    assert_eq!(run.steps(), 2);
    run.restore(snapshot);
    assert_eq!(run.steps(), 1);
    assert_eq!(run.step(0).unwrap(), 5);
}

#[test]
fn transaction_rolls_back() {
    let mut run = run();
    run.step(1).unwrap();
    {
        let mut tx = run.transaction();
        tx.run_mut().step(100).unwrap();
        tx.rollback();
    }
    assert_eq!(run.step(0).unwrap(), 1);
}

#[test]
fn transaction_drop_rolls_back() {
    let mut run = run();
    run.step(1).unwrap();
    {
        let mut tx = run.transaction();
        tx.run_mut().step(100).unwrap();
    }
    assert_eq!(run.step(0).unwrap(), 1);
}

#[test]
fn transaction_commit_persists() {
    let mut run = run();
    {
        let mut tx = run.transaction();
        tx.run_mut().step(4).unwrap();
        tx.commit();
    }
    assert_eq!(run.step(0).unwrap(), 4);
}

#[test]
fn dense_state_can_grow() {
    let mut store = modern_chainer_run::StateStore::new(CpuBackend::new());
    let id = store
        .declare("dense", StateSpec::dense([2], DType::F32))
        .unwrap();
    let grown = Tensor::from_f32(CpuBackend::new(), &[1.0, 2.0, 3.0], [3]).unwrap();
    store.set_tensor(id, grown).unwrap();
    assert_eq!(store.tensor(id).unwrap().shape().dims(), &[3]);
}

#[test]
fn incompatible_redeclaration_is_rejected() {
    let mut store = modern_chainer_run::StateStore::new(CpuBackend::new());
    store.declare("x", StateSpec::scalar(DType::I64)).unwrap();
    let error = store
        .declare("x", StateSpec::scalar(DType::F32))
        .unwrap_err();
    assert!(error.to_string().contains("incompatible"));
}

#[test]
fn paged_state_allocates_blocks() {
    let mut store = modern_chainer_run::StateStore::new(CpuBackend::new());
    let id = store
        .declare("paged", StateSpec::paged(4, 2, [2, 3], DType::F32))
        .unwrap();
    let paged = store.paged(id).unwrap();
    assert_eq!(paged.blocks.shape().dims(), &[2, 4, 2, 3]);
    assert_eq!(paged.block_table.shape().dims(), &[2]);
    assert_eq!(paged.len, 0);
    assert!(store.memory_usage() > 0);
}
