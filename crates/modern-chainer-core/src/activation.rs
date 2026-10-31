/// Activation fused into a linear epilogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Relu,
    Gelu,
    Silu,
}
