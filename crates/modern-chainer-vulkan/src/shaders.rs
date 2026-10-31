//! WGSL compute kernels.  All kernels use 2-D dispatch to lift the 65535
//! workgroup limit of a single dimension.

pub(crate) const ADD: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 4194240u;
    if (i >= arrayLength(&out)) { return; }
    out[i] = a[i] + b[i];
}
"#;

pub(crate) const SUB: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 4194240u;
    if (i >= arrayLength(&out)) { return; }
    out[i] = a[i] - b[i];
}
"#;

pub(crate) const MUL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 4194240u;
    if (i >= arrayLength(&out)) { return; }
    out[i] = a[i] * b[i];
}
"#;

pub(crate) const DIV: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 4194240u;
    if (i >= arrayLength(&out)) { return; }
    out[i] = a[i] / b[i];
}
"#;

pub(crate) const RELU: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 4194240u;
    if (i >= arrayLength(&out)) { return; }
    out[i] = max(a[i], 0.0);
}
"#;

pub(crate) const RELU_BACKWARD: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 4194240u;
    if (i >= arrayLength(&out)) { return; }
    out[i] = select(0.0, g[i], x[i] > 0.0);
}
"#;

pub(crate) const SQRT: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 4194240u;
    if (i >= arrayLength(&out)) { return; }
    out[i] = sqrt(a[i]);
}
"#;

pub(crate) const SCALE: &str = r#"
struct Params { factor: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 4194240u;
    if (i >= arrayLength(&out)) { return; }
    out[i] = a[i] * params.factor;
}
"#;

pub(crate) const FILL: &str = r#"
struct Params { value: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 4194240u;
    if (i >= arrayLength(&out)) { return; }
    out[i] = params.value;
}
"#;

pub(crate) const BIAS_ADD: &str = r#"
struct Params { cols: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 4194240u;
    let n = arrayLength(&out);
    if (i >= n) { return; }
    out[i] = a[i] + bias[i % params.cols];
}
"#;

pub(crate) const TRANSPOSE: &str = r#"
struct Params { rows: u32, cols: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 4194240u;
    let n = arrayLength(&out);
    if (i >= n) { return; }
    let row = i / params.cols;
    let col = i % params.cols;
    out[col * params.rows + row] = a[i];
}
"#;

pub(crate) const MATMUL: &str = r#"
struct Params { m: u32, k: u32, n: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 16u;
var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.y * TILE + lid.y;
    let col = wid.x * TILE + lid.x;
    var acc = 0.0;

    let tiles = (params.k + TILE - 1u) / TILE;
    for (var t: u32 = 0u; t < tiles; t = t + 1u) {
        let ka = t * TILE + lid.x;
        let kb = t * TILE + lid.y;

        if (row < params.m && ka < params.k) {
            tile_a[lid.y * TILE + lid.x] = a[row * params.k + ka];
        } else {
            tile_a[lid.y * TILE + lid.x] = 0.0;
        }
        if (col < params.n && kb < params.k) {
            tile_b[lid.y * TILE + lid.x] = b[kb * params.n + col];
        } else {
            tile_b[lid.y * TILE + lid.x] = 0.0;
        }
        workgroupBarrier();

        for (var j: u32 = 0u; j < TILE; j = j + 1u) {
            acc = acc + tile_a[lid.y * TILE + j] * tile_b[j * TILE + lid.x];
        }
        workgroupBarrier();
    }

    if (row < params.m && col < params.n) {
        out[row * params.n + col] = acc;
    }
}
"#;
