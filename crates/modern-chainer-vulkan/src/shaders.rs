//! WGSL compute kernels.  All kernels use 2-D dispatch to lift the 65535
//! workgroup limit of a single dimension.

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

const TILE: u32 = 32u;
var<workgroup> tile_a: array<f32, 1024>;
var<workgroup> tile_b: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row0 = wid.y * TILE;
    let col0 = wid.x * TILE;
    let tid = lid.y * 16u + lid.x;

    var acc00 = 0.0;
    var acc01 = 0.0;
    var acc10 = 0.0;
    var acc11 = 0.0;

    let tiles = (params.k + TILE - 1u) / TILE;
    for (var t: u32 = 0u; t < tiles; t = t + 1u) {
        let kt = t * TILE;
        for (var i: u32 = tid; i < TILE * TILE; i = i + 256u) {
            let r = i / TILE;
            let c = i % TILE;
            let ar = row0 + r;
            let ak = kt + c;
            if (ar < params.m && ak < params.k) {
                tile_a[i] = a[ar * params.k + ak];
            } else {
                tile_a[i] = 0.0;
            }
            let br = kt + r;
            let bc = col0 + c;
            if (br < params.k && bc < params.n) {
                tile_b[i] = b[br * params.n + bc];
            } else {
                tile_b[i] = 0.0;
            }
        }
        workgroupBarrier();
        for (var j: u32 = 0u; j < TILE; j = j + 1u) {
            let a0 = tile_a[lid.y * TILE + j];
            let a1 = tile_a[(lid.y + 16u) * TILE + j];
            let b0 = tile_b[j * TILE + lid.x];
            let b1 = tile_b[j * TILE + lid.x + 16u];
            acc00 = acc00 + a0 * b0;
            acc01 = acc01 + a0 * b1;
            acc10 = acc10 + a1 * b0;
            acc11 = acc11 + a1 * b1;
        }
        workgroupBarrier();
    }

    let row1 = row0 + lid.y;
    let row2 = row1 + 16u;
    let col1 = col0 + lid.x;
    let col2 = col1 + 16u;
    if (row1 < params.m && col1 < params.n) { out[row1 * params.n + col1] = acc00; }
    if (row1 < params.m && col2 < params.n) { out[row1 * params.n + col2] = acc01; }
    if (row2 < params.m && col1 < params.n) { out[row2 * params.n + col1] = acc10; }
    if (row2 < params.m && col2 < params.n) { out[row2 * params.n + col2] = acc11; }
}
"#;

pub(crate) const MATMUL_NT: &str = r#"
struct Params { m: u32, k: u32, n: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 32u;
var<workgroup> tile_a: array<f32, 1024>;
var<workgroup> tile_b: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row0 = wid.y * TILE;
    let col0 = wid.x * TILE;
    let tid = lid.y * 16u + lid.x;

    var acc00 = 0.0;
    var acc01 = 0.0;
    var acc10 = 0.0;
    var acc11 = 0.0;

    let tiles = (params.k + TILE - 1u) / TILE;
    for (var t: u32 = 0u; t < tiles; t = t + 1u) {
        let kt = t * TILE;
        for (var i: u32 = tid; i < TILE * TILE; i = i + 256u) {
            let r = i / TILE;
            let c = i % TILE;
            let ar = row0 + r;
            let ak = kt + c;
            if (ar < params.m && ak < params.k) {
                tile_a[i] = a[ar * params.k + ak];
            } else {
                tile_a[i] = 0.0;
            }
            let br = kt + r;
            let bc = col0 + c;
            if (br < params.k && bc < params.n) {
                tile_b[i] = b[bc * params.k + br];
            } else {
                tile_b[i] = 0.0;
            }
        }
        workgroupBarrier();
        for (var j: u32 = 0u; j < TILE; j = j + 1u) {
            let a0 = tile_a[lid.y * TILE + j];
            let a1 = tile_a[(lid.y + 16u) * TILE + j];
            let b0 = tile_b[j * TILE + lid.x];
            let b1 = tile_b[j * TILE + lid.x + 16u];
            acc00 = acc00 + a0 * b0;
            acc01 = acc01 + a0 * b1;
            acc10 = acc10 + a1 * b0;
            acc11 = acc11 + a1 * b1;
        }
        workgroupBarrier();
    }

    let row1 = row0 + lid.y;
    let row2 = row1 + 16u;
    let col1 = col0 + lid.x;
    let col2 = col1 + 16u;
    if (row1 < params.m && col1 < params.n) { out[row1 * params.n + col1] = acc00; }
    if (row1 < params.m && col2 < params.n) { out[row1 * params.n + col2] = acc01; }
    if (row2 < params.m && col1 < params.n) { out[row2 * params.n + col1] = acc10; }
    if (row2 < params.m && col2 < params.n) { out[row2 * params.n + col2] = acc11; }
}
"#;

pub(crate) const ADAM: &str = r#"
struct Params {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
    n: u32,
    pad: u32,
};
@group(0) @binding(0) var<storage, read> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read> m: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> new_p: array<f32>;
@group(0) @binding(5) var<storage, read_write> new_m: array<f32>;
@group(0) @binding(6) var<storage, read_write> new_v: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 4194240u;
    if (i >= params.n) { return; }
    let g = grad[i];
    let nm = params.beta1 * m[i] + (1.0 - params.beta1) * g;
    let nv = params.beta2 * v[i] + (1.0 - params.beta2) * g * g;
    let mhat = nm / params.bc1;
    let vhat = nv / params.bc2;
    new_m[i] = nm;
    new_v[i] = nv;
    new_p[i] = param[i] - params.lr * mhat / (sqrt(vhat) + params.eps);
}
"#;

pub(crate) const SUM_PARTIAL: &str = r#"
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let start = wid.x * 256u;
    let index = start + lid.x;
    var value = 0.0;
    if (index < params.n) { value = input[index]; }
    scratch[lid.x] = value;
    workgroupBarrier();
    for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) { output[wid.x] = scratch[0]; }
}
"#;

pub(crate) const SUM_FINAL: &str = r#"
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    var value = 0.0;
    for (var i: u32 = lid.x; i < params.n; i = i + 256u) { value = value + input[i]; }
    scratch[lid.x] = value;
    workgroupBarrier();
    for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) { output[0] = scratch[0]; }
}
"#;

pub(crate) const SUM_TO_BIAS: &str = r#"
struct Params { rows: u32, cols: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= params.cols) { return; }
    var sum = 0.0;
    for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
        sum = sum + input[r * params.cols + col];
    }
    output[col] = sum;
}
"#;

pub(crate) const ADD4: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = (gid.x + gid.y * 4194240u) * 4u;
    let n = arrayLength(&out);
    if (base >= n) { return; }
    if (base + 3u < n) {
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            out[base + i] = a[base + i] + b[base + i];
        }
    } else {
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let idx = base + i;
            if (idx < n) { out[idx] = a[idx] + b[idx]; }
        }
    }
}
"#;

pub(crate) const SUB4: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = (gid.x + gid.y * 4194240u) * 4u;
    let n = arrayLength(&out);
    if (base >= n) { return; }
    if (base + 3u < n) {
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            out[base + i] = a[base + i] - b[base + i];
        }
    } else {
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let idx = base + i;
            if (idx < n) { out[idx] = a[idx] - b[idx]; }
        }
    }
}
"#;

pub(crate) const MUL4: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = (gid.x + gid.y * 4194240u) * 4u;
    let n = arrayLength(&out);
    if (base >= n) { return; }
    if (base + 3u < n) {
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            out[base + i] = a[base + i] * b[base + i];
        }
    } else {
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let idx = base + i;
            if (idx < n) { out[idx] = a[idx] * b[idx]; }
        }
    }
}
"#;

pub(crate) const DIV4: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = (gid.x + gid.y * 4194240u) * 4u;
    let n = arrayLength(&out);
    if (base >= n) { return; }
    if (base + 3u < n) {
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            out[base + i] = a[base + i] / b[base + i];
        }
    } else {
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let idx = base + i;
            if (idx < n) { out[idx] = a[idx] / b[idx]; }
        }
    }
}
"#;
