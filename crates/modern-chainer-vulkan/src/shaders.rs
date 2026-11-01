//! WGSL compute kernels.  All kernels use 2-D dispatch to lift the 65535
//! workgroup limit of a single dimension.

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
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
const TILE: u32 = 16u;
var<workgroup> tile: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.y * TILE + lid.y;
    let col = wid.x * TILE + lid.x;
    if (row < params.rows && col < params.cols) {
        tile[lid.y * TILE + lid.x] = input[row * params.cols + col];
    } else {
        tile[lid.y * TILE + lid.x] = 0.0;
    }
    workgroupBarrier();
    let out_row = wid.x * TILE + lid.y;
    let out_col = wid.y * TILE + lid.x;
    if (out_row < params.cols && out_col < params.rows) {
        output[out_row * params.rows + out_col] = tile[lid.x * TILE + lid.y];
    }
}
"#;

pub(crate) const MATMUL: &str = r#"
struct Params { m: u32, k: u32, n: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 32u;
const KTILE: u32 = 32u;
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

    let tiles = (params.k + KTILE - 1u) / KTILE;
    for (var t: u32 = 0u; t < tiles; t = t + 1u) {
        let kt = t * KTILE;
        for (var i: u32 = tid; i < TILE * KTILE; i = i + 256u) {
            let r = i / KTILE;
            let kk = i % KTILE;
            let ar = row0 + r;
            let ak = kt + kk;
            if (ar < params.m && ak < params.k) {
                tile_a[i] = a[ar * params.k + ak];
            } else {
                tile_a[i] = 0.0;
            }
            let bk = kt + i / TILE;
            let bc = col0 + i % TILE;
            if (bk < params.k && bc < params.n) {
                tile_b[i] = b[bk * params.n + bc];
            } else {
                tile_b[i] = 0.0;
            }
        }
        workgroupBarrier();
        for (var j: u32 = 0u; j < KTILE; j = j + 1u) {
            let a0 = tile_a[lid.y * KTILE + j];
            let a1 = tile_a[(lid.y + 16u) * KTILE + j];
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

pub(crate) const MATMUL_TN: &str = r#"
struct Params { m: u32, k: u32, n: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 32u;
const KTILE: u32 = 64u;
var<workgroup> tile_a: array<f32, 2048>;
var<workgroup> tile_b: array<f32, 2048>;

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

    let tiles = (params.k + KTILE - 1u) / KTILE;
    for (var t: u32 = 0u; t < tiles; t = t + 1u) {
        let kt = t * KTILE;
        for (var i: u32 = tid; i < TILE * KTILE; i = i + 256u) {
            let r = i / KTILE;
            let kk = i % KTILE;
            let ar = row0 + r;
            let ak = kt + kk;
            if (ar < params.m && ak < params.k) {
                tile_a[i] = a[ak * params.m + ar];
            } else {
                tile_a[i] = 0.0;
            }
            let bk = kt + i / TILE;
            let bc = col0 + i % TILE;
            if (bk < params.k && bc < params.n) {
                tile_b[i] = b[bk * params.n + bc];
            } else {
                tile_b[i] = 0.0;
            }
        }
        workgroupBarrier();
        for (var j: u32 = 0u; j < KTILE; j = j + 1u) {
            let a0 = tile_a[lid.y * KTILE + j];
            let a1 = tile_a[(lid.y + 16u) * KTILE + j];
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

pub(crate) const MATMUL_NT_BIAS: &str = r#"
struct Params { m: u32, k: u32, n: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

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
    if (row1 < params.m && col1 < params.n) { out[row1 * params.n + col1] = acc00 + bias[col1]; }
    if (row1 < params.m && col2 < params.n) { out[row1 * params.n + col2] = acc01 + bias[col2]; }
    if (row2 < params.m && col1 < params.n) { out[row2 * params.n + col1] = acc10 + bias[col1]; }
    if (row2 < params.m && col2 < params.n) { out[row2 * params.n + col2] = acc11 + bias[col2]; }
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

pub(crate) const SOFTMAX_CE_FORWARD: &str = r#"
struct Params { rows: u32, classes: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> labels: array<u32>;
@group(0) @binding(2) var<storage, read_write> loss: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= params.rows) { return; }
    let base = row * params.classes;
    var maxv = -1e30;
    for (var c: u32 = lid.x; c < params.classes; c = c + 256u) {
        maxv = max(maxv, logits[base + c]);
    }
    scratch[lid.x] = maxv;
    workgroupBarrier();
    for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride) { scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + stride]); }
        workgroupBarrier();
    }
    maxv = scratch[0];
    var sum = 0.0;
    for (var c: u32 = lid.x; c < params.classes; c = c + 256u) {
        sum = sum + exp(logits[base + c] - maxv);
    }
    scratch[lid.x] = sum;
    workgroupBarrier();
    for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        let label = labels[row * 2u];
        loss[row] = ((maxv + log(scratch[0])) - logits[base + label]) / f32(params.rows);
    }
}
"#;

pub(crate) const SOFTMAX_CE_BACKWARD: &str = r#"
struct Params { rows: u32, classes: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> labels: array<u32>;
@group(0) @binding(2) var<storage, read_write> grad: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= params.rows) { return; }
    let base = row * params.classes;
    var maxv = -1e30;
    for (var c: u32 = lid.x; c < params.classes; c = c + 256u) {
        maxv = max(maxv, logits[base + c]);
    }
    scratch[lid.x] = maxv;
    workgroupBarrier();
    for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride) { scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + stride]); }
        workgroupBarrier();
    }
    maxv = scratch[0];
    var sum = 0.0;
    for (var c: u32 = lid.x; c < params.classes; c = c + 256u) {
        sum = sum + exp(logits[base + c] - maxv);
    }
    scratch[lid.x] = sum;
    workgroupBarrier();
    for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride]; }
        workgroupBarrier();
    }
    let inv = 1.0 / scratch[0];
    let label = labels[row * 2u];
    for (var c: u32 = lid.x; c < params.classes; c = c + 256u) {
        var value = exp(logits[base + c] - maxv) * inv;
        if (c == label) { value = value - 1.0; }
        grad[base + c] = value / f32(params.rows);
    }
}
"#;

pub(crate) const GELU: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 4194240u;
    if (i >= arrayLength(&output)) { return; }
    let x = input[i];
    let c = 0.7978845608 * (x + 0.044715 * x * x * x);
    output[i] = 0.5 * x * (1.0 + tanh(c));
}
"#;

pub(crate) const SILU: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 4194240u;
    if (i >= arrayLength(&output)) { return; }
    let x = input[i];
    output[i] = x / (1.0 + exp(-x));
}
"#;

pub(crate) const MATMUL_SMALL: &str = r#"
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
        if (kb < params.k && col < params.n) {
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

pub(crate) const MATMUL_NT_SMALL: &str = r#"
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
            tile_b[lid.y * TILE + lid.x] = b[col * params.k + kb];
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

pub(crate) const RELU4: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= arrayLength(&output)) { return; }
    output[i] = max(input[i], vec4<f32>(0.0));
}
"#;

pub(crate) const SQRT4: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= arrayLength(&output)) { return; }
    output[i] = sqrt(input[i]);
}
"#;

pub(crate) const SCALE4: &str = r#"
struct Params { factor: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= arrayLength(&output)) { return; }
    output[i] = input[i] * params.factor;
}
"#;

pub(crate) const FILL4: &str = r#"
struct Params { value: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= arrayLength(&output)) { return; }
    output[i] = vec4<f32>(params.value);
}
"#;

pub(crate) const ADD_VEC: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= arrayLength(&out)) { return; }
    out[i] = a[i] + b[i];
}
"#;

pub(crate) const SUB_VEC: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= arrayLength(&out)) { return; }
    out[i] = a[i] - b[i];
}
"#;

pub(crate) const MUL_VEC: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= arrayLength(&out)) { return; }
    out[i] = a[i] * b[i];
}
"#;

pub(crate) const DIV_VEC: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= arrayLength(&out)) { return; }
    out[i] = a[i] / b[i];
}
"#;

pub(crate) const SUM_TO_GENERIC: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> info: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_idx = gid.x + gid.y * 4194240u;
    let out_rank = info[0];
    let red_rank = info[1];
    let offset = info[2];
    let red_count = info[3];
    if (out_idx >= info[4]) { return; }
    var rem = out_idx;
    var base = 0u;
    var d = out_rank;
    loop {
        if (d == 0u) { break; }
        d = d - 1u;
        let dim = info[24u + d];
        let coord = rem % dim;
        rem = rem / dim;
        base = base + coord * info[16u + offset + d];
    }
    var sum = 0.0;
    for (var r: u32 = 0u; r < red_count; r = r + 1u) {
        var rr = r;
        var idx = base;
        for (var rd: u32 = 0u; rd < red_rank; rd = rd + 1u) {
            let dim_index = info[32u + rd];
            let size = info[8u + dim_index];
            let coord = rr % size;
            rr = rr / size;
            idx = idx + coord * info[16u + dim_index];
        }
        sum = sum + input[idx];
    }
    output[out_idx] = sum;
}
"#;

pub(crate) const MATMUL_ALIGNED: &str = r#"
struct Params { m: u32, k: u32, n: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 32u;
const KTILE: u32 = 64u;
var<workgroup> tile_a: array<f32, 2048>;
var<workgroup> tile_b: array<f32, 2048>;

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
    for (var kt: u32 = 0u; kt < params.k; kt = kt + KTILE) {
        for (var i: u32 = tid; i < TILE * KTILE; i = i + 256u) {
            let r = i / KTILE;
            let kk = i % KTILE;
            tile_a[i] = a[(row0 + r) * params.k + kt + kk];
            tile_b[i] = b[(kt + i / TILE) * params.n + col0 + i % TILE];
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < KTILE; kk = kk + 1u) {
            let a0 = tile_a[lid.y * KTILE + kk];
            let a1 = tile_a[(lid.y + 16u) * KTILE + kk];
            let b0 = tile_b[kk * TILE + lid.x];
            let b1 = tile_b[kk * TILE + lid.x + 16u];
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
    out[row1 * params.n + col1] = acc00;
    out[row1 * params.n + col2] = acc01;
    out[row2 * params.n + col1] = acc10;
    out[row2 * params.n + col2] = acc11;
}
"#;
