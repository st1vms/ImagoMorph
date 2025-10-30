// Input size
@group(0) @binding(0) var<storage, read> N: u32;

@group(0) @binding(1) var<storage, read_write> usedJ: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> result: array<atomic<u32>>;


@compute @workgroup_size(64)
fn checkSolved(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    // Out of bounds check
    if(i >= N) { return; }

    // Signal if we found an empty cell
    if(atomicLoad(&usedJ[i]) == 0u) {
        atomicStore(&result[0], 0u); // Default result is 1 (Success)
    }
}
