// Input size
@group(0) @binding(0) var<storage, read> N: u32;

@group(0) @binding(1) var<storage, read> claimsCount: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: array<atomic<u32>>;


@compute @workgroup_size(64)
fn checkConflicts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    // Out of bounds check
    if(i >= N) { return; }

    // Signal if we found a conflict
    if(claimsCount[i] != 1u) {
        atomicStore(&result[0], 0u);
    }
}
