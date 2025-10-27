// Input size
@group(0) @binding(0) var<storage, read> N: u32;

// Input pixels
@group(0) @binding(1) var<storage, read> assignments: array<u32>;

// Output claim counts and priorities
@group(0) @binding(2) var<storage, read_write> claim_counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> priorities: array<u32>;

@compute @workgroup_size(64)
fn calculateClaimCountsAndPriorities(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    // Out of bounds check
    if(i >= N) { return; }

    // Increment claim count (atomically)
    let prev = atomicAdd(&claim_counts[assignments[i]], 1u);
    
    // Assign this pixel priority for the claim
    priorities[i] = prev;
}
