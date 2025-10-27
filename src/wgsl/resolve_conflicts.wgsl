// Input size
@group(0) @binding(0) var<storage, read> N: u32;

// Input pixels
@group(0) @binding(1) var<storage, read> pixelsA: array<u32>;
@group(0) @binding(2) var<storage, read> pixelsB: array<u32>;

// Input claims count and priorities
@group(0) @binding(3) var<storage, read_write> claimsCount: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> priorities: array<u32>;

// Output assignments
@group(0) @binding(5) var<storage, read_write> assignments: array<u32>;

fn colorDistance(a: u32, b: u32) -> u32 {
    return u32(abs(i32(a) - i32(b)));
}

@compute @workgroup_size(64)
fn resolveConflicts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    // Out of bounds check
    if(i >= N) { return; }

    var minD: u32 = 0xffffffffu;
    var bestJ: u32 = 0u;

    let prevAssignment = assignments[i];

    if(atomicLoad(&claimsCount[prevAssignment]) == 1u || priorities[i] == 0u) {
        // We are the only one claiming this pixel position
        // Or
        // We have the priority on this pixel position
        return;
    }
            
    // Find another best position with the minimum distance
    for (var j: u32 = 0u; j < N; j++) {

        // Discard the already claimed positions
        if(atomicLoad(&claimsCount[j]) > 0u) {
            continue;
        }

        let d = colorDistance(pixelsA[i], pixelsB[j]);
        if (d < minD) {
            minD = d;
            bestJ = j;
        }
    }

    assignments[i] = bestJ;
}
