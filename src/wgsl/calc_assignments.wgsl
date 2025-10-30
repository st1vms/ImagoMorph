// Input size
@group(0) @binding(0) var<storage, read> N: u32;

// Input pixels
@group(0) @binding(1) var<storage, read> pixelsA: array<u32>;
@group(0) @binding(2) var<storage, read> pixelsB: array<u32>;

// Used positions arrays
@group(0) @binding(3) var<storage, read_write> usedJ: array<atomic<u32>>;

// Output assignments
@group(0) @binding(4) var<storage, read_write> assignments: array<u32>;

fn colorDistance(a: u32, b: u32) -> f32 {
    // Extract RGBA components from packed u32 (RGBA format: R=bits 24-31, G=16-23, B=8-15, A=0-7)
    let r1 = f32((a >> 24u) & 0xFFu);
    let g1 = f32((a >> 16u) & 0xFFu);
    let b1 = f32((a >> 8u) & 0xFFu);
    let a1 = f32(a & 0xFFu);
    
    let r2 = f32((b >> 24u) & 0xFFu);
    let g2 = f32((b >> 16u) & 0xFFu);
    let b2 = f32((b >> 8u) & 0xFFu);
    let a2 = f32(b & 0xFFu);
    
    // Calculate Euclidean distance
    let dr = r1 - r2;
    let dg = g1 - g2;
    let db = b1 - b2;
    let da = a1 - a2;
    
    return sqrt(dr * dr + dg * dg + db * db + da * da);
}

@compute @workgroup_size(64)
fn calculateAssignments(@builtin(global_invocation_id) gid: vec3<u32>) {
    
    storageBarrier();

    let i = gid.x;

    // Out of bounds check
    if(i >= N) { return; }

    if(assignments[i] != N) {
        return; // Skip claimed pixels
    }

    // Find the output position with the minimum distance
    var minD: f32 = 3.40282e+38;
    var bestJ: u32 = 0u;
    var foundUnclaimed: bool = false;

    for (var j: u32 = 0u; j < N; j++) {

        if(atomicLoad(&usedJ[j]) != 0u) {
            continue; // Skip claimed pixels
        }

        let d = colorDistance(pixelsA[i], pixelsB[j]);
        if (d < minD) {
            minD = d;
            bestJ = j;
            foundUnclaimed = true;
        }
    }

    if(foundUnclaimed) {
        // Atomically check if we can claim this pixel position
        if(atomicCompareExchangeWeak(&usedJ[bestJ], 0u, 1u).exchanged) {
            assignments[i] = bestJ;
        }
    }
    
}
