// Input pixels
@group(0) @binding(0) var<storage, read> pixelsA: array<u32>;
@group(0) @binding(1) var<storage, read> pixelsB: array<u32>;

// Output assigment vector
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;

// Synchronization vector for flagging occupied columns (Initialized to 0u)
@group(0) @binding(3) var<storage, read_write> usedJ: array<atomic<u32>>;

fn colorDistance(a: u32, b: u32) -> u32 {
    return u32(abs(i32(a) - i32(b)));
}

// Nearest Neighbour  assignment
@compute @workgroup_size(64)
fn calculateNearestColorAssignments(@builtin(global_invocation_id) gid: vec3<u32>) {
    
    let i = gid.x;
    let N = arrayLength(&pixelsA);

    // Out of bounds check
    if (i >= N) { return; }

    loop {

        // Initialize minimum distance and best output position
        var minD: u32 = 0xffffffffu;
        var bestJ: u32 = 0u;
        
        // Find the output position with the minimum distance
        for (var j: u32 = 0u; j < N; j++) {

            // Skip occupied positions
            if (atomicLoad(&usedJ[j]) == 1u) { 
                continue;
            }

            let d = colorDistance(pixelsA[i], pixelsB[j]);
            if (d < minD) {
                minD = d;
                bestJ = j;
            }
        }

        // Check if this position was already occupied, if so find another
        if (atomicLoad(&usedJ[bestJ]) == 0u) { 

            // Signal the position as occupied
            atomicStore(&usedJ[bestJ], 1u);

            // Assign the position and return
            assignments[i] = bestJ;
            return;
        }
    }
}
