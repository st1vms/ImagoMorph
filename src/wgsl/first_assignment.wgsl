// Input size
@group(0) @binding(0) var<storage, read> N: u32;

// Input pixels
@group(0) @binding(1) var<storage, read> pixelsA: array<u32>;
@group(0) @binding(2) var<storage, read> pixelsB: array<u32>;

// Output assignments
@group(0) @binding(3) var<storage, read_write> assignments: array<u32>;

// Shared memory for tile-based processing
const TILE_SIZE: u32 = 256u;
var<workgroup> sharedPixelsB: array<u32, TILE_SIZE>;
var<workgroup> sharedDistances: array<f32, 64>;
var<workgroup> sharedIndices: array<u32, 64>;

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
    
    // Calculate Euclidean distance squared (avoid sqrt for comparison)
    let dr = r1 - r2;
    let dg = g1 - g2;
    let db = b1 - b2;
    let da = a1 - a2;
    
    return dr * dr + dg * dg + db * db + da * da;
}

@compute @workgroup_size(64)
fn calculateFirstAssignments(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let i = gid.x;
    let local_id = lid.x;

    // Out of bounds check
    if(i >= N) { return; }

    let pixelA = pixelsA[i];
    var minD: f32 = 3.402823466e+38; // FLT_MAX
    var bestJ: u32 = 0u;
    
    // Process pixelsB in tiles using shared memory
    let numTiles = (N + TILE_SIZE - 1u) / TILE_SIZE;
    
    for (var tile: u32 = 0u; tile < numTiles; tile++) {
        let tileStart = tile * TILE_SIZE;
        let tileEnd = min(tileStart + TILE_SIZE, N);
        let tileCount = tileEnd - tileStart;
        
        // Load tile into shared memory (cooperative loading)
        for (var offset = local_id; offset < tileCount; offset += 64u) {
            let globalIdx = tileStart + offset;
            if (globalIdx < N) {
                sharedPixelsB[offset] = pixelsB[globalIdx];
            }
        }
        
        // Synchronize to ensure all data is loaded
        workgroupBarrier();
        
        // Find minimum distance within this tile
        for (var j: u32 = 0u; j < tileCount; j++) {
            let d = colorDistance(pixelA, sharedPixelsB[j]);
            if (d < minD) {
                minD = d;
                bestJ = tileStart + j;
            }
        }
        
        // Synchronize before loading next tile
        workgroupBarrier();
    }

    assignments[i] = bestJ;
}
