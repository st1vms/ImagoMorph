// Single-pass pixel assignment shader.
// Each thread handles one source pixel i. It searches an expanding local
// window (spatial grid) of destination pixels for the best unclaimed match,
// using the SAME cost function as the CPU worker:
//   cost = colorDistance(RGBA) + SPATIAL_WEIGHT * euclideanDistance
// A global fallback scan guarantees every i gets a claim in this single pass.

// Input size
@group(0) @binding(0) var<storage, read> N: u32;

// Image dimensions (W, H)
@group(0) @binding(1) var<storage, read> dims: vec2<u32>;

// Spatial weight (as f32 bits packed in a u32)
@group(0) @binding(2) var<storage, read> spatialWeightBits: u32;

// Input pixels
@group(0) @binding(3) var<storage, read> pixelsA: array<u32>;
@group(0) @binding(4) var<storage, read> pixelsB: array<u32>;

// Used positions arrays
@group(0) @binding(5) var<storage, read_write> usedJ: array<atomic<u32>>;

// Output assignments
@group(0) @binding(6) var<storage, read_write> assignments: array<u32>;

// Grid parameters
let CELL: u32 = 12u;
let RADIUS: u32 = 3u;

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

    let dr = r1 - r2;
    let dg = g1 - g2;
    let db = b1 - b2;
    let da = a1 - a2;

    return sqrt(dr * dr + dg * dg + db * db + da * da);
}

fn spatialDistance(i: u32, j: u32) -> f32 {
    let W = dims.x;
    let x0 = f32(i % W);
    let y0 = f32(i / W);
    let x1 = f32(j % W);
    let y1 = f32(j / W);
    let dx = x1 - x0;
    let dy = y1 - y0;
    return sqrt(dx * dx + dy * dy);
}

fn totalCost(i: u32, j: u32) -> f32 {
    let sw = bitcast<f32>(spatialWeightBits);
    return colorDistance(pixelsA[i], pixelsB[j]) + sw * spatialDistance(i, j);
}

// Try to claim destination j for source i. Returns true if claimed.
fn tryClaim(i: u32, j: u32) -> bool {
    let result = atomicCompareExchangeWeak(&usedJ[j], 0u, 1u);
    if (result.exchanged) {
        assignments[i] = j;
        return true;
    }
    return false;
}

@compute @workgroup_size(64)
fn calculateAssignments(@builtin(global_invocation_id) gid: vec3<u32>) {

    let i = gid.x;

    // Out of bounds check
    if (i >= N) { return; }

    let W = dims.x;
    let H = dims.y;
    let cellsX = (W + CELL - 1u) / CELL;
    let cellsY = (H + CELL - 1u) / CELL;

    let x0 = i % W;
    let y0 = i / W;
    let cx0 = x0 / CELL;
    let cy0 = y0 / CELL;

    // Expanding window search over neighboring cells.
    var bestJ: u32 = 0u;
    var bestCost: f32 = 3.40282e+38;
    var found: bool = false;

    for (var radius: u32 = RADIUS; radius <= cellsX + cellsY && !found; radius = radius + 1u) {
        let minX = select(0u, cx0 - radius, cx0 >= radius);
        let maxX = select(cellsX - 1u, cx0 + radius, cx0 + radius < cellsX);
        let minY = select(0u, cy0 - radius, cy0 >= radius);
        let maxY = select(cellsY - 1u, cy0 + radius, cy0 + radius < cellsY);

        for (var cy: u32 = minY; cy <= maxY; cy = cy + 1u) {
            for (var cx: u32 = minX; cx <= maxX; cx = cx + 1u) {
                // Scan all pixels in this cell's region.
                let xStart = cx * CELL;
                let xEnd = min(W, xStart + CELL);
                let yStart = cy * CELL;
                let yEnd = min(H, yStart + CELL);
                for (var yy: u32 = yStart; yy < yEnd; yy = yy + 1u) {
                    for (var xx: u32 = xStart; xx < xEnd; xx = xx + 1u) {
                        let j = yy * W + xx;
                        if (atomicLoad(&usedJ[j]) != 0u) { continue; }
                        let c = totalCost(i, j);
                        if (c < bestCost) {
                            bestCost = c;
                            bestJ = j;
                        }
                    }
                }
            }
        }
        found = bestCost < 3.40282e+38;
    }

    // Global fallback scan if the window found nothing.
    if (!found) {
        for (var j: u32 = 0u; j < N; j = j + 1u) {
            if (atomicLoad(&usedJ[j]) != 0u) { continue; }
            let c = totalCost(i, j);
            if (c < bestCost) {
                bestCost = c;
                bestJ = j;
            }
        }
    }

    // Claim the best destination.
    if (bestCost < 3.40282e+38) {
        tryClaim(i, bestJ);
    }

    // If the claim failed (contention), do a final global scan for any
    // unclaimed destination and claim it, guaranteeing a complete assignment.
    if (assignments[i] == N) {
        for (var j: u32 = 0u; j < N; j = j + 1u) {
            if (atomicLoad(&usedJ[j]) != 0u) { continue; }
            if (tryClaim(i, j)) { break; }
        }
    }
}
