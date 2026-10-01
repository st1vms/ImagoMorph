// CPU pixel-assignment worker.
// Fast greedy matching using a spatial grid + expanding local search window.
// Runs off the main thread so the UI never freezes.
//
// Message in:  { pixelsA: Uint8ClampedArray, pixelsB: Uint8ClampedArray,
//                W, H, CELL_SIZE, SEARCH_RADIUS, SPATIAL_WEIGHT }
// Message out: { type: "progress", value: number }
//              { type: "done", assignments: Uint32Array }

self.onmessage = function (e) {
    const pixelsA = e.data.pixelsA;
    const pixelsB = e.data.pixelsB;
    const W = e.data.W;
    const H = e.data.H;
    const CELL = e.data.CELL_SIZE;
    const RADIUS = e.data.SEARCH_RADIUS;
    const SPATIAL_WEIGHT = e.data.SPATIAL_WEIGHT;

    const N = W * H;
    const assignments = new Uint32Array(N);

    // ---- Build a spatial grid of destination pixels ----
    const cellsX = Math.max(1, Math.ceil(W / CELL));
    const cellsY = Math.max(1, Math.ceil(H / CELL));
    const numCells = cellsX * cellsY;

    // Head of each cell's linked list + next pointer per pixel.
    const head = new Int32Array(numCells).fill(-1);
    const next = new Int32Array(N);

    for (let j = 0; j < N; j++) {
        const cx = Math.min(cellsX - 1, (j % W) / CELL | 0);
        const cy = Math.min(cellsY - 1, ((j / W) | 0) / CELL | 0);
        const c = cy * cellsX + cx;
        next[j] = head[c];
        head[c] = j;
    }

    const used = new Uint8Array(N);

    // Precompute source pixel coordinates and colors.
    const srcX = new Int32Array(N);
    const srcY = new Int32Array(N);
    const srcR = new Int32Array(N);
    const srcG = new Int32Array(N);
    const srcB = new Int32Array(N);
    const srcA = new Int32Array(N);
    for (let i = 0; i < N; i++) {
        srcX[i] = i % W;
        srcY[i] = (i / W) | 0;
        srcR[i] = pixelsA[i * 4];
        srcG[i] = pixelsA[i * 4 + 1];
        srcB[i] = pixelsA[i * 4 + 2];
        srcA[i] = pixelsA[i * 4 + 3];
    }

    // Destination colors (kept in pixelsB, indexed directly).

    let progressStep = Math.max(1, N / 20);
    let lastProgress = 0;

    for (let i = 0; i < N; i++) {
        const x0 = srcX[i];
        const y0 = srcY[i];
        const r0 = srcR[i];
        const g0 = srcG[i];
        const b0 = srcB[i];
        const a0 = srcA[i];

        const cx0 = Math.min(cellsX - 1, x0 / CELL | 0);
        const cy0 = Math.min(cellsY - 1, y0 / CELL | 0);

        let bestJ = -1;
        let bestCost = Infinity;

        // Expanding window search over neighboring cells.
        let found = false;
        for (let radius = RADIUS; radius <= cellsX + cellsY && !found; radius++) {
            const minX = Math.max(0, cx0 - radius);
            const maxX = Math.min(cellsX - 1, cx0 + radius);
            const minY = Math.max(0, cy0 - radius);
            const maxY = Math.min(cellsY - 1, cy0 + radius);

            for (let cy = minY; cy <= maxY; cy++) {
                for (let cx = minX; cx <= maxX; cx++) {
                    let j = head[cy * cellsX + cx];
                    while (j !== -1) {
                        if (used[j] === 0) {
                            const x1 = j % W;
                            const y1 = (j / W) | 0;
                            const dr = r0 - pixelsB[j * 4];
                            const dg = g0 - pixelsB[j * 4 + 1];
                            const db = b0 - pixelsB[j * 4 + 2];
                            const da = a0 - pixelsB[j * 4 + 3];
                            const colorDist = Math.sqrt(dr * dr + dg * dg + db * db + da * da);
                            const dx = x1 - x0;
                            const dy = y1 - y0;
                            const spatialDist = Math.sqrt(dx * dx + dy * dy);
                            const cost = colorDist + SPATIAL_WEIGHT * spatialDist;
                            if (cost < bestCost) {
                                bestCost = cost;
                                bestJ = j;
                            }
                        }
                        j = next[j];
                    }
                }
            }
            found = bestJ !== -1;
        }

        // Fallback: if the window found nothing (all claimed), scan globally.
        if (bestJ === -1) {
            for (let j = 0; j < N; j++) {
                if (used[j] === 0) {
                    const x1 = j % W;
                    const y1 = (j / W) | 0;
                    const dr = r0 - pixelsB[j * 4];
                    const dg = g0 - pixelsB[j * 4 + 1];
                    const db = b0 - pixelsB[j * 4 + 2];
                    const da = a0 - pixelsB[j * 4 + 3];
                    const colorDist = Math.sqrt(dr * dr + dg * dg + db * db + da * da);
                    const dx = x1 - x0;
                    const dy = y1 - y0;
                    const spatialDist = Math.sqrt(dx * dx + dy * dy);
                    const cost = colorDist + SPATIAL_WEIGHT * spatialDist;
                    if (cost < bestCost) {
                        bestCost = cost;
                        bestJ = j;
                    }
                }
            }
        }

        // Safety: if still nothing (shouldn't happen), take any unclaimed.
        if (bestJ === -1) {
            for (let j = 0; j < N; j++) {
                if (used[j] === 0) { bestJ = j; break; }
            }
        }

        used[bestJ] = 1;
        assignments[i] = bestJ;

        if (i - lastProgress >= progressStep) {
            lastProgress = i;
            self.postMessage({ type: "progress", value: (i / N) * 100 });
        }
    }

    self.postMessage({ type: "done", assignments }, [assignments.buffer]);
};
