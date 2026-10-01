// CPU pixel-assignment: thin async wrapper around the Web Worker.
// The heavy lifting (spatial-grid greedy matching) lives in match_worker.js
// so the main thread / UI never blocks.

function assignPixelPositions(pixelsA, pixelsB, W, H, onProgress) {
    return new Promise((resolve, reject) => {
        let worker;
        try {
            worker = new Worker("src/js/match_worker.js");
        } catch (err) {
            reject(new Error("Failed to create matching worker: " + err.message));
            return;
        }

        worker.onmessage = function (e) {
            const msg = e.data;
            if (msg.type === "progress") {
                if (typeof onProgress === "function") onProgress(msg.value);
            } else if (msg.type === "done") {
                worker.terminate();
                resolve(msg.assignments);
            }
        };

        worker.onerror = function (err) {
            worker.terminate();
            reject(new Error("Matching worker error: " + (err.message || err)));
        };

        worker.postMessage({
            pixelsA: pixelsA,
            pixelsB: pixelsB,
            W: W,
            H: H,
            CELL_SIZE: CONFIG.CELL_SIZE,
            SEARCH_RADIUS: CONFIG.SEARCH_RADIUS,
            SPATIAL_WEIGHT: CONFIG.SPATIAL_WEIGHT,
        });
    });
}
