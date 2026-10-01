// Shared configuration for ImagoMorph.
// Loaded first (before match_shader.js / match.js / pixels.js / index.js).

const CONFIG = {
    // Spatial grid used by both the CPU worker and the GPU shader.
    CELL_SIZE: 12,            // pixels per grid cell
    SEARCH_RADIUS: 3,         // initial search window in cells (grows if needed)
    SPATIAL_WEIGHT: 0.5,      // weight of the spatial term in the cost function

    // Animation
    ANIMATION_SECONDS: 10,
    FPS: 60,

    // Performance budget: cap on total pixels (W*H) processed at once.
    MAX_PIXELS: 160000,
    MIN_PIXELS: 4000,
};

// Compute a single target {W, H} that BOTH images will be drawn into.
// Both images are drawn into the same W x H using "contain" (letterbox),
// which guarantees equal pixel counts (valid bijection) while preserving
// each image's aspect ratio.
//
// @param {number} boxW  available width of the output area (CSS px)
// @param {number} boxH  available height of the output area (CSS px)
// @param {number} imgAW width of image A (natural px)
// @param {number} imgAH height of image A
// @param {number} imgBW width of image B
// @param {number} imgBH height of image B
// @returns {{W:number, H:number}}
function computeTargetSize(boxW, boxH, imgAW, imgAH, imgBW, imgBH) {
    // The shared canvas must be able to contain BOTH images (letterboxed).
    // The limiting factor is the image with the larger aspect ratio.
    const aspectA = imgAW / imgAH;
    const aspectB = imgBW / imgBH;
    const maxAspect = Math.max(aspectA, aspectB);
    const minAspect = Math.min(aspectA, aspectB);

    // Fit a W x H box (aspect = maxAspect) into the available area.
    let W, H;
    if (boxW / boxH >= maxAspect) {
        // Height limited
        H = boxH;
        W = boxH * maxAspect;
    } else {
        // Width limited
        W = boxW;
        H = boxW / maxAspect;
    }

    // Round to integers.
    W = Math.max(1, Math.round(W));
    H = Math.max(1, Math.round(H));

    // Enforce the performance budget by scaling down (keep aspect).
    const N = W * H;
    if (N > CONFIG.MAX_PIXELS) {
        const scale = Math.sqrt(CONFIG.MAX_PIXELS / N);
        W = Math.max(1, Math.round(W * scale));
        H = Math.max(1, Math.round(H * scale));
    } else if (N < CONFIG.MIN_PIXELS) {
        const scale = Math.sqrt(CONFIG.MIN_PIXELS / N);
        W = Math.min(4096, Math.round(W * scale));
        H = Math.min(4096, Math.round(H * scale));
    }

    return { W, H };
}
