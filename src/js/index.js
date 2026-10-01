const N_FRAMES = Math.max(1, CONFIG.ANIMATION_SECONDS * CONFIG.FPS)

const imageInputA = document.getElementById("imageInputA")
const imageInputB = document.getElementById("imageInputB")

const inputCanvasA = document.getElementById("canvasA")
const inputCanvasB = document.getElementById("canvasB")

const morphButton = document.getElementById("morph-button")
const gpuWarning = document.getElementById("gpu-warning")
const statusLine = document.getElementById("status-line")

let imageA = null
let imageB = null

let inputPixelsA = null
let inputPixelsB = null

// Shared processing dimensions (same for both images).
let W = 0
let H = 0

let GPU_AVAILABLE = false

let IS_ANIMATING = false
let IS_COMPUTING = false
// True once the morph has been computed and the animation has finished.
// Drives the Reset -> Play button cycle.
let HAS_MORPHED = false
// Incremented every time the viewport size changes. In-flight async work
// (morph computation, progress callbacks) captures the generation at start
// and discards its result if the generation has changed, preventing stale
// writes to the status, button, or animation state.
let SIZE_GENERATION = 0

// Display size (CSS px) of every canvas. The processing buffer (W x H) is
// derived from this so putImageData blits 1:1 with no zoom/crop. The size is
// measured from the live canvas element so the canvases fill the viewport.
let CANVAS_DISPLAY_SIZE = 220

// Reusable animation buffers (allocated once per morph).
let animFrameBuffer = null
let animImageData = null
let animInputVector = null
let animWeightedX = null
let animWeightedY = null
let animMorphedImage = null
let animFrameCount = 0
let animOnEnd = null

function setStatus(text) {
    if (statusLine) statusLine.textContent = text
}

// Measure the current CSS display size of the input canvas so the processing
// buffer matches it 1:1. Falls back to the last known size if not laid out.
function measureCanvasDisplaySize() {
    const rect = inputCanvasA.getBoundingClientRect()
    const size = Math.round(Math.min(rect.width, rect.height))
    if (size > 0) CANVAS_DISPLAY_SIZE = size
    return CANVAS_DISPLAY_SIZE
}

async function computeAssignments() {
    const packedA = packUint8VecToUint32Vec(inputPixelsA)
    const packedB = packUint8VecToUint32Vec(inputPixelsB)

    if (GPU_AVAILABLE) {
        console.warn("Using GPU shaders for assignment calculations...")
        const result = await assignPixelPositionsGPU(packedA, packedB, W, H)
        if (result != null) return result
        console.warn("GPU assignment failed, falling back to CPU...")
    }

    console.warn("Using CPU (Web Worker) for assignment calculations...")
    return await assignPixelPositions(inputPixelsA, inputPixelsB, W, H, (p) => {
        setStatus(`Computing assignments (CPU): ${Math.round(p)}%`)
    })
}

async function OnMorphButtonClick(event) {
    event.preventDefault()
    event.stopPropagation()

    if (inputPixelsA == null || inputPixelsB == null) return
    if (IS_COMPUTING) return

    // Stop any running animation.
    IS_ANIMATING = false

    IS_COMPUTING = true
    morphButton.disabled = true
    setStatus("Computing assignments...")

    // Capture the generation so we can detect a viewport resize after the
    // await and discard the stale result.
    const gen = SIZE_GENERATION

    let assignments = null
    try {
        assignments = await computeAssignments()
    } catch (err) {
        console.error("Error calculating pixel assignments!", err)
        setStatus("Error computing assignments.")
    }

    IS_COMPUTING = false
    morphButton.disabled = false

    // If the viewport resized while we were computing, the assignments were
    // calculated for the old dimensions. Discard them and let the user
    // re-trigger the morph at the new size.
    if (gen !== SIZE_GENERATION) {
        console.warn("Viewport resized during morph; discarding stale assignments.")
        return
    }

    if (assignments == null) {
        console.error("Error calculating pixel assignments!")
        setStatus("Error computing assignments.")
        return
    }

    // Build the morphed image (image A's pixels rearranged into B's layout).
    const N = W * H
    const morphedImage = new Uint8ClampedArray(N * 4)
    morphedImage.width = W
    morphedImage.height = H

    for (let i = 0; i < N; i++) {
        let j = assignments[i]
        if (j < 0 || j >= N) {
            // Unclaimed (shouldn't happen); keep in place.
            j = i
        }
        morphedImage[j * 4] = inputPixelsA[i * 4]
        morphedImage[j * 4 + 1] = inputPixelsA[i * 4 + 1]
        morphedImage[j * 4 + 2] = inputPixelsA[i * 4 + 2]
        morphedImage[j * 4 + 3] = inputPixelsA[i * 4 + 3]
    }

    // Precompute animation data once.
    animInputVector = packUint8VecToUint32Vec(inputPixelsA)
    animWeightedX = new Float32Array(N)
    animWeightedY = new Float32Array(N)
    for (let i = 0; i < N; i++) {
        let j = assignments[i]
        if (j < 0 || j >= N) j = i
        animWeightedX[i] = ((j % W) - (i % W)) / N_FRAMES
        animWeightedY[i] = (((j / W) | 0) - ((i / W) | 0)) / N_FRAMES
    }
    animMorphedImage = morphedImage
    animFrameBuffer = new Uint8ClampedArray(N * 4)
    animImageData = new ImageData(animFrameBuffer, W, H)
    animFrameCount = 0

    // The input image (left canvas) is the animation surface. It currently
    // shows the original input; do NOT reveal the final result yet — that
    // only happens once the animation completes.
    HAS_MORPHED = false
    clearCanvas(inputCanvasA)
    drawImageData(inputCanvasA, inputPixelsA)

    setStatus("Ready. Press Play to animate.")
    morphButton.textContent = "Play"
}


function startAnimation() {
    if (animInputVector == null) return

    IS_ANIMATING = true
    animFrameCount = 0
    clearCanvas(inputCanvasA)
    requestAnimationFrame(animateFrame)
}

// The animation finished on its own (reached the last frame). Reveal the
// final morphed result on the input canvas and switch the button to Reset.
function finishAnimation() {
    IS_ANIMATING = false
    if (animMorphedImage != null) {
        clearCanvas(inputCanvasA)
        drawImageData(inputCanvasA, animMorphedImage)
    }
    HAS_MORPHED = true
    morphButton.textContent = "Reset"
    setStatus("Morph complete. Press Reset to restore the original image.")
}

// Reset: redisplay the original input image and switch the button to Play.
function resetAnimation() {
    IS_ANIMATING = false
    if (inputPixelsA != null) {
        clearCanvas(inputCanvasA)
        drawImageData(inputCanvasA, inputPixelsA)
    }
    morphButton.textContent = "Play"
    setStatus("Ready. Press Play to animate.")
}

function animateFrame() {
    if (IS_ANIMATING === false) return

    const N = W * H
    const frameBuffer = animFrameBuffer
    const inputVector = animInputVector
    const wx = animWeightedX
    const wy = animWeightedY
    const fc = animFrameCount

    // Clear to transparent so letterbox bars stay clean.
    frameBuffer.fill(0)

    for (let i = 0; i < N; i++) {
        const x0 = i % W
        const y0 = (i / W) | 0

        const x1 = Math.round(x0 + wx[i] * fc)
        const y1 = Math.round(y0 + wy[i] * fc)

        if (x1 < 0 || x1 >= W || y1 < 0 || y1 >= H) continue

        const j = (y1 * W + x1) * 4
        const color = inputVector[i] >>> 0
        frameBuffer[j] = (color >>> 24) & 0xFF     // R
        frameBuffer[j + 1] = (color >>> 16) & 0xFF // G
        frameBuffer[j + 2] = (color >>> 8) & 0xFF  // B
        frameBuffer[j + 3] = color & 0xFF          // A
    }

    // Blit the whole buffer in one call.
    inputCanvasA.getContext("2d").putImageData(animImageData, 0, 0)

    animFrameCount++
    if (animFrameCount > N_FRAMES) {
        finishAnimation()
        return
    }
    requestAnimationFrame(animateFrame)
}

function loadImage(path) {
    return new Promise(resolve => {
        const url = URL.createObjectURL(path)
        const img = new Image()
        img.onload = () => {
            URL.revokeObjectURL(url)
            resolve(img)
        }
        img.onerror = () => {
            URL.revokeObjectURL(url)
            resolve(null)
        }
        img.src = url
    })
}

// Draw an image into a W x H canvas using "contain" (letterbox), preserving
// the image's aspect ratio. Letterbox bars are transparent.
function drawImageContain(img, canvas) {
    const ctx = canvas.getContext("2d")
    const cw = canvas.width
    const ch = canvas.height
    ctx.clearRect(0, 0, cw, ch)

    const iw = img.naturalWidth || img.width
    const ih = img.naturalHeight || img.height
    if (iw === 0 || ih === 0) return

    const scale = Math.min(cw / iw, ch / ih)
    const dw = Math.round(iw * scale)
    const dh = Math.round(ih * scale)
    const dx = Math.round((cw - dw) / 2)
    const dy = Math.round((ch - dh) / 2)
    ctx.drawImage(img, dx, dy, dw, dh)
}

// Recompute the shared W x H from the current viewport + both images, and
// resize all three canvases. Called when both images are present and on resize.
function recomputeSize() {
    if (imageA == null || imageB == null) return

    // Measure the live canvas display size so the buffer matches it 1:1.
    measureCanvasDisplaySize()

    // Size the processing buffer to the canvas display size so the
    // buffer dimensions equal the canvas dimensions and putImageData blits
    // 1:1 (no zoom/crop).
    const size = computeTargetSize(
        CANVAS_DISPLAY_SIZE, CANVAS_DISPLAY_SIZE,
        imageA.naturalWidth || imageA.width,
        imageA.naturalHeight || imageA.height,
        imageB.naturalWidth || imageB.width,
        imageB.naturalHeight || imageB.height
    )
    W = size.W
    H = size.H

    // Resize both input canvases to the shared dimensions.
    for (const c of [inputCanvasA, inputCanvasB]) {
        c.width = W
        c.height = H
    }

    // Redraw both inputs (letterboxed) and refresh their pixel data.
    drawImageContain(imageA, inputCanvasA)
    drawImageContain(imageB, inputCanvasB)
    inputPixelsA = loadImagePixelData(inputCanvasA, 0, 0, W, H)
    inputPixelsB = loadImagePixelData(inputCanvasB, 0, 0, W, H)

    // Reset any previous morph state.
    IS_ANIMATING = false
    HAS_MORPHED = false
    animInputVector = null
    animMorphedImage = null

    // Bump the generation so any in-flight morph computation (which captured
    // the old generation) will discard its stale result.
    SIZE_GENERATION++

    morphButton.disabled = false
    morphButton.textContent = "Morph"
    setStatus(`Loaded. Processing size: ${W}×${H}. Press Morph.`)
}

function initPage() {
    imageInputA.addEventListener("change", async (event) => {
        const inp = event.target
        if (inp.files && inp.files[0]) {
            imageA = await loadImage(inp.files[0])
            if (imageA == null) {
                console.error("Error loading image", inp.files[0])
                return
            }
            // If both images are present, recompute size; otherwise just store.
            if (imageB != null) {
                recomputeSize()
            } else {
                // Show a letterboxed preview of A on its own canvas. Measure
                // the live display size so the internal buffer matches the
                // CSS size and the image fills the canvas exactly.
                measureCanvasDisplaySize()
                inputCanvasA.width = CANVAS_DISPLAY_SIZE
                inputCanvasA.height = CANVAS_DISPLAY_SIZE
                drawImageContain(imageA, inputCanvasA)
                inputPixelsA = loadImagePixelData(inputCanvasA, 0, 0, CANVAS_DISPLAY_SIZE, CANVAS_DISPLAY_SIZE)
                setStatus(`[v2] Image A loaded. Choose image B.`)
            }
        }
    })

    imageInputB.addEventListener("change", async (event) => {
        const inp = event.target
        if (inp.files && inp.files[0]) {
            imageB = await loadImage(inp.files[0])
            if (imageB == null) {
                console.error("Error loading image", inp.files[0])
                return
            }
            if (imageA != null) {
                recomputeSize()
            }
        }
    })

    // Morph / Play / Stop button.
    morphButton.addEventListener("click", (event) => {
        event.preventDefault()
        event.stopPropagation()

        if (IS_COMPUTING) return

        if (morphButton.textContent === "Play") {
            // Start the animation.
            startAnimation()
            morphButton.textContent = "Stop"
        } else if (morphButton.textContent === "Stop") {
            // Stop the animation mid-flight: reveal the final result and
            // switch to Reset.
            finishAnimation()
        } else if (morphButton.textContent === "Reset") {
            // Redisplay the original input image and switch to Play.
            resetAnimation()
        } else {
            // "Morph" — compute assignments.
            OnMorphButtonClick(event)
        }
    })

    // Recompute size on window resize (debounced).
    let resizeTimer = null
    window.addEventListener("resize", () => {
        if (imageA == null || imageB == null) return
        clearTimeout(resizeTimer)
        resizeTimer = setTimeout(recomputeSize, 200)
    })
}

async function main() {
    // Initialize GPU Shaders (auto-detect).
    GPU_AVAILABLE = await initShaders()
    if (GPU_AVAILABLE == false) {
        if (gpuWarning) gpuWarning.style.display = "block"
    } else {
        if (gpuWarning) gpuWarning.style.display = "none"
    }

    // Main entry point
    initPage()
}

main()
