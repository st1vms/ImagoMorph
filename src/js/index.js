const CANVAS_WIDTH = 100
const CANVAS_HEIGHT = 100
const ANIMATION_PLAY_TIME_SECONDS = 10
const FPS = 60
const N_FRAMES = Math.max(1, ANIMATION_PLAY_TIME_SECONDS * FPS)

const imageInputA = document.getElementById("imageInputA")
const imageInputB = document.getElementById("imageInputB")

const inputCanvasA = document.getElementById("canvasA")
inputCanvasA.width = CANVAS_WIDTH
inputCanvasA.height = CANVAS_HEIGHT

const inputCanvasB = document.getElementById("canvasB")
inputCanvasB.width = CANVAS_WIDTH
inputCanvasB.height = CANVAS_HEIGHT

const outputCanvas = document.getElementById("canvas-output")
outputCanvas.width = CANVAS_WIDTH
outputCanvas.height = CANVAS_HEIGHT

const useGPUCheckbox = document.getElementById("useGPUCheckbox")
const morphButton = document.getElementById("morph-button")

let imageA = null
let imageB = null

let inputPixelsA = null
let inputPixelsB = null

let GPU_AVAILABLE = false

let IS_ANIMATING = false

async function OnMorphButtonClick(event) {

    event.preventDefault()
    event.stopPropagation()

    if (inputPixelsA == null || inputPixelsB == null) {
        return
    }

    let morphBtnText = morphButton.textContent
    morphButton.textContent = "Wait..."

    let assignments = null

    // Calculate assignments
    if (useGPUCheckbox.checked) {
        console.warn("Using GPU shaders for assignment calculations...")
        assignments = await assignPixelPositionsGPU(
            packUint8VecToUint32Vec(inputPixelsA),
            packUint8VecToUint32Vec(inputPixelsB)
        )
    } else {
        console.warn("GPU not available, falling back to CPU for assignment calculations...")
        assignments = assignPixelPositions(inputPixelsA, inputPixelsB)
    }

    if (assignments == null) {
        console.error("Error calculating pixel assignments!")
        return
    }

    // Perform assignment
    let morphedImage = new Uint8ClampedArray(inputPixelsA.length)
    morphedImage.width = inputPixelsA.width
    morphedImage.height = inputPixelsA.height

    const N = inputPixelsA.length / 4
    for (let i = 0; i < N; i++) {
        if (i < 0 || i >= assignments.length) {
            console.warn("Invalid assignment: " + i)
            continue
        }
        const j = assignments[i]

        morphedImage[j * 4] = inputPixelsA[i * 4]
        morphedImage[j * 4 + 1] = inputPixelsA[i * 4 + 1]
        morphedImage[j * 4 + 2] = inputPixelsA[i * 4 + 2]
        morphedImage[j * 4 + 3] = inputPixelsA[i * 4 + 3]
    }

    morphButton.textContent = morphBtnText

    // Draw morphed picture
    clearCanvas(outputCanvas)
    drawImagePixelData(morphedImage, outputCanvas)


    morphButton.textContent = "Play!"
    morphButton.removeEventListener("click", OnMorphButtonClick)
    morphButton.addEventListener("click", function (event) {
        event.preventDefault()
        event.stopPropagation()

        if (IS_ANIMATING === true) {
            // Stop button behavior
            IS_ANIMATING = false
            return
        }

        morphButton.textContent = "Stop"

        MorphAnimation(morphedImage, assignments, () => {
            // On animation end callback
            morphButton.textContent = "Play!"
        })
    })
}


function MorphAnimation(morphedImage, assignments, OnAnimationEnd) {
    const inputVector = packUint8VecToUint32Vec(inputPixelsA)

    // Input dimensions
    const N = inputVector.length
    const W = Math.sqrt(N)

    // Calculate weighted distances based off animation duration
    let weightedDistanceX = new Float32Array(N)
    let weightedDistanceY = new Float32Array(N)

    for (let i = 0; i < N; i++) {

        const j = assignments[i]

        // Use index to 2D point formula to calculate distance
        weightedDistanceX[i] = (j % W) - (i % W)
        weightedDistanceY[i] = Math.floor(j / W) - Math.floor(i / W)
    }

    for (let i = 0; i < weightedDistanceX.length; i++) {
        weightedDistanceX[i] = weightedDistanceX[i] / N_FRAMES
        weightedDistanceY[i] = weightedDistanceY[i] / N_FRAMES
    }

    let frame_count = 0

    function animateFrame() {

        if (IS_ANIMATING === false || frame_count > N_FRAMES) {
            // User stopped the animation
            IS_ANIMATING = false
            clearCanvas(outputCanvas)
            drawImagePixelData(morphedImage, outputCanvas)

            if (typeof OnAnimationEnd === "function") {
                OnAnimationEnd() // Run callback
            }
            return
        }

        // Create the frame buffer that will be printed in the canvas
        const frameBuffer = new Uint8ClampedArray(inputPixelsB.length)
        frameBuffer.width = W
        frameBuffer.height = W

        // Generate frame
        for (let i = 0; i < inputVector.length; i++) {

            const x0 = i % W
            const y0 = Math.floor(i / W)

            const x1 = Math.round(x0 + weightedDistanceX[i] * frame_count)
            const y1 = Math.round(y0 + weightedDistanceY[i] * frame_count)

            const j = (y1 * W + x1) * 4

            if (x1 < 0 || x1 >= W || y1 < 0 || y1 >= W) continue

            const color = inputVector[i] >>> 0;
            frameBuffer[j] = (color >>> 24) & 0xFF; // R
            frameBuffer[j + 1] = (color >>> 16) & 0xFF; // G
            frameBuffer[j + 2] = (color >>> 8) & 0xFF; // B
            frameBuffer[j + 3] = color & 0xFF;          // A
        }

        // Draw the frame
        drawImagePixelData(frameBuffer, outputCanvas)
        frame_count++
        requestAnimationFrame(animateFrame)
    }

    // Signal that the animation has started
    IS_ANIMATING = true

    // Start the animation with an empty canvas
    clearCanvas(outputCanvas)

    // Animation loop
    requestAnimationFrame(animateFrame)
}

function loadImage(path) {
    return new Promise(resolve => {
        const img = new Image()
        img.onload = () => { resolve(img) }
        img.onerror = () => { resolve(null) }
        img.src = URL.createObjectURL(path)
    })
}


function initPage() {
    imageInputA.addEventListener("change", async (event) => {
        const inp = event.target
        if (inp.files) {
            imageA = await loadImage(inp.files[0])
            if (imageA == null) {
                console.error("Error loading image", inp.files[0])
                return
            }

            drawCanvasImageAutoScaled(imageA, inputCanvasA)

            // Get input A pixels
            inputPixelsA = loadImagePixelData(inputCanvasA, 0, 0, inputCanvasA.width, inputCanvasA.height)

            // Initialize output canvas
            drawImagePixelData(inputPixelsA, outputCanvas)

            // Enable secondary input and morph button
            imageInputB.disabled = false
            morphButton.disabled = false
        }
    })

    imageInputB.addEventListener("change", async (event) => {
        const inp = event.target
        if (inp.files) {
            imageB = await loadImage(inp.files[0])
            if (imageB == null) {
                console.error("Error loading image", inp.files[0])
                return
            }

            drawCanvasImageAutoScaled(imageB, inputCanvasB)

            // Get input B pixels
            inputPixelsB = loadImagePixelData(inputCanvasB, 0, 0, inputCanvasB.width, inputCanvasB.height)
        }
    })

    morphButton.addEventListener("click", OnMorphButtonClick)
}

async function main() {
    // Initialize GPU Shaders
    GPU_AVAILABLE = await initShaders()
    if (GPU_AVAILABLE == false) {
        useGPUCheckbox.disabled = true
    } else {
        useGPUCheckbox.disabled = false
    }

    // Main entry point
    initPage()
}

main()
