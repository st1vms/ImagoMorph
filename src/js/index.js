const CANVAS_WIDTH = 100
const CANVAS_HEIGHT = 100

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

const morphButton = document.getElementById("morph-button")

let imageA = null
let imageB = null

let inputPixelsA = null
let inputPixelsB = null

let GPU_AVAILABLE = false

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
    if (GPU_AVAILABLE) {
        assignments = await assignPixelPositionsGPU(
            packRGBAtoUint32(inputPixelsA), 
            packRGBAtoUint32(inputPixelsB)
        )
    } else {
        console.warn("GPU not available, falling back to CPU for assignment calculations, this may take a very long time...")
        assignments = assignPixelPositions(inputPixelsA, inputPixelsB)
    }

    if (assignments == null) {
        console.error("Error calculating pixel assignments!")
        return
    }

    // Perform assignment
    let newOutput = new Uint8ClampedArray(inputPixelsA.length)
    newOutput.width = inputPixelsA.width
    newOutput.height = inputPixelsA.height

    const N = inputPixelsA.length / 4
    for (let i = 0; i < N; i++) {
        const j = assignments[i]

        newOutput[j * 4] = inputPixelsA[i * 4]
        newOutput[j * 4 + 1] = inputPixelsA[i * 4 + 1]
        newOutput[j * 4 + 2] = inputPixelsA[i * 4 + 2]
        newOutput[j * 4 + 3] = inputPixelsA[i * 4 + 3]
    }

    morphButton.textContent = morphBtnText

    // Draw morphed picture
    clearCanvas(outputCanvas)
    drawImagePixelData(newOutput, outputCanvas)
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

    // Main entry point
    initPage()
}

main()
