function pointToIndex(x, y, width, cell_size = 1) {
    return (y * width + x) * cell_size
}

function clearCanvas(canvas) {
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height)
}

function drawCanvasImageAutoScaled(img, canvas) {
    canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height)
}

function drawCanvasPixel(pixel_rgba, canvas_ctx, w, x, y) {
    const i = pointToIndex(x, y, w, 4)
    canvas_ctx.fillStyle = `rgba(${pixel_rgba[i]},${pixel_rgba[i + 1]},${pixel_rgba[i + 2]},${pixel_rgba[i + 3] / 255})`;
    canvas_ctx.fillRect(x, y, 1, 1);
}

function loadImagePixelData(canvas, x, y, img_w, img_h) {
    let inputPixels = canvas.getContext("2d").getImageData(x, y, img_w, img_h).data;
    inputPixels.width = img_w
    inputPixels.height = img_h
    return inputPixels
}

function drawImagePixelData(img_pixel_data, canvas) {
    const ctx = canvas.getContext("2d");
    for (let y = 0; y < img_pixel_data.height; y++) {
        for (let x = 0; x < img_pixel_data.width; x++) {
            drawCanvasPixel(img_pixel_data, ctx, img_pixel_data.width, x, y)
        }
    }
}