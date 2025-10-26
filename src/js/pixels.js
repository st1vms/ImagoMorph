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

function packRGBAtoUint32(bytes) {
    if (!(bytes instanceof Uint8ClampedArray)) {
        throw new TypeError('bytes must be a Uint8ClampedArray');
    }
    const len = Math.floor(bytes.length / 4);
    const out = new Uint32Array(len);
    for (let i = 0, j = 0; i < len; i++, j += 4) {
        const r = bytes[j];
        const g = bytes[j + 1];
        const b = bytes[j + 2];
        const a = bytes[j + 3];
        out[i] = (((r << 24) >>> 0) | (g << 16) | (b << 8) | a) >>> 0;
    }
    return out;
}

function unpackUint32toRGBA(pixels32) {
    if (!(pixels32 instanceof Uint32Array)) {
        throw new TypeError('pixels32 must be a Uint32Array');
    }
    const out = new Uint8ClampedArray(pixels32.length * 4);
    for (let i = 0, j = 0; i < pixels32.length; i++, j += 4) {
        const v = pixels32[i] >>> 0;
        out[j] = (v >>> 24) & 0xFF; // R
        out[j + 1] = (v >>> 16) & 0xFF; // G
        out[j + 2] = (v >>> 8) & 0xFF; // B
        out[j + 3] = v & 0xFF;          // A
    }
    return out;
}
