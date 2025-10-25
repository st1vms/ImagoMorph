function pixelColorDistance(pixelsA, pixelsB, i, j) {
    let sum = 0
    for (let s = 0; s < 4; s++) {
        sum += (pixelsA[i * 4 + s] - pixelsB[j * 4 + s]) ** 2
    }
    return Math.sqrt(sum)
}

function assignPixelPositions(pixelsA, pixelsB) {

    let assignments = {}

    const N = pixelsA.length / 4

    let used = new Array(N).fill(false)

    for (let i = 0; i < N; i++) {

        let min = Infinity
        let bestJ = 0
        for (let j = 0; j < N; j++) {

            if (used[j] === true) {
                continue
            }

            const cost = pixelColorDistance(pixelsA, pixelsB, i, j)
            if (cost < min) {
                min = cost
                bestJ = j
            }
        }
        assignments[i] = bestJ
        used[bestJ] = true
    }

    return assignments
}