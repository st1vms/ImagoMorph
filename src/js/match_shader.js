let GPU_ADAPTER = undefined
let GPU_DEVICE = undefined

const DEFAULT_WORKGROUP_SIZE = 64

let CALC_ASSIGNMENTS_PIPELINE = undefined

async function loadShaderModule(device, path) {
    const response = await fetch(path);
    const shaderCode = await response.text();
    return device.createShaderModule({ code: shaderCode });
}

function createBindGroup(device, pipeline, binding_buffers) {

    let entries = []
    for (let i = 0; i < binding_buffers.length; i++) {
        entries.push(
            { binding: i, resource: { buffer: binding_buffers[i] } }
        )
    }

    return device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: entries
    });
}

function createShaderPipeline(device, binding_buffer_types, shaderModule, entryPoint) {

    let entries = []
    for (let i = 0; i < binding_buffer_types.length; i++) {
        entries.push(
            { binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding_buffer_types[i] } },
        )
    }

    return device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [
                device.createBindGroupLayout({
                    entries: entries
                })
            ]
        }),
        compute: { module: shaderModule, entryPoint: entryPoint },
    });
}

function runComputePass(device, pipeline, bindGroup, inputLength, workgroup_size = 64) {
    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
        Math.min(
            Math.ceil(inputLength / workgroup_size),
            GPU_ADAPTER.limits.maxComputeWorkgroupsPerDimension
        )
    );
    pass.end();

    // Perform assignment computations
    device.queue.submit([commandEncoder.finish()]);

    return device.queue.onSubmittedWorkDone();
}

function createBufferU32(device, byteSize, usage, copySrcBuffer = null) {

    const buffer = device.createBuffer({
        size: byteSize,
        usage: usage
    });

    if (copySrcBuffer != null) {
        device.queue.writeBuffer(buffer, 0, copySrcBuffer);
    } else {
        device.queue.writeBuffer(buffer, 0, new Uint32Array(byteSize / Uint32Array.BYTES_PER_ELEMENT).fill(0));
    }
    return buffer
}

async function computeBufferToCPUBuffer(device, buffer, bufferByteSize) {
    const readback = device.createBuffer({
        size: bufferByteSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(buffer, 0, readback, 0, bufferByteSize);
    device.queue.submit([copyEncoder.finish()]);

    await readback.mapAsync(GPUMapMode.READ);
    const result = new Uint32Array(readback.getMappedRange()).slice();
    readback.unmap();
    return result
}

async function initShaders() {
    // Guard: WebGPU may not be available in this browser.
    if (typeof navigator === "undefined" || navigator.gpu == null) {
        console.warn("WebGPU is not available in this browser.");
        return false
    }

    let adapter
    try {
        adapter = await navigator.gpu.requestAdapter();
    } catch (err) {
        console.warn("Failed to request WebGPU adapter:", err);
        return false
    }

    if (adapter == null) {
        console.warn("Cannot retrieve GPU adapter...")
        return false
    }
    GPU_ADAPTER = adapter;

    // Request available device limits
    GPU_DEVICE = await GPU_ADAPTER.requestDevice({
        requiredLimits: {
            maxBufferSize: GPU_ADAPTER.limits.maxBufferSize,
            maxStorageBufferBindingSize: GPU_ADAPTER.limits.maxStorageBufferBindingSize
        }
    });

    if (GPU_DEVICE == null) {
        console.error("Cannot retrieve GPU device from adapter...")
        return false
    }

    // Load shader module
    CALC_ASSIGNMENTS_PIPELINE = createShaderPipeline(
        GPU_DEVICE, [
        "read-only-storage", // N
        "read-only-storage", // dims (vec2<u32>)
        "read-only-storage", // spatialWeightBits
        "read-only-storage", // pixelsA
        "read-only-storage", // pixelsB
        "storage",           // usedJ
        "storage"            // assignments
    ],
        await loadShaderModule(GPU_DEVICE, "src/wgsl/calc_assignments.wgsl"),
        "calculateAssignments"
    )
    if (CALC_ASSIGNMENTS_PIPELINE == null) {
        console.error("Error loading shader...")
        return false
    }

    return true
}

// Single-pass GPU assignment.
// inputA / inputB are packed Uint32Array (RGBA) of length N = W*H.
// Returns a Promise<Uint32Array> of assignments, or null on failure.
async function assignPixelPositionsGPU(inputA, inputB, W, H) {

    const N = inputA.length

    // Create buffer to store the input size constant (N)
    const sizeConstantBuffer = createBufferU32(
        GPU_DEVICE,
        4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        new Uint32Array([N])
    )

    // Dimensions buffer (vec2<u32> = 8 bytes)
    const dimsBuffer = createBufferU32(
        GPU_DEVICE,
        8,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        new Uint32Array([W, H])
    )

    // Spatial weight as f32 bits packed in a u32.
    const f32sw = new Float32Array(1);
    f32sw[0] = CONFIG.SPATIAL_WEIGHT;
    const spatialWeightBuffer = createBufferU32(
        GPU_DEVICE,
        4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        new Uint32Array(f32sw.buffer)
    )

    // Create input buffers
    const _inputBufferA = createBufferU32(
        GPU_DEVICE,
        inputA.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        inputA
    )

    const _inputBufferB = createBufferU32(
        GPU_DEVICE,
        inputB.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        inputB
    )

    // Buffer for storing used positions
    const usedJBuffer = createBufferU32(
        GPU_DEVICE,
        N * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    )

    // Output buffer (initialized to N = "unclaimed")
    const assignmentsBuffer = createBufferU32(
        GPU_DEVICE,
        N * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        new Uint32Array(N).fill(N)
    )

    // Create bind group for the assignments calculation compute pass
    const bindGroup = createBindGroup(GPU_DEVICE, CALC_ASSIGNMENTS_PIPELINE,
        [
            sizeConstantBuffer,
            dimsBuffer,
            spatialWeightBuffer,
            _inputBufferA,
            _inputBufferB,
            usedJBuffer,
            assignmentsBuffer,
        ]
    )

    // Single compute pass.
    await runComputePass(GPU_DEVICE,
        CALC_ASSIGNMENTS_PIPELINE,
        bindGroup,
        N,
        DEFAULT_WORKGROUP_SIZE)

    // Single readback of the assignments buffer.
    return await computeBufferToCPUBuffer(
        GPU_DEVICE,
        assignmentsBuffer,
        N * 4
    )
}
