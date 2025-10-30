let GPU_ADAPTER = undefined
let GPU_DEVICE = undefined

const DEFAULT_WORKGROUP_SIZE = 64

let FIRST_ASSIGNMENT_PIPELINE = undefined
let CHECK_SOLVED_PIPELINE = undefined
let PRIORITY_CLAIMS_PIPELINE = undefined
let IDENTIFY_CONFLICTS_PIPELINE = undefined
let RESOLVE_CONFLICTS_PIPELINE = undefined

async function loadShaderModule(device, path) {
    const response = await fetch(path);
    const shaderCode = await response.text();
    return device.createShaderModule({ code: shaderCode });
}

async function createBindGroup(device, pipeline, binding_buffers) {

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

async function createShaderPipeline(device, binding_buffer_types, shaderModule, entryPoint) {

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

async function runComputePass(device, pipeline, bindGroup, inputLength, workgroup_size = 64) {
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

    await device.queue.onSubmittedWorkDone();
}

async function createBufferU32(device, byteSize, usage, copySrcBuffer = null) {

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
    GPU_ADAPTER = await navigator.gpu.requestAdapter();
    if (GPU_ADAPTER == null) {
        console.error("Cannot retrieve GPU adapter...")
        return false
    }

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

    // Load shader modules
    CHECK_SOLVED_PIPELINE = await createShaderPipeline(
        GPU_DEVICE,
        [
            "read-only-storage",
            "storage",
            "storage",
        ],
        await loadShaderModule(GPU_DEVICE, "src/wgsl/check_solved.wgsl"),
        "checkSolved"
    )
    if (CHECK_SOLVED_PIPELINE == null) {
        console.error("Error loading shader...")
        return false
    }

    FIRST_ASSIGNMENT_PIPELINE = await createShaderPipeline(
        GPU_DEVICE, [
        "read-only-storage",
        "read-only-storage",
        "read-only-storage",
        "storage",
        "storage"
    ],
        // TODO Change this
        await loadShaderModule(GPU_DEVICE, "src/wgsl/calc_assignments.wgsl"),
        "calculateAssignments"
    )
    if (FIRST_ASSIGNMENT_PIPELINE == null) {
        console.error("Error loading shader...")
        return false
    }

    return true
}

async function checkSolved(usedJBuffer, N) {

    const sizeConstantBuffer = await createBufferU32(
        GPU_DEVICE,
        4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        new Uint32Array([N])
    )

    const successResultBuffer = await createBufferU32(
        GPU_DEVICE,
        4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        new Uint32Array([1]) // Default 1: Success
    )
    let successResultBindGroup = await createBindGroup(
        GPU_DEVICE,
        CHECK_SOLVED_PIPELINE,
        [
            sizeConstantBuffer,
            usedJBuffer,
            successResultBuffer
        ]
    )

    // Check if all positions are occupied
    await runComputePass(
        GPU_DEVICE,
        CHECK_SOLVED_PIPELINE,
        successResultBindGroup,
        N,
        DEFAULT_WORKGROUP_SIZE
    )

    // Read the result buffer
    const success = await computeBufferToCPUBuffer(
        GPU_DEVICE,
        successResultBuffer,
        4
    )

    return success[0] === 1
}

async function assignPixelPositionsGPU(inputA, inputB) {

    const N = inputA.length

    // Create buffer to store the input size constant (N)
    const sizeConstantBuffer = await createBufferU32(
        GPU_DEVICE,
        4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        new Uint32Array([N])
    )

    // Create input buffers
    const _inputBufferA = await createBufferU32(
        GPU_DEVICE,
        inputA.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        inputA
    )

    const _inputBufferB = await createBufferU32(
        GPU_DEVICE,
        inputB.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        inputB
    )

    // Buffer for storing used positions
    const usedJBuffer = await createBufferU32(
        GPU_DEVICE,
        N * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    )

    // Output buffer
    const assignmentsBuffer = await createBufferU32(
        GPU_DEVICE,
        N * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        new Uint32Array(N).fill(N)
    )

    // Create bind group for the assignments calculation compute pass
    const bindGroup = await createBindGroup(GPU_DEVICE, FIRST_ASSIGNMENT_PIPELINE,
        [
            sizeConstantBuffer,
            _inputBufferA,
            _inputBufferB,
            usedJBuffer,
            assignmentsBuffer,
        ]
    )

    // Keep running compute pass to calculate positions
    for (let i = 0; i < N; i++) {
        await runComputePass(GPU_DEVICE,
            FIRST_ASSIGNMENT_PIPELINE,
            bindGroup,
            N,
            DEFAULT_WORKGROUP_SIZE)

        // OPTIONAL
        // Use the 1s count of usedJ to determine a progress status
        const perc_status = Math.round((Array.from(await computeBufferToCPUBuffer(
            GPU_DEVICE,
            usedJBuffer,
            N * 4
        )).filter(v => v === 1).length / N) * 100 * 100, 2) / 100
        console.log(`Completion: ${perc_status}%`)

        // 0 = some cells were left empty
        if (await checkSolved(usedJBuffer, N)) {
            break;
        }
    }

    // Return assignments buffer as UInt32Array
    return await computeBufferToCPUBuffer(
        GPU_DEVICE,
        assignmentsBuffer,
        N * 4
    )
}
