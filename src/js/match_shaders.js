let GPU_ADAPTER = undefined
let GPU_DEVICE = undefined

const DEFAULT_WORKGROUP_SIZE = 64

let FIRST_ASSIGNMENT_PIPELINE = undefined
let CHECK_CONFLICT_PIPELINE = undefined
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

    FIRST_ASSIGNMENT_PIPELINE = await createShaderPipeline(
        GPU_DEVICE, [
        "read-only-storage",
        "read-only-storage",
        "read-only-storage",
        "storage"
    ],
        await loadShaderModule(GPU_DEVICE, "src\\wgsl\\first_assignment.wgsl"),
        "calculateFirstAssignments"
    )
    if (FIRST_ASSIGNMENT_PIPELINE == null) {
        console.error("Error loading shader...")
        return false
    }

    CHECK_CONFLICT_PIPELINE = await createShaderPipeline(
        GPU_DEVICE,
        [
            "read-only-storage",
            "read-only-storage",
            "storage",
        ],
        await loadShaderModule(GPU_DEVICE, "src\\wgsl\\check_conflicts.wgsl"),
        "checkConflicts"
    )
    if (CHECK_CONFLICT_PIPELINE == null) {
        console.error("Error loading shader...")
        return false
    }

    PRIORITY_CLAIMS_PIPELINE = await createShaderPipeline(
        GPU_DEVICE, [
        "read-only-storage",
        "read-only-storage",
        "storage",
        "storage"
    ],
        await loadShaderModule(GPU_DEVICE, "src\\wgsl\\priority_claims.wgsl"),
        "calculateClaimCountsAndPriorities"
    )
    if (PRIORITY_CLAIMS_PIPELINE == null) {
        console.error("Error loading shader...")
        return false
    }


    RESOLVE_CONFLICTS_PIPELINE = await createShaderPipeline(
        GPU_DEVICE, [
        "read-only-storage",
        "read-only-storage",
        "read-only-storage",
        "storage",
        "read-only-storage",
        "storage",
    ],
        await loadShaderModule(GPU_DEVICE, "src\\wgsl\\resolve_conflicts.wgsl"),
        "resolveConflicts"
    )
    if (RESOLVE_CONFLICTS_PIPELINE == null) {
        console.error("Error loading shader...")
        return false
    }

    return true
}

async function getFirstAssignments(inputA, inputB, inputLength) {

    const N = inputLength

    // Create buffer to store the input size constant (N)
    const sizeConstantBuffer = await createBufferU32(
        GPU_DEVICE,
        4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        new Uint32Array([N])
    )

    // Create input/output buffers
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

    const firstAssignmentsBuffer = await createBufferU32(
        GPU_DEVICE,
        N * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    )

    // Create bind group
    const bindGroup = await createBindGroup(GPU_DEVICE, FIRST_ASSIGNMENT_PIPELINE,
        [
            sizeConstantBuffer,
            _inputBufferA,
            _inputBufferB,
            firstAssignmentsBuffer,
        ]
    )

    // Run compute pass
    await runComputePass(GPU_DEVICE,
        FIRST_ASSIGNMENT_PIPELINE,
        bindGroup,
        N,
        DEFAULT_WORKGROUP_SIZE)

    // Destroy unused buffers
    sizeConstantBuffer.destroy()

    return {
        assignmentsBuffer: firstAssignmentsBuffer,
        inputBufferA: _inputBufferA,
        inputBufferB: _inputBufferB
    }
}


async function getClaimsAndPriorityBuffers(assignmentsBuffer, inputLength) {

    const N = inputLength

    // Create buffer to store the input size constant (N)
    const sizeConstantBuffer = await createBufferU32(
        GPU_DEVICE,
        4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        new Uint32Array([N])
    )

    const _claimsBuffer = await createBufferU32(
        GPU_DEVICE,
        N * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    )

    const _priorityBuffer = await createBufferU32(
        GPU_DEVICE,
        N * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    )

    const bindGroup = await createBindGroup(
        GPU_DEVICE,
        PRIORITY_CLAIMS_PIPELINE,
        [
            sizeConstantBuffer,
            assignmentsBuffer,
            _claimsBuffer,
            _priorityBuffer
        ]
    )

    await runComputePass(
        GPU_DEVICE,
        PRIORITY_CLAIMS_PIPELINE,
        bindGroup,
        N,
        DEFAULT_WORKGROUP_SIZE
    )

    return {
        claimsBuffer: _claimsBuffer,
        priorityBuffer: _priorityBuffer,
    }
}

async function checkSolved(claimsBuffer, inputLength) {

    const N = inputLength

    const sizeConstantBuffer = await createBufferU32(
        GPU_DEVICE,
        4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        new Uint32Array([N])
    )

    const resultBuffer = await createBufferU32(
        GPU_DEVICE,
        4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        new Uint32Array([1])
    )

    let resultBindGroup = await createBindGroup(
        GPU_DEVICE,
        CHECK_CONFLICT_PIPELINE,
        [
            sizeConstantBuffer,
            claimsBuffer,
            resultBuffer
        ]
    )

    await runComputePass(
        GPU_DEVICE,
        CHECK_CONFLICT_PIPELINE,
        resultBindGroup,
        N,
        DEFAULT_WORKGROUP_SIZE
    )

    // Read the result buffer
    const res = await computeBufferToCPUBuffer(
        GPU_DEVICE,
        resultBuffer,
        4
    )

    // 1 = Conflicts, 0 = No conflicts
    return res[0] === 1
}

async function resolveConflicts(
    inputBufferA,
    inputBufferB,
    claimsCountBuffer,
    prioritiesBuffer,
    assignmentsBuffer,
    inputLength
) {
    const N = inputLength

    const sizeConstantBuffer = await createBufferU32(
        GPU_DEVICE,
        4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        new Uint32Array([N])
    )

    const bindGroup = await createBindGroup(
        GPU_DEVICE,
        RESOLVE_CONFLICTS_PIPELINE,
        [
            sizeConstantBuffer,
            inputBufferA,
            inputBufferB,
            claimsCountBuffer,
            prioritiesBuffer,
            assignmentsBuffer,
        ]
    )

    await runComputePass(
        GPU_DEVICE,
        RESOLVE_CONFLICTS_PIPELINE,
        bindGroup,
        N,
        DEFAULT_WORKGROUP_SIZE
    )
}

async function assignPixelPositionsGPU(inputA, inputB) {

    await initShaders()

    if (GPU_DEVICE == null) {
        return
    }

    const N = inputA.length

    const assignmentsData = await getFirstAssignments(
        inputA,
        inputB,
        N * N
    )

    // Check if the first assignments have no conflicts
    let solved = false
    // Resolve conflict loop
    while (solved === false) {

        // Calculate claims count and priorities
        const claimsOrderResult = await getClaimsAndPriorityBuffers(
            assignmentsData.assignmentsBuffer,
            N
        )

        solved = await checkSolved(
            claimsOrderResult.claimsBuffer,
            N
        )
        if (solved === true) {
            break
        }

        await resolveConflicts(
            assignmentsData.inputBufferA,
            assignmentsData.inputBufferB,
            claimsOrderResult.claimsBuffer,
            claimsOrderResult.priorityBuffer,
            assignmentsData.assignmentsBuffer,
            N
        )
    }

    // Return the assignments as a CPU buffer
    return await computeBufferToCPUBuffer(
        GPU_DEVICE,
        assignmentsData.assignmentsBuffer,
        N * 4
    )
}
