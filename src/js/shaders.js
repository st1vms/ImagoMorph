let GPU_ADAPTER = undefined
let GPU_DEVICE = undefined

let SHADER_MODULE = undefined

let SHADER_PIPELINE = undefined

// Same as in wgsl files
const WORKGROUP_SIZE = 64

async function loadShader(path) {
    const response = await fetch(path);
    const shaderCode = await response.text();
    return GPU_DEVICE.createShaderModule({ code: shaderCode });
}

async function initShaders() {

    GPU_ADAPTER = await navigator.gpu.requestAdapter();
    if (GPU_ADAPTER == null) {
        console.error("Cannot retrieve GPU adapter...")
        return false
    }

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

    SHADER_MODULE = await loadShader("src\\wgsl\\nn.wgsl");
    if (SHADER_MODULE == null) {
        console.error("Failed to load nn.wgsl shader module",)
        return false
    }

    SHADER_PIPELINE = GPU_DEVICE.createComputePipeline({
        layout: GPU_DEVICE.createPipelineLayout({
            bindGroupLayouts: [GPU_DEVICE.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                ]
            })
            ]
        }),
        compute: { module: SHADER_MODULE, entryPoint: 'calculateNearestColorAssignments' },
    });

    return true
}

async function assignPixelPositionsGPU(pixelsA, pixelsB) {

    const pixelsAInput = packRGBAtoUint32(pixelsA)
    const pixelsBInput = packRGBAtoUint32(pixelsB)

    const N = pixelsAInput.length;

    // Buffer input RGBA float
    const pixelsABuffer = GPU_DEVICE.createBuffer({
        size: pixelsAInput.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    GPU_DEVICE.queue.writeBuffer(pixelsABuffer, 0, pixelsAInput);

    const pixelsBBuffer = GPU_DEVICE.createBuffer({
        size: pixelsBInput.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    GPU_DEVICE.queue.writeBuffer(pixelsBBuffer, 0, pixelsBInput);

    // Output assignment vector buffer
    const assignmentsArray = new Uint32Array(N).fill(0)
    const assignmentsBuffer = GPU_DEVICE.createBuffer({
        size: assignmentsArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    GPU_DEVICE.queue.writeBuffer(assignmentsBuffer, 0, assignmentsArray);

    // Synchronization vector buffer for flagging occupied pixels
    const usedJArray = new Uint32Array(N).fill(0);
    const usedJBuffer = GPU_DEVICE.createBuffer({
        size: usedJArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    GPU_DEVICE.queue.writeBuffer(usedJBuffer, 0, usedJArray);

    // Bind group creation
    const bindGroup = GPU_DEVICE.createBindGroup({
        layout: SHADER_PIPELINE.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: pixelsABuffer } },
            { binding: 1, resource: { buffer: pixelsBBuffer } },
            { binding: 2, resource: { buffer: assignmentsBuffer } },
            { binding: 3, resource: { buffer: usedJBuffer } },
        ]
    });

    // Prepare command encoder for compute pass
    const commandEncoder = GPU_DEVICE.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(SHADER_PIPELINE);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(N / WORKGROUP_SIZE));
    pass.end();

    // Perform assignment computations
    GPU_DEVICE.queue.submit([commandEncoder.finish()]);

    await GPU_DEVICE.queue.onSubmittedWorkDone();

    // Return a CPU buffer
    const readback = GPU_DEVICE.createBuffer({
        size: assignmentsArray.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const copyEncoder = GPU_DEVICE.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(assignmentsBuffer, 0, readback, 0, assignmentsArray.byteLength);
    GPU_DEVICE.queue.submit([copyEncoder.finish()]);

    await readback.mapAsync(GPUMapMode.READ);
    const assignments = new Uint32Array(readback.getMappedRange()).slice();

    readback.unmap();

    return assignments
}