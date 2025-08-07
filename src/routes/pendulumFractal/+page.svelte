<script lang="ts">
    import { onMount, onDestroy } from 'svelte'
    import tgpu, {
        type StorageFlag,
        type TgpuBuffer,
        type TgpuRoot,
    } from 'typegpu'
    import * as d from 'typegpu/data'
    import computeShaderCode from '$lib/shaders/pendulumFractal/compute.wgsl?raw'
    import vertexShaderCode from '$lib/shaders/pendulumFractal/vert.wgsl?raw'
    import fragmentShaderCode from '$lib/shaders/pendulumFractal/frag.wgsl?raw'
    import { RollingAverage } from '$lib/RollingAverage'

    // Dynamically import all colorcet colormap CSVs using Vite's import.meta.glob
    const colormapModules = import.meta.glob('$lib/colorcet-maps/*.csv', {
        query: '?raw',
        import: 'default',
        eager: true,
    })

    const colormapList = Object.keys(colormapModules)
        .map((path) => path.split('/').pop()!)
        .sort()

    const colormapMap: Record<string, string> = {}
    for (const [path, raw] of Object.entries(colormapModules)) {
        const name = path.split('/').pop()!
        colormapMap[name] = raw as string
    }

    const fractalCanvasWidth = 720
    const fractalCanvasHeight = fractalCanvasWidth
    const sampledCanvasSize = 250

    let selectedColormap = $state('CET-C6s.csv')
    let colormapRaw = $derived(colormapMap[selectedColormap])
    let targetTps = $state(200)
    let measuredTps = $state(0)
    let measuredFps = $state(0)
    let zoomAmount = $state(1.0)
    let zoomFactor = $state(2.0)
    let zoomCenter = $state([0, 0])
    let timestep = 0.01
    let timestepTemp = $state(timestep)
    let reset: () => void = $state(() => {})
    let resetTps: () => void = $state(() => {})
    let resetInitialStates: () => void = $state(() => {})
    let resetShaders = $state(() => {})
    let root: TgpuRoot
    let fractalCanvas: HTMLCanvasElement
    let sampledCanvas: HTMLCanvasElement
    let gradientCanvas: HTMLCanvasElement

    let gridSize = fractalCanvasWidth
    const pixelCount = gridSize * gridSize

    let clickActions = $state([
        {
            id: 1,
            text: `Sample pendulum`,
        },
        {
            id: 2,
            text: `Zoom in`,
        },
    ])
    let selectedClickAction = $state(clickActions[0])
    let sampledPendulumXY = $state([
        Math.floor(gridSize / 2),
        Math.floor(gridSize / 2),
    ])
    let sampledPendulumLocation = $state([0, 0])
    let sampledPendulum = $state([0, 0])
    let stateBuffer: TgpuBuffer<d.WgslArray<d.Vec4f>> & StorageFlag

    function getXYCoordinates(e: MouseEvent) {
        const rect = fractalCanvas.getBoundingClientRect()
        const x = Math.floor((e.clientX - rect.left) / (rect.width / gridSize))
        const y =
            gridSize -
            1 -
            Math.floor((e.clientY - rect.top) / (rect.height / gridSize))
        return { x, y }
    }

    function getThetaCoordinates(x: number, y: number) {
        const theta1 =
            (Math.PI / zoomAmount) * ((x / (gridSize - 1)) * 2 - 1) +
            zoomCenter[0]
        const theta2 =
            (Math.PI / zoomAmount) * ((y / (gridSize - 1)) * 2 - 1) +
            zoomCenter[1]
        return [theta1, theta2]
    }

    function zoomIn(x: number, y: number) {
        const [theta1, theta2] = getThetaCoordinates(x, y)
        zoomAmount *= zoomFactor
        zoomCenter = [theta1, theta2]

        // Reset the sampled pendulum to the new zoom center
        sampledPendulumXY = [x, y]
        samplePendulum(sampledPendulumXY[0], sampledPendulumXY[1])
        resetInitialStates()
    }

    function loadColorMap(csv: string) {
        return csv
            .split('\n')
            .slice(1)
            .filter((line) => line.trim() !== '')
            .map((line) => {
                const [r, g, b] = line.split(',').map(Number)
                return d.vec3f(r / 255, g / 255, b / 255)
            })
    }

    async function samplePendulum(x: number, y: number) {
        const [theta1, theta2] = getThetaCoordinates(x, y)
        sampledPendulumXY = [x, y]
        sampledPendulumLocation = [theta1 / Math.PI, theta2 / Math.PI]

        // Read theta1/theta2 from GPU state buffer for the selected pixel
        if (!root) return
        const device = root.device
        const i = x + y * gridSize

        // Create a staging buffer to read back the state
        const readBuffer = device.createBuffer({
            size: 4 * 4, // vec4f = 4 floats = 16 bytes
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })

        // Copy the state for the selected pixel (theta1, omega1, theta2, omega2)
        const commandEncoder = device.createCommandEncoder()
        commandEncoder.copyBufferToBuffer(
            stateBuffer.buffer,
            i * 16, // offset in bytes
            readBuffer,
            0,
            16
        )
        device.queue.submit([commandEncoder.finish()])

        await readBuffer.mapAsync(GPUMapMode.READ)
        const arrayBuffer = readBuffer.getMappedRange()
        const f32 = new Float32Array(arrayBuffer)
        sampledPendulum = [f32[0], f32[2]]
        trace = []
        readBuffer.unmap()
        readBuffer.destroy()
    }

    function fractalCanvasClick(e: MouseEvent) {
        if (e.target !== fractalCanvas) {
            return
        }
        const { x, y } = getXYCoordinates(e)
        if (selectedClickAction.id === 1) {
            samplePendulum(x, y)
        } else if (selectedClickAction.id === 2) {
            zoomIn(x, y)
        }
    }

    onMount(async () => {
        root = await tgpu.init()
        const device = root.device

        const ctx = fractalCanvas.getContext('webgpu') as GPUCanvasContext
        const format = navigator.gpu.getPreferredCanvasFormat()
        ctx.configure({
            device,
            format,
            alphaMode: 'premultiplied',
        })

        // Track animation frame and resources for cleanup
        let animationFrameId: number | null = null
        let cleanupFns: (() => void)[] = []

        resetShaders = () => {
            // Cancel previous animation loop
            if (animationFrameId !== null) {
                cancelAnimationFrame(animationFrameId)
                animationFrameId = null
            }
            // Cleanup previous GPU resources
            cleanupFns.forEach((fn) => fn())
            cleanupFns = []
            timestep = timestepTemp
            trace = []

            drawGradient()

            // Create persistent sampled pendulum buffer
            const sampledPendulumBuffer = device.createBuffer({
                size: 4 * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            })
            cleanupFns.push(() => {
                if (sampledPendulumBuffer) sampledPendulumBuffer.destroy()
            })

            const computeModule = device.createShaderModule({
                code: computeShaderCode,
            })

            const uniformData = d.struct({
                dt: d.f32, // time step
                gravity: d.f32, // gravitational acceleration
                l1: d.f32, // length of first pendulum
                l2: d.f32, // length of second pendulum
                m1: d.f32, // mass of first pendulum
                m2: d.f32, // mass of second pendulum
                gridSize: d.u32, // grid size
            })

            const uniformBuffer = root
                .createBuffer(uniformData, {
                    dt: timestep, // time step
                    gravity: 10, // gravitational acceleration
                    l1: 1.0, // length of first pendulum
                    l2: 1.0, // length of second pendulum
                    m1: 1.0, // mass of first pendulum
                    m2: 1.0, // mass of second pendulum
                    gridSize,
                })
                .$usage('uniform')
            cleanupFns.push(() => uniformBuffer.destroy())

            stateBuffer = root
                .createBuffer(d.arrayOf(d.vec4f, pixelCount * 2))
                .$usage('storage')
            cleanupFns.push(() => stateBuffer.destroy())

            const initialStates: d.v4f[] = new Array(pixelCount * 2)
            resetInitialStates = () => {
                for (let x = 0; x < gridSize; x++) {
                    for (let y = 0; y < gridSize; y++) {
                        const i = x + y * gridSize
                        // Initial values for theta1 and theta2 are in the range [-pi, pi]
                        const theta1 =
                            (Math.PI / zoomAmount) *
                                ((x / (gridSize - 1)) * 2 - 1) +
                            zoomCenter[0]
                        const theta2 =
                            (Math.PI / zoomAmount) *
                                ((y / (gridSize - 1)) * 2 - 1) +
                            zoomCenter[1]
                        initialStates[i] = d.vec4f(theta1, 0, theta2, 0)

                        // Perturb the second pendulum slightly in phase space
                        // Perturbation amount in pixel units
                        const perturbation = 0.05
                        const r = Math.PI * (2 / (gridSize - 1)) * perturbation
                        const angle = Math.random() * 2 * Math.PI
                        const deltaTheta1 = r * Math.cos(angle)
                        const deltaTheta2 = r * Math.sin(angle)
                        initialStates[i + pixelCount] = d.vec4f(
                            theta1 + deltaTheta1,
                            0,
                            theta2 + deltaTheta2,
                            0
                        )
                    }
                }
                stateBuffer.write(initialStates)
            }
            resetInitialStates()

            const pixelsBufferStruct = d.arrayOf(
                d.struct({
                    energy: d.vec2f, // kinetic_energy, potential_energy
                    distance: d.f32, // distance between the 2 pendulums
                }),
                pixelCount
            )

            const pixelsBuffer = root
                .createBuffer(pixelsBufferStruct)
                .$usage('storage', 'vertex')
            cleanupFns.push(() => pixelsBuffer.destroy())

            const colorMap = loadColorMap(colormapRaw)
            const colorMapBuffer = root
                .createBuffer(d.arrayOf(d.vec3f, colorMap.length), colorMap)
                .$usage('storage', 'vertex')
            cleanupFns.push(() => colorMapBuffer.destroy())

            const computeBindGroupLayout = device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'uniform' },
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'storage' },
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'storage' },
                    },
                ],
            })

            const computeBindGroup = device.createBindGroup({
                layout: computeBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: uniformBuffer.buffer } },
                    { binding: 1, resource: { buffer: stateBuffer.buffer } },
                    { binding: 2, resource: { buffer: pixelsBuffer.buffer } },
                ],
            })

            const computePipeline = device.createComputePipeline({
                layout: device.createPipelineLayout({
                    bindGroupLayouts: [computeBindGroupLayout],
                }),
                compute: {
                    module: computeModule,
                    entryPoint: 'main',
                },
            })

            const vertexModule = device.createShaderModule({
                code: vertexShaderCode,
            })
            const fragmentModule = device.createShaderModule({
                code: fragmentShaderCode,
            })

            const renderBindGroupLayout = device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.VERTEX,
                        buffer: {
                            type: 'uniform',
                        },
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.VERTEX,
                        buffer: {
                            type: 'read-only-storage',
                        },
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.VERTEX,
                        buffer: {
                            type: 'read-only-storage',
                        },
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.VERTEX,
                        buffer: {
                            type: 'read-only-storage',
                        },
                    },
                ],
            })

            const gridSizeBuffer = root
                .createBuffer(d.u32, gridSize)
                .$usage('uniform')
            cleanupFns.push(() => gridSizeBuffer.destroy())

            const renderBindGroup = device.createBindGroup({
                layout: renderBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: gridSizeBuffer.buffer } },
                    { binding: 1, resource: { buffer: stateBuffer.buffer } },
                    { binding: 2, resource: { buffer: pixelsBuffer.buffer } },
                    { binding: 3, resource: { buffer: colorMapBuffer.buffer } },
                ],
            })

            const renderPipeline = device.createRenderPipeline({
                layout: device.createPipelineLayout({
                    bindGroupLayouts: [renderBindGroupLayout],
                }),
                vertex: {
                    module: vertexModule,
                    // entryPoint: 'main_distance',
                    entryPoint: 'main_angles',
                    buffers: [
                        {
                            arrayStride: 2 * Uint32Array.BYTES_PER_ELEMENT,
                            stepMode: 'vertex',
                            attributes: [
                                {
                                    shaderLocation: 0,
                                    offset: 0,
                                    format: 'uint32x2',
                                },
                            ],
                        },
                    ],
                },
                fragment: {
                    module: fragmentModule,
                    entryPoint: 'main',
                    targets: [{ format }],
                },
                primitive: {
                    topology: 'triangle-strip',
                },
            })

            const squareBuffer = root
                .createBuffer(d.arrayOf(d.u32, 4 * 2), [0, 0, 1, 0, 0, 1, 1, 1])
                .$usage('vertex')
            cleanupFns.push(() => squareBuffer.destroy())

            reset = () => {
                zoomCenter = [0, 0]
                zoomAmount = 1.0
                resetInitialStates()
            }

            let rollingAverageTps: RollingAverage
            let resetTps = () => {
                rollingAverageTps = new RollingAverage(5 * targetTps)
            }
            resetTps()

            let tick = 0
            function stepAndRender() {
                const encoder = device.createCommandEncoder()

                // Compute pass
                const computePass = encoder.beginComputePass()
                computePass.setPipeline(computePipeline)
                computePass.setBindGroup(0, computeBindGroup)
                computePass.dispatchWorkgroups(gridSize, gridSize)
                computePass.end()

                // Render pass
                const view = ctx.getCurrentTexture().createView()
                const renderPass = encoder.beginRenderPass({
                    colorAttachments: [
                        {
                            view,
                            clearValue: { r: 0, g: 0, b: 0, a: 1 },
                            loadOp: 'clear',
                            storeOp: 'store',
                        },
                    ],
                })

                renderPass.setPipeline(renderPipeline)
                renderPass.setVertexBuffer(0, squareBuffer.buffer)
                renderPass.setBindGroup(0, renderBindGroup)
                renderPass.draw(4, pixelCount)
                renderPass.end()

                device.queue.submit([encoder.finish()])

                tick++
            }

            let now = performance.now()
            let start = now
            let rollingAverageFps = new RollingAverage(5 * 60)
            async function loop() {
                now = performance.now()
                let elapsedMillis = now - start
                rollingAverageFps.push(1000 / elapsedMillis)
                measuredFps = rollingAverageFps.getAverage()

                if (elapsedMillis > 1000 / targetTps) {
                    measuredTps = rollingAverageTps.getAverage()
                    start = now

                    rollingAverageTps.push(1000 / elapsedMillis)
                    stepAndRender()
                }

                // Continuously update sampledPendulum from GPU buffer
                // Use a persistent staging buffer for sampled pendulum reads
                if (root && sampledPendulumXY) {
                    const i =
                        sampledPendulumXY[0] + sampledPendulumXY[1] * gridSize
                    if (stateBuffer) {
                        if (!sampledPendulumBuffer) return
                        const readBuffer = sampledPendulumBuffer
                        const commandEncoder = device.createCommandEncoder()
                        commandEncoder.copyBufferToBuffer(
                            stateBuffer.buffer,
                            i * 4 * 4,
                            readBuffer,
                            0,
                            4 * 4
                        )
                        device.queue.submit([commandEncoder.finish()])
                        await readBuffer.mapAsync(GPUMapMode.READ)
                        const arrayBuffer = readBuffer.getMappedRange()
                        const f32 = new Float32Array(arrayBuffer)
                        sampledPendulum = [f32[0], f32[2]]
                        readBuffer.unmap()
                    }
                }

                animationFrameId = requestAnimationFrame(loop)
                drawSampledPendulum(sampledPendulum[0], sampledPendulum[1])
            }

            animationFrameId = requestAnimationFrame(loop)
        }
        resetShaders()
    })

    onDestroy(() => {
        root?.destroy()
    })

    // Trace state for the second bob
    let trace: { x: number; y: number; alpha: number }[] = []
    const maxTraceLength = 1000
    const fadeStep = 0.999 // fade factor per frame

    function drawSampledPendulum(theta1: number, theta2: number) {
        if (!sampledCanvas) return
        const ctx = sampledCanvas.getContext('2d')
        if (!ctx) return

        ctx.clearRect(0, 0, sampledCanvas.width, sampledCanvas.height)
        let colorMap = loadColorMap(colormapRaw)
        const cmapLen = colorMap.length

        function angleToColorIdx(theta: number) {
            let norm = (((theta / (2 * Math.PI)) % 1) + 1) % 1
            return Math.floor(norm * (cmapLen - 1))
        }

        function rgb(arr: number[]) {
            return `rgb(${Math.round(arr[0] * 255)},${Math.round(arr[1] * 255)},${Math.round(arr[2] * 255)})`
        }

        const color1 = colorMap[angleToColorIdx(theta1)]
        const color2 = colorMap[angleToColorIdx(theta2)]

        // Pendulum parameters
        const margin = 20
        const maxLength =
            Math.min(sampledCanvas.width, sampledCanvas.height) / 2 - margin
        const l1 = maxLength * 0.5,
            l2 = maxLength * 0.5
        const m1 = 8,
            m2 = 8
        const origin = {
            x: sampledCanvas.width / 2,
            y: sampledCanvas.height / 2,
        }

        // Calculate positions
        const x1 = origin.x + l1 * Math.sin(theta1)
        const y1 = origin.y + l1 * Math.cos(theta1)
        const x2 = x1 + l2 * Math.sin(theta2)
        const y2 = y1 + l2 * Math.cos(theta2)

        // Add current position to trace
        trace.push({ x: x2, y: y2, alpha: 1.0 })
        if (trace.length > maxTraceLength) trace.shift()

        // Draw trace (white)
        for (let i = 1; i < trace.length; ++i) {
            const prev = trace[i - 1]
            const curr = trace[i]
            ctx.save()
            ctx.globalAlpha = curr.alpha * (i / trace.length)
            ctx.strokeStyle = '#fff'
            ctx.lineWidth = 2
            ctx.beginPath()
            ctx.moveTo(prev.x, prev.y)
            ctx.lineTo(curr.x, curr.y)
            ctx.stroke()
            ctx.restore()
        }
        // Fade trace alphas
        for (let t of trace) t.alpha *= fadeStep

        // Draw first arm
        ctx.globalAlpha = 1.0
        ctx.strokeStyle = rgb(color1)
        ctx.lineWidth = 3
        ctx.beginPath()
        ctx.moveTo(origin.x, origin.y)
        ctx.lineTo(x1, y1)
        ctx.stroke()

        // Draw second arm
        ctx.strokeStyle = rgb(color2)
        ctx.lineWidth = 3
        ctx.beginPath()
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
        ctx.stroke()

        // Draw first bob
        ctx.fillStyle = rgb(color1)
        ctx.beginPath()
        ctx.arc(x1, y1, m1, 0, 2 * Math.PI)
        ctx.fill()

        // Draw second bob
        ctx.fillStyle = rgb(color2)
        ctx.beginPath()
        ctx.arc(x2, y2, m2, 0, 2 * Math.PI)
        ctx.fill()
    }

    function drawGradient() {
        if (!gradientCanvas) return
        const ctx = gradientCanvas.getContext('2d')
        if (!ctx) return
        const colorMap = loadColorMap(colormapRaw)
        const w = gradientCanvas.width
        const h = gradientCanvas.height
        for (let x = 0; x < w; x++) {
            const t = x / (w - 1)
            const idx = Math.floor(t * (colorMap.length - 1))
            const rgbArr = colorMap[idx]
            ctx.fillStyle = `rgb(${Math.round(rgbArr[0] * 255)},${Math.round(rgbArr[1] * 255)},${Math.round(rgbArr[2] * 255)})`
            ctx.fillRect(x, 0, 1, h)
        }
    }
</script>

<main class="m-4 flex flex-row gap-4">
    <div class="flex flex-col">
        <canvas
            class="rounded-lg border border-gray-700"
            width={fractalCanvasWidth}
            height={fractalCanvasHeight}
            bind:this={fractalCanvas}
            onclick={fractalCanvasClick}
            style="width: {fractalCanvasWidth}px; height: {fractalCanvasHeight}px; flex: none; display: block;"
        >
        </canvas>
    </div>

    <div class="flex flex-col overflow-y-auto py-4 pr-4">
        <div class="flex flex-row flex-wrap items-start gap-6">
            <div class="flex min-w-[260px] flex-col gap-2">
                <fieldset class="fieldset m-2">
                    <!-- Click action buttons -->
                    <legend class="fieldset-legend">Click action</legend>
                    <div class="join">
                        {#each clickActions as clickAction}
                            <input
                                class="btn join-item"
                                type="radio"
                                name="clickAction"
                                aria-label={clickAction.text}
                                onclick={() =>
                                    (selectedClickAction = clickAction)}
                                checked={clickAction.id === 1}
                            />
                        {/each}
                    </div>

                    <!-- Color map selector -->
                    <div
                        class="tooltip tooltip-bottom"
                        data-tip="ColorCET color maps"
                    >
                        <legend class="fieldset-legend">Color map</legend>
                    </div>
                    <select
                        class="select"
                        bind:value={selectedColormap}
                        onchange={resetShaders}
                    >
                        {#each colormapList as cmap}
                            <option value={cmap}>{cmap}</option>
                        {/each}
                    </select>
                    <canvas
                        class="mt-2 h-4 w-full rounded-lg border border-gray-700"
                        bind:this={gradientCanvas}
                    ></canvas>

                    <!-- Tick rate slider -->
                    <legend class="fieldset-legend">
                        Tick rate
                        <span>{targetTps}</span>
                    </legend>
                    <input
                        type="range"
                        class="range"
                        min="1"
                        max="200"
                        onclick={resetTps}
                        bind:value={targetTps}
                    />
                    <p class="label">Simulation ticks per second</p>

                    <!-- Time step slider -->
                    <div
                        class="tooltip tooltip-bottom"
                        data-tip="Lower is slower, but more accurate"
                    >
                        <legend class="fieldset-legend">
                            Time step
                            <span>{timestepTemp.toFixed(3)}</span>
                        </legend>
                    </div>
                    <input
                        type="range"
                        class="range"
                        min="0.001"
                        max="0.02"
                        step="0.001"
                        onmouseup={resetShaders}
                        bind:value={timestepTemp}
                    />
                    <p class="label">Simulation time step</p>

                    <div class="divider"></div>

                    <!-- Zoom controls -->
                    <button class="btn btn-primary" onclick={reset}
                        >Reset zoom</button
                    >
                </fieldset>
            </div>

            <div class="flex min-w-[260px] flex-col items-center">
                <!-- Sampled pendulum display -->
                <span class="label font-semibold"
                    >Sampled pendulum initial angles</span
                >
                <span class="label font-mono">
                    ({(sampledPendulumLocation[0] >= 0 ? '+' : '') +
                        sampledPendulumLocation[0].toFixed(5)} pi,
                    {(sampledPendulumLocation[1] >= 0 ? '+' : '') +
                        sampledPendulumLocation[1].toFixed(5)} pi)
                </span>
                <canvas
                    bind:this={sampledCanvas}
                    width={sampledCanvasSize}
                    height={sampledCanvasSize}
                    class="rounded-lg border border-gray-700"
                ></canvas>

                <div class="divider"></div>

                <!-- Performance stats -->
                <div class="stats">
                    <div class="stat">
                        <div class="stat-title">Frames per second</div>
                        <div class="stat-value">{measuredFps.toFixed(0)}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-title">Ticks per second</div>
                        <div class="stat-value">{measuredTps.toFixed(0)}</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</main>

<footer
    class="footer sm:footer-horizontal footer-center bg-base-300 text-base-content p-4"
>
    <aside>
        <p>
            Made by
            <a href="https://github.com/ajs1998" class="link" target="_blank">
                Alex Sweeney
            </a>
        </p>

        <p>
            Inspired by 2swap's YouTube video
            <a
                href="https://www.youtube.com/watch?v=dtjb2OhEQcU"
                class="link"
                target="_blank"
            >
                Double Pendulums are Chaoticn't
            </a>
        </p>
    </aside>
</footer>
