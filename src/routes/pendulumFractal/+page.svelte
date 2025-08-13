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

    interface ColorCETMap {
        fileName: string
        displayName: string
        csv: string
    }

    // Dynamically import all colorcet colormap CSVs using Vite's import.meta.glob
    const colormapModules = import.meta.glob('$lib/colorcet-maps/*.csv', {
        query: '?raw',
        import: 'default',
        eager: true,
    })

    const cyclicColorMaps = Object.entries(colormapModules)
        .filter(([path]) => {
            return /CET-C\ds?\.csv$/.test(path)
        })
        .map(([path, csv]) => {
            const fileName = path.split('/').pop()!
            const displayName =
                'Cyclic ' +
                fileName.replace(/CET-C(\d)(s?)\.csv/, (_, num, s) =>
                    s ? `${num} shift 25%` : num
                )
            return { fileName, displayName, csv } as ColorCETMap
        })

    const fractalCanvasWidth = 720
    const fractalCanvasHeight = fractalCanvasWidth
    const sampledCanvasSize = 250

    let length1 = $state(1.0)
    let length2 = $state(1.0)
    let mass1 = $state(1.0)
    let mass2 = $state(1.0)
    let gravity = $state(10.0)
    let selectedColormap = $state(cyclicColorMaps[11])
    let colormapCsv = $derived(selectedColormap.csv)
    let measuredTps = $state(0)
    let measuredFps = $state(0)
    let targetTicksPerSecond = $state(500)
    let millisAccumulator = 0
    let lastTickTime = performance.now()
    let zoomAmount = $state(1.0)
    let zoomFactor = $state(2.0)
    let zoomCenter = $state([0, 0])
    let timestep = 0.005
    let timestepTemp = $state(timestep)
    let visualizationModeBuffer: TgpuBuffer<typeof d.u32>;
    let reset: () => void = $state(() => {})
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
            id: 0,
            text: `Sample pendulum`,
        },
        {
            id: 1,
            text: `Zoom in`,
        },
    ])
    let visualizationModes = $state([
        {
            id: 0,
            label: `Angle 1`,
        },
        {
            id: 1,
            label: `Energy loss`,
        },
        {
            id: 2,
            label: `Sensitivity`,
        },
    ])

    let selectedClickAction = $state(clickActions[0])
    let selectedVisualizationMode = $state(visualizationModes[0])
    let sampledPendulumXY = $state([
        Math.floor(gridSize / 2),
        Math.floor(gridSize / 2),
    ])
    let sampledPendulumLocation = $state([0, 0])
    let sampledPendulum = $state([0, 0, 0, 0])
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
        sampledPendulumLocation = [theta1 / Math.PI, theta2 / Math.PI]
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
        const sampledPendulumStateBuffer = readBuffer.getMappedRange()
        const sampledPendulumState = new Float32Array(sampledPendulumStateBuffer)
        sampledPendulum = [sampledPendulumState[0], sampledPendulumState[1], sampledPendulumState[2], sampledPendulumState[3]]
        readBuffer.unmap()
        readBuffer.destroy()
    }

    function fractalCanvasClick(e: MouseEvent) {
        if (e.target !== fractalCanvas) {
            return
        }
        const { x, y } = getXYCoordinates(e)
        if (selectedClickAction.id === 0) {
            const [theta1, theta2] = getThetaCoordinates(x, y)
            sampledPendulumXY = [x, y]
            sampledPendulumLocation = [theta1 / Math.PI, theta2 / Math.PI]
            samplePendulum(x, y)
            trace = []
        } else if (selectedClickAction.id === 1) {
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
        let simulationTimerId: number | null = null
        let cleanupFns: (() => void)[] = []

        resetShaders = () => {
            // Cancel previous animation loop
            if (animationFrameId !== null) {
                cancelAnimationFrame(animationFrameId)
                animationFrameId = null
            }
            // Cancel previous simulation loop
            if (simulationTimerId !== null) {
                clearTimeout(simulationTimerId)
                simulationTimerId = null
            }
            // Cleanup previous GPU resources
            cleanupFns.forEach((fn) => fn())
            cleanupFns = []
            timestep = timestepTemp
            trace = []

            drawGradient()

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
                    gravity: gravity, // gravitational acceleration
                    l1: length1, // length of first pendulum
                    l2: length2, // length of second pendulum
                    m1: mass1, // mass of first pendulum
                    m2: mass2, // mass of second pendulum
                    gridSize,
                })
                .$usage('uniform')
            cleanupFns.push(() => uniformBuffer.destroy())

            const pixelsBufferStruct = d.arrayOf(
                d.struct({
                    energy: d.vec2f, // kinetic_energy, potential_energy
                    initial_energy: d.f32, // total initial energy
                    distance: d.f32, // distance between the 2 pendulums
                }),
                pixelCount
            )
            const pixelsBuffer = root
                .createBuffer(pixelsBufferStruct)
                .$usage('storage', 'vertex')
            cleanupFns.push(() => pixelsBuffer.destroy())

            stateBuffer = root
                .createBuffer(d.arrayOf(d.vec4f, pixelCount * 2))
                .$usage('storage')
            cleanupFns.push(() => stateBuffer.destroy())

            const initialStates: d.v4f[] = new Array(pixelCount * 2)
            const initialEnergies = new Array(pixelCount)
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
                        let potential_energy = -(mass1 + mass2) * length1 * gravity * Math.cos(theta1) 
                            - mass2 * length2 * gravity * Math.cos(theta2);
                        initialEnergies[i] = d.f32(potential_energy)
                    }
                }
                stateBuffer.write(initialStates)
                pixelsBuffer.write(initialEnergies.map(e => {
                    return {
                        energy: d.vec2f(0, 0),
                        initial_energy: e,
                        distance: 0.0
                    };
                }))
                trace = []
            }
            resetInitialStates()

            // Visualization mode uniform buffer
            visualizationModeBuffer = root
                .createBuffer(d.u32, selectedVisualizationMode.id)
                .$usage('uniform')
            cleanupFns.push(() => visualizationModeBuffer.destroy())

            const colorMap = loadColorMap(colormapCsv)
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
                    {
                        binding: 4,
                        visibility: GPUShaderStage.VERTEX,
                        buffer: {
                            type: 'uniform',
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
                    { binding: 4, resource: { buffer: visualizationModeBuffer.buffer } },
                ],
            })

            const renderPipeline = device.createRenderPipeline({
                layout: device.createPipelineLayout({
                    bindGroupLayouts: [renderBindGroupLayout],
                }),
                vertex: {
                    module: vertexModule,
                    entryPoint: 'main',
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

            function runComputePass() {
                const encoder = device.createCommandEncoder()
                const computePass = encoder.beginComputePass()
                computePass.setPipeline(computePipeline)
                computePass.setBindGroup(0, computeBindGroup)
                computePass.dispatchWorkgroups(gridSize, gridSize)
                computePass.end()
                device.queue.submit([encoder.finish()])
            }

            function runRenderPass() {
                const encoder = device.createCommandEncoder()
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
            }

            let rollingAverageTps = new RollingAverage(5 * 60)
            let rollingAverageFps = new RollingAverage(5 * 60)

            function simulationLoop() {
                const now = performance.now()
                const elapsedMillis = now - lastTickTime
                lastTickTime = now
                millisAccumulator += elapsedMillis

                // Compute steps at fixed tick rate
                let ticks = 0
                const millisPerTick = 1000 / targetTicksPerSecond
                while (millisAccumulator >= millisPerTick) {
                    runComputePass()
                    millisAccumulator -= millisPerTick
                    ticks++
                }
                if (elapsedMillis > 0) {
                    rollingAverageTps.push(ticks / (elapsedMillis / 1000))
                }
                measuredTps = rollingAverageTps.getAverage()

                simulationTimerId = window.setTimeout(simulationLoop, 0) // Run as fast as possible
            }

            let lastRenderTime = 0
            async function renderLoop() {
                const now = performance.now()
                const elapsed = now - lastRenderTime
                lastRenderTime = now

                // Render
                runRenderPass()

                // Correct FPS calculation: time between animation frames
                if (elapsed > 0) {
                    rollingAverageFps.push(1000 / elapsed)
                }
                measuredFps = rollingAverageFps.getAverage()

                // Continuously update sampledPendulum from GPU buffer
                samplePendulum(sampledPendulumXY[0], sampledPendulumXY[1])
                drawSampledPendulum(sampledPendulum[0], sampledPendulum[1], sampledPendulum[2], sampledPendulum[3])
                animationFrameId = requestAnimationFrame(renderLoop)
            }

            simulationLoop()
            animationFrameId = requestAnimationFrame(renderLoop)
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

    function drawSampledPendulum(theta1: number, omega1: number, theta2: number, omega2: number) {
        if (!sampledCanvas) return
        const ctx = sampledCanvas.getContext('2d')
        if (!ctx) return

        ctx.clearRect(0, 0, sampledCanvas.width, sampledCanvas.height)
        let colorMap = loadColorMap(colormapCsv)
        const cmapLen = colorMap.length

        function angleToColorIndex(theta: number) {
            let norm = (((theta / (2 * Math.PI)) % 1) + 1) % 1
            return Math.floor(norm * (cmapLen - 1))
        }

        function rgb(arr: number[]) {
            return `rgb(${Math.round(arr[0] * 255)},${Math.round(arr[1] * 255)},${Math.round(arr[2] * 255)})`
        }

        let baseContentColor = getComputedStyle(document.documentElement)
                .getPropertyValue('--color-base-content')
                .trim();
        let color1 = baseContentColor
        let color2 = baseContentColor
        let traceColor = color1
        if (selectedVisualizationMode.id === 0) {
            color1 = rgb(colorMap[angleToColorIndex(theta1)])
        }

        // Rendered pendulum parameters
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

        for (let i = 1; i < trace.length; ++i) {
            const prev = trace[i - 1]
            const curr = trace[i]
            ctx.save()
            ctx.globalAlpha = curr.alpha * (i / trace.length)
            ctx.strokeStyle = traceColor
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
        ctx.strokeStyle = color1
        ctx.lineWidth = 3
        ctx.beginPath()
        ctx.moveTo(origin.x, origin.y)
        ctx.lineTo(x1, y1)
        ctx.stroke()

        // Draw second arm
        ctx.strokeStyle = color2
        ctx.lineWidth = 3
        ctx.beginPath()
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
        ctx.stroke()

        // Draw first bob
        ctx.fillStyle = color1
        ctx.beginPath()
        ctx.arc(x1, y1, m1, 0, 2 * Math.PI)
        ctx.fill()

        // Draw second bob
        ctx.fillStyle = color2
        ctx.beginPath()
        ctx.arc(x2, y2, m2, 0, 2 * Math.PI)
        ctx.fill()
    }

    function drawGradient() {
        if (!gradientCanvas) return
        const ctx = gradientCanvas.getContext('2d')
        if (!ctx) return
        const colorMap = loadColorMap(colormapCsv)
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

<main class="m-4 md:overflow-x-scroll">
    <div class="flex flex-col md:flex-row md:gap-4 md:p-2">
        <div class="flex-none">
            <canvas
                class="skeleton w-full border border-gray-700 md:rounded-lg"
                width={fractalCanvasWidth}
                height={fractalCanvasHeight}
                bind:this={fractalCanvas}
                onclick={fractalCanvasClick}
            >
            </canvas>
        </div>
        <div class="m-4 flex-2 md:m-0">
            <!-- Sampled pendulum display -->
            <div class="justify-items-center">
                <canvas
                    width={sampledCanvasSize}
                    height={sampledCanvasSize}
                    bind:this={sampledCanvas}
                    class="bg-base-300 justify-center rounded-lg border border-gray-700"
                ></canvas>
                <span class="label w-full justify-center font-mono">
                    ({(sampledPendulumLocation[0] >= 0 ? '+' : '') +
                        sampledPendulumLocation[0].toFixed(5)} pi,
                    {(sampledPendulumLocation[1] >= 0 ? '+' : '') +
                        sampledPendulumLocation[1].toFixed(5)} pi)
                </span>
                <span class="label w-full justify-center">
                    Sampled pendulum initial angles
                </span>
            </div>

            <fieldset class="fieldset">
                <!-- Click action buttons -->
                <legend class="fieldset-legend">Click action</legend>
                <div class="join join-vertical">
                    {#each clickActions as clickAction}
                        <input
                            class="btn join-item"
                            type="radio"
                            name="clickAction"
                            aria-label={clickAction.text}
                            onclick={() => (selectedClickAction = clickAction)}
                            checked={selectedClickAction === clickAction}
                        />
                    {/each}
                </div>

                <!-- Zoom controls -->
                <legend class="fieldset-legend">Zoom controls</legend>
                <button class="btn btn-primary" onclick={reset}>
                    Reset zoom
                </button>

                <!-- Energy visualization mode toggle -->
                <legend class="fieldset-legend">Visualization Mode</legend>
                <div class="join join-horizontal">
                    {#each visualizationModes as visualizationMode}
                        <input
                            class="btn join-item"
                            type="radio"
                            name="visualizationMode"
                            aria-label={visualizationMode.label}
                            onclick={() => {
                                selectedVisualizationMode = visualizationMode
                                if (visualizationMode.id === 0) {
                                    selectedColormap = cyclicColorMaps[11]
                                } else if (visualizationMode.id === 1) {
                                    selectedColormap = cyclicColorMaps[9]
                                } else if (visualizationMode.id === 2) {
                                    // TODO Linear color map
                                    selectedColormap = cyclicColorMaps[9]
                                }
                                resetShaders()
                            }}
                            checked={selectedVisualizationMode === visualizationMode}
                        />
                    {/each}
                </div>
                
                <!-- Color map selector -->
                <legend class="fieldset-legend">Color map</legend>
                <select
                    class="select w-full"
                    bind:value={selectedColormap}
                    onchange={resetShaders}
                >
                    {#each cyclicColorMaps as cmap}
                        <option value={cmap}>{cmap.displayName}</option>
                    {/each}
                </select>
                <canvas
                    class="h-4 w-full rounded-lg border border-gray-700"
                    bind:this={gradientCanvas}
                ></canvas>
                <a
                    class="label link"
                    href="https://colorcet.com/gallery.html"
                    target="_blank"
                >
                    ColorCET maps
                </a>
            </fieldset>
        </div>

        <div class="m-4 flex-1 md:m-0">
            <!-- Performance stats -->
            <div class="stats flex justify-items-center">
                <div class="stat">
                    <div class="stat-title">Ticks per second</div>
                    <div class="stat-value font-mono">{Math.abs(measuredTps).toFixed(0)}</div>
                </div>
                <div class="stat">
                    <div class="stat-title">Frames per second</div>
                    <div class="stat-value font-mono">{measuredFps.toFixed(0)}</div>
                </div>
            </div>

            <!-- Compute steps per frame slider -->
            <legend class="fieldset-legend">
                Ticks per second
                {#if targetTicksPerSecond === 0}
                    <span class="text-warning">Paused</span>
                {:else}
                    <span class="font-mono">{targetTicksPerSecond}</span>
                {/if}
            </legend>
            <input
                type="range"
                class="range w-full"
                min="0"
                max="5000"
                bind:value={targetTicksPerSecond}
            />
            <p class="label w-full">Simulation steps per second</p>

            <!-- Time step slider -->
            <legend class="fieldset-legend">
                Time step
                <span class="font-mono">{timestepTemp.toFixed(4)}</span>
            </legend>
            <input
                type="range"
                class="range w-full"
                min="0.0001"
                max="0.01"
                step="0.0001"
                onmouseup={resetShaders}
                bind:value={timestepTemp}
            />
            <div
                class="tooltip tooltip-bottom"
                data-tip="Lower is slower, but more accurate"
            >
                <p class="label w-full">Simulation time step</p>
            </div>
        </div>
    </div>
</main>
