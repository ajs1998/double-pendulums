<script lang="ts">
    import { onMount, onDestroy } from 'svelte'
    import tgpu, { type TgpuBuffer, type TgpuRoot } from 'typegpu'
    import * as d from 'typegpu/data'
    import computeShaderCode from '$lib/shaders/pendulumFractal/compute.wgsl?raw'
    import vertexShaderCode from '$lib/shaders/pendulumFractal/vert.wgsl?raw'
    import fragmentShaderCode from '$lib/shaders/pendulumFractal/frag.wgsl?raw'
    import { RollingAverage } from '$lib/RollingAverage'
    import { cyclicColorMaps, linearColorMaps } from '$lib/ColorCET'

    const sampledCanvasSize = 300

    // Crosshair overlay state
    let showCrosshair = $state(true)
    const defaultCyclicColorMap = cyclicColorMaps[6]
    const defaultDivergingColorMap = cyclicColorMaps[6]
    const defaultLinearColorMap = linearColorMaps[15]

    let length1 = $state(1.0)
    let length2 = $state(1.0)
    let mass1 = $state(1.0)
    let mass2 = $state(1.0)
    let gravity = $state(10.0)
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
    let visualizationModeBuffer: TgpuBuffer<typeof d.u32>
    let resetZoom: () => void = $state(() => {})
    let resetPendulums: () => void = $state(() => {})
    let resetShaders = $state(() => {})
    let root: TgpuRoot
    let fractalCanvas: HTMLCanvasElement
    let crosshairCanvas: HTMLCanvasElement
    let sampledCanvas: HTMLCanvasElement
    let gradientCanvas: HTMLCanvasElement

    // 8 * 8 = 64 threads per workgroup
    // https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html
    const workgroupSize = 8
    // Canvas size is a multiple of the workgroup size
    // 8 * 90 = 720 pixels
    const workgroupCount = 90
    const gridSize = workgroupSize * workgroupCount
    const pixelCount = gridSize * gridSize
    const centerXY = [Math.floor(gridSize / 2), Math.floor(gridSize / 2)]
    const maxTicksPerSecond = 5000

    let colorMaps = $state([...cyclicColorMaps, ...linearColorMaps])
    let selectedColormap = $state(defaultCyclicColorMap)
    let colormapCsv = $derived(selectedColormap.csv)

    interface ClickAction {
        id: number
        name: string
    }

    let clickActions: ClickAction[] = $state([
        {
            id: 0,
            name: `Sample pendulum`,
        },
        {
            id: 1,
            name: `Zoom in`,
        },
        {
            id: 2,
            name: `Zoom out`,
        },
    ])

    interface ResetControl {
        name: string
        onclick: () => void
    }

    let resetControls: ResetControl[] = $state([
        {
            name: `Reset pendulums`,
            onclick: () => resetPendulums(),
        },
        {
            name: `Reset zoom`,
            onclick: () => resetZoom(),
        },
    ])

    interface VisualizationMode {
        id: number
        name: string
    }

    let visualizationModes: VisualizationMode[] = $state([
        {
            id: 0,
            name: `Angle 1`,
        },
        {
            id: 1,
            name: `Angle 2`,
        },
        {
            id: 2,
            name: `Sensitivity`,
        },
        {
            id: 3,
            name: `Energy loss`,
        },
    ])

    let selectedClickAction = $state(clickActions[0])
    let selectedVisualizationMode = $state(visualizationModes[0])
    let sampledPendulumXY = $state(centerXY)
    let sampledPendulumLocation = $state([0, 0]) // Theta coordinates
    let sampledPendulum = $state([0, 0, 0, 0])
    let stateBuffer: TgpuBuffer<d.WgslArray<d.Vec4f>>

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

    // Sample both pendulums for sensitivity mode
    async function samplePendulum(x: number, y: number) {
        const device = root.device
        const i = x + y * gridSize

        // Create staging buffers to read both states
        const readBufferA = root.createBuffer(d.vec4f)
        const readBufferB = root.createBuffer(d.vec4f)

        // Copy the state for the selected pixel (main pendulum)
        const commandEncoder = device.createCommandEncoder()
        commandEncoder.copyBufferToBuffer(
            stateBuffer.buffer,
            i * d.sizeOf(d.vec4f), // offset in bytes
            readBufferA.buffer,
            0,
            d.sizeOf(d.vec4f)
        )
        // Copy the perturbed state for the selected pixel (second pendulum)
        commandEncoder.copyBufferToBuffer(
            stateBuffer.buffer,
            (i + pixelCount) * d.sizeOf(d.vec4f),
            readBufferB.buffer,
            0,
            d.sizeOf(d.vec4f)
        )
        device.queue.submit([commandEncoder.finish()])
        const mainPendulum = await readBufferA.read()
        const perturbedPendulum = await readBufferB.read()
        sampledPendulum = [mainPendulum[0], mainPendulum[1], mainPendulum[2], mainPendulum[3],
                           perturbedPendulum[0], perturbedPendulum[1], perturbedPendulum[2], perturbedPendulum[3]]
        readBufferA.destroy()
        readBufferB.destroy()
    }

    function zoom(x: number, y: number, factor: number) {
        const [theta1, theta2] = getThetaCoordinates(x, y)
        zoomAmount *= factor
        zoomCenter = [theta1, theta2]

        // Reset the sampled pendulum to the new zoom center
        samplePendulum(x, y)
        resetPendulums()
    }

    function zoomIn(x: number, y: number) {
        zoom(x, y, zoomFactor)
    }

    function zoomOut(x: number, y: number) {
        zoom(x, y, 1 / zoomFactor)
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

    function fractalCanvasClick(e: MouseEvent) {
        if (e.target !== fractalCanvas) {
            return
        }
        const { x, y } = getXYCoordinates(e)
        const [theta1, theta2] = getThetaCoordinates(x, y)
        sampledPendulumLocation = [theta1 / Math.PI, theta2 / Math.PI]
        sampledPendulumXY = [x, y]
        trace = []
        if (selectedClickAction.id === 0) {
            samplePendulum(x, y)
        } else if (selectedClickAction.id === 1) {
            sampledPendulumXY = centerXY
            zoomIn(x, y)
        } else if (selectedClickAction.id === 2) {
            sampledPendulumXY = centerXY
            zoomOut(x, y)
        }
    }

    onMount(async () => {
        root = await tgpu.init()
        const device = root.device

        const context = fractalCanvas.getContext('webgpu') as GPUCanvasContext
        const format = navigator.gpu.getPreferredCanvasFormat()
        context.configure({
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

            drawColormapPreview()

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
                    energy: d.vec3f, // initial_energy, kinetic_energy, potential_energy
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
            resetPendulums = () => {
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
                        const perturbation = 0.5
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
                        let potential_energy =
                            -(mass1 + mass2) *
                                length1 *
                                gravity *
                                Math.cos(theta1) -
                            mass2 * length2 * gravity * Math.cos(theta2)
                        initialEnergies[i] = d.f32(potential_energy)
                    }
                }
                stateBuffer.write(initialStates)
                pixelsBuffer.write(
                    initialEnergies.map((e) => {
                        return {
                            energy: d.vec3f(e, 0, 0),
                            distance: 0.0,
                        }
                    })
                )
                trace = []
            }
            resetPendulums()

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
                    constants: {
                        workgroupSize: workgroupSize,
                    }
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
                    {
                        binding: 4,
                        resource: { buffer: visualizationModeBuffer.buffer },
                    },
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

            resetZoom = () => {
                zoomCenter = [0, 0]
                zoomAmount = 1.0
                resetPendulums()
            }

            function runComputePass() {
                const encoder = device.createCommandEncoder()
                const computePass = encoder.beginComputePass()
                computePass.setPipeline(computePipeline)
                computePass.setBindGroup(0, computeBindGroup)
                computePass.dispatchWorkgroups(workgroupCount, workgroupCount)
                computePass.end()
                device.queue.submit([encoder.finish()])
            }

            function runRenderPass() {
                const encoder = device.createCommandEncoder()
                const view = context.getCurrentTexture().createView()
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
                // Draw crosshair after fractal rendering
                drawCrosshair()

                // Correct FPS calculation: time between animation frames
                if (elapsed > 0) {
                    rollingAverageFps.push(1000 / elapsed)
                }
                measuredFps = rollingAverageFps.getAverage()

                baseContentColor = getCssValue('--color-base-content')

                // Continuously update sampledPendulum from GPU buffer
                samplePendulum(sampledPendulumXY[0], sampledPendulumXY[1])
                drawSampledPendulum(
                    sampledPendulum[0],
                    sampledPendulum[2],
                )
                drawCrosshair()
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

    // Trace state for the sampled pendulum visualization
    // For sensitivity mode, trace is an array of lines between bobs with color and alpha
    // For other modes, trace is an array of bob positions with alpha
    let trace: {
        x1: number; y1: number;
        x2: number; y2: number;
        color: string; alpha: number
    }[] = []
    const maxTraceLength = 1000
    const fadeStep = 0.999 // fade factor per frame
    let baseContentColor: string

    function getCssValue(variable: string) {
        return getComputedStyle(document.documentElement)
            .getPropertyValue(variable)
            .trim()
    }

    function drawSampledPendulum(
        theta1: number,
        theta2: number,
    ) {
        if (!sampledCanvas) return
        const context = sampledCanvas.getContext('2d')
        if (!context) return

        context.clearRect(0, 0, sampledCanvas.width, sampledCanvas.height)
        // TODO
        let colorMap = loadColorMap(colormapCsv)

        // TODO
        function angleToColorIndex(theta: number) {
            let norm = (((theta / (2 * Math.PI)) % 1) + 1) % 1
            return Math.floor(norm * (colorMap.length - 1))
        }

        // TODO
        function rgb(arr: number[]) {
            return `rgb(${Math.round(arr[0] * 255)},${Math.round(arr[1] * 255)},${Math.round(arr[2] * 255)})`
        }

        let color1 = baseContentColor
        let color2 = baseContentColor
        let traceColor = color1

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

        // Calculate positions for first pendulum
        const x1 = origin.x + l1 * Math.sin(theta1)
        const y1 = origin.y + l1 * Math.cos(theta1)
        const x2 = x1 + l2 * Math.sin(theta2)
        const y2 = y1 + l2 * Math.cos(theta2)

        if (selectedVisualizationMode.id === 2) {
            // Sensitivity mode: draw both pendulums and trace line between bobs
            // Get perturbed pendulum state from stateBuffer (second pendulum)
            // This assumes sampledPendulumXY is up to date
            const i = sampledPendulumXY[0] + sampledPendulumXY[1] * gridSize
            // Read perturbed state from stateBuffer (already available in initialStates, but here use sampledPendulum for first, and need to read second)
            // For simplicity, use sampledPendulum for first, and reuse theta1/theta2 for second if not available
            // If you have access to the perturbed state, replace below with actual values
            // REMOVE: For now, just offset slightly for visualization
            // Use sampledPendulum[4] and [6] for perturbed pendulum angles
            const theta1b = sampledPendulum[4]
            const theta2b = sampledPendulum[6]

            // Calculate positions for second pendulum
            const x1b = origin.x + l1 * Math.sin(theta1b)
            const y1b = origin.y + l1 * Math.cos(theta1b)
            const x2b = x1b + l2 * Math.sin(theta2b)
            const y2b = y1b + l2 * Math.cos(theta2b)

            // Calculate distance between bobs
            const dist = Math.sqrt((x2 - x2b) ** 2 + (y2 - y2b) ** 2)
            // Normalize distance for color map (assuming maxLength is max possible)
            const normDist = Math.min(dist / (2 * maxLength), 1)
            const colorIdx = Math.floor(normDist * (colorMap.length - 1))
            const lineColor = rgb(colorMap[colorIdx])

            // Add line trace between bobs
            trace.push({ x1: x2, y1: y2, x2: x2b, y2: y2b, color: lineColor, alpha: 1.0 })
            if (trace.length > maxTraceLength) trace.shift()

            // Draw all trace lines (faded)
            for (let i = 0; i < trace.length; ++i) {
                const t = trace[i]
                context.save()
                context.globalAlpha = t.alpha * ((i + 1) / trace.length)
                context.strokeStyle = t.color
                context.lineWidth = 2
                context.beginPath()
                context.moveTo(t.x1, t.y1)
                context.lineTo(t.x2, t.y2)
                context.stroke()
                context.restore()
            }
            // Fade trace alphas
            for (let t of trace) t.alpha *= fadeStep

            // Draw both pendulums (arms and bobs)
            // First pendulum
            context.globalAlpha = 1.0
            context.strokeStyle = baseContentColor
            context.lineWidth = 3
            context.beginPath()
            context.moveTo(origin.x, origin.y)
            context.lineTo(x1, y1)
            context.stroke()
            context.strokeStyle = baseContentColor
            context.beginPath()
            context.moveTo(x1, y1)
            context.lineTo(x2, y2)
            context.stroke()
            context.fillStyle = baseContentColor
            context.beginPath()
            context.arc(x1, y1, m1, 0, 2 * Math.PI)
            context.fill()
            context.fillStyle = baseContentColor
            context.beginPath()
            context.arc(x2, y2, m2, 0, 2 * Math.PI)
            context.fill()

            // Second pendulum
            context.globalAlpha = 1.0
            context.strokeStyle = baseContentColor
            context.lineWidth = 3
            context.beginPath()
            context.moveTo(origin.x, origin.y)
            context.lineTo(x1b, y1b)
            context.stroke()
            context.strokeStyle = baseContentColor
            context.beginPath()
            context.moveTo(x1b, y1b)
            context.lineTo(x2b, y2b)
            context.stroke()
            context.fillStyle = baseContentColor
            context.beginPath()
            context.arc(x1b, y1b, m1, 0, 2 * Math.PI)
            context.fill()
            context.fillStyle = baseContentColor
            context.beginPath()
            context.arc(x2b, y2b, m2, 0, 2 * Math.PI)
            context.fill()
        } else {
            // Angle and energy loss modes: unchanged
            // Change arm color for Theta1 and Theta2 modes
            if (selectedVisualizationMode.id === 0) {
                color1 = rgb(colorMap[angleToColorIndex(theta1)])
            } else if (selectedVisualizationMode.id === 1) {
                color2 = rgb(colorMap[angleToColorIndex(theta2)])
            }

            // Add current position to trace
            trace.push({ x1: x2, y1: y2, x2: x2, y2: y2, color: baseContentColor, alpha: 1.0 })
            if (trace.length > maxTraceLength) trace.shift()

            // Draw trace as bob path
            for (let i = 1; i < trace.length; ++i) {
                const prev = trace[i - 1]
                const curr = trace[i]
                context.save()
                context.globalAlpha = curr.alpha * (i / trace.length)
                context.strokeStyle = baseContentColor
                context.lineWidth = 2
                context.beginPath()
                context.moveTo(prev.x1, prev.y1)
                context.lineTo(curr.x1, curr.y1)
                context.stroke()
                context.restore()
            }
            // Fade trace alphas
            for (let t of trace) t.alpha *= fadeStep

            // Draw first arm
            context.globalAlpha = 1.0
            context.strokeStyle = color1
            context.lineWidth = 3
            context.beginPath()
            context.moveTo(origin.x, origin.y)
            context.lineTo(x1, y1)
            context.stroke()

            // Draw second arm
            context.strokeStyle = color2
            context.lineWidth = 3
            context.beginPath()
            context.moveTo(x1, y1)
            context.lineTo(x2, y2)
            context.stroke()

            // Draw first bob
            context.fillStyle = color1
            context.beginPath()
            context.arc(x1, y1, m1, 0, 2 * Math.PI)
            context.fill()

            // Draw second bob
            context.fillStyle = color2
            context.beginPath()
            context.arc(x2, y2, m2, 0, 2 * Math.PI)
            context.fill()
        }
    }

    function drawColormapPreview() {
        const context = gradientCanvas.getContext('2d')
        if (!context) return

        // TODO
        const colorMap = loadColorMap(colormapCsv)
        const w = gradientCanvas.width
        const h = gradientCanvas.height
        for (let x = 0; x < w; x++) {
            const t = x / (w - 1)
            const rgb = colorMap[Math.floor(t * (colorMap.length - 1))]
            context.fillStyle = `rgb(${Math.round(rgb[0] * 255)},${Math.round(rgb[1] * 255)},${Math.round(rgb[2] * 255)})`
            context.fillRect(x, 0, 1, h)
        }
    }

    function onSelectClickAction(clickAction: ClickAction) {
        selectedClickAction = clickAction
    }

    function onSelectVisualizationMode() {
        if (
            selectedVisualizationMode.id === 0 ||
            selectedVisualizationMode.id === 1
        ) {
            selectedColormap = defaultCyclicColorMap
        } else if (selectedVisualizationMode.id === 2) {
            selectedColormap = defaultLinearColorMap
        } else if (selectedVisualizationMode.id === 3) {
            selectedColormap = defaultDivergingColorMap
        }
        resetShaders()
    }

    function onSelectColormap() {
        resetShaders()
    }

    // Draw crosshair overlay
    function drawCrosshair() {
        const context = crosshairCanvas.getContext('2d')
        if (!context) return

        context.clearRect(0, 0, crosshairCanvas.width, crosshairCanvas.height)
        if (!showCrosshair) return

        // Convert grid coordinates to canvas pixels
        const [x, y] = sampledPendulumXY
        const px = x * (crosshairCanvas.width / gridSize)
        const py = (gridSize - y) * (crosshairCanvas.height / gridSize)
        context.save()
        context.strokeStyle = baseContentColor
        context.lineWidth = 1

        // Draw horizontal line
        context.beginPath()
        context.moveTo(0, py)
        context.lineTo(crosshairCanvas.width, py)
        context.stroke()

        // Draw vertical line
        context.beginPath()
        context.moveTo(px, 0)
        context.lineTo(px, crosshairCanvas.height)
        context.stroke()

        // Draw small center dot
        context.beginPath()
        context.arc(px, py, 4, 0, 2 * Math.PI)
        context.fillStyle = 'rgba(255,255,255,0.7)'
        context.fill()
        context.restore()
    }
</script>

{#snippet stat(title: string, stat: string)}
    <div class="stat">
        <div class="stat-title">{title}</div>
        <div class="stat-value font-mono">{stat}</div>
    </div>
{/snippet}

<main class="md:m-2 md:overflow-x-scroll">
    <div class="flex flex-col md:flex-row md:gap-4 md:p-2">
        <div class="flex-none" style="position:relative;">
            <!-- Fractal canvas -->
            <canvas
                class="skeleton w-full rounded-lg border border-gray-700"
                width={gridSize}
                height={gridSize}
                bind:this={fractalCanvas}
                onclick={fractalCanvasClick}
                style="position:relative;z-index:1;"
            ></canvas>
            <!-- Crosshair canvas -->
            <canvas
                width={gridSize}
                height={gridSize}
                bind:this={crosshairCanvas}
                class="pointer-events-none absolute top-0 left-0"
                style="z-index:2;"
            ></canvas>
        </div>

        <div class="m-2 flex-1">
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

            <!-- Crosshair toggle -->
            <label class="label">
                <input type="checkbox" bind:checked={showCrosshair} />
                Show crosshair
            </label>

            <fieldset class="fieldset">
                <!-- Click action buttons -->
                <legend class="fieldset-legend">Click action</legend>
                <div class="join join-vertical">
                    {#each clickActions as clickAction}
                        <input
                            class="btn join-item"
                            type="radio"
                            name="clickAction"
                            aria-label={clickAction.name}
                            onclick={() => onSelectClickAction(clickAction)}
                            checked={selectedClickAction === clickAction}
                        />
                    {/each}
                </div>

                <!-- Reset controls -->
                <legend class="fieldset-legend">Reset controls</legend>
                <div class="join join-vertical">
                    {#each resetControls as resetControl}
                        <input
                            class="btn join-item btn-primary"
                            type="radio"
                            name="resetControls"
                            aria-label={resetControl.name}
                            onclick={resetControl.onclick}
                        />
                    {/each}
                </div>

                <!-- Visualization mode selector -->
                <legend class="fieldset-legend">Visualization Mode</legend>
                <select
                    class="select w-full"
                    bind:value={selectedVisualizationMode}
                    onchange={() => onSelectVisualizationMode()}
                >
                    {#each visualizationModes as visualizationMode}
                        <option value={visualizationMode}
                            >{visualizationMode.name}</option
                        >
                    {/each}
                </select>

                <!-- Color map selector -->
                <legend class="fieldset-legend">Color map</legend>
                <select
                    class="select w-full"
                    bind:value={selectedColormap}
                    onchange={onSelectColormap}
                >
                    {#each colorMaps as cmap}
                        <option value={cmap}>{cmap.displayName}</option>
                    {/each}
                </select>
                <canvas
                    class="h-4 w-full rounded-lg border border-gray-700"
                    bind:this={gradientCanvas}
                ></canvas>
                <div
                    class="label flex justify-between font-mono text-xs tabular-nums"
                >
                    {#if selectedVisualizationMode.id === 0 || selectedVisualizationMode.id === 1}
                        <span>0</span>
                        <span>+&pi;/2</span>
                        <span>&#x00B1;pi</span>
                        <span>+&pi;/2</span>
                        <span>0</span>
                    {:else if selectedVisualizationMode.id === 2}
                        <span>0</span>
                        <span>0.5</span>
                        <span>1</span>
                    {/if}
                </div>
                <a
                    class="label link"
                    href="https://colorcet.com/gallery.html"
                    target="_blank"
                >
                    ColorCET maps
                </a>
            </fieldset>
        </div>

        <div class="m-2 flex-1 md:m-0">
            <!-- Performance stats -->
            <div class="stats flex justify-items-center">
                {@render stat('Ticks / sec', Math.abs(measuredTps).toFixed(0))}
                {@render stat('Frames / sec', measuredFps.toFixed(0))}
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
                max={maxTicksPerSecond}
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
                <p class="label w-full">RK4 integration time step</p>
            </div>
        </div>
    </div>
</main>
