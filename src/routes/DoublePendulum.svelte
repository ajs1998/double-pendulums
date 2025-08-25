<script lang="ts">
    import { onMount, onDestroy } from 'svelte'
    import tgpu, {
        type StorageFlag,
        type TgpuBuffer,
        type TgpuRoot,
        type UniformFlag,
    } from 'typegpu'
    import * as d from 'typegpu/data'
    import computeShaderCode from '$lib/shaders/pendulumFractal/compute.wgsl?raw'
    import vertexShaderCode from '$lib/shaders/pendulumFractal/vert.wgsl?raw'
    import fragmentShaderCode from '$lib/shaders/pendulumFractal/frag.wgsl?raw'
    import { RollingAverage } from '$lib/RollingAverage'
    import {
        colorCETMaps,
        findColorCETMap,
        type ColorCETMap,
    } from '$lib/ColorCET'

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
    let zoomCenter = $state({ theta1: 0, theta2: 0 })
    let integrationTimestep = 0.005
    // Temporary value for timestep slider until the slider is released
    let integrationTimestepTemp = $state(integrationTimestep)
    let showCrosshair = $state(true)
    let visualizationModeBuffer: TgpuBuffer<typeof d.u32> & UniformFlag
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
    const sampledCanvasSize = 300
    // Secondary pendulum perturbation amount in pixel units
    // Lower is more stable, but slower to converge
    const perturbationAmount = 0.5
    const maxTraceLength = 1000
    const traceWidth = 2
    const crosshairWidth = 2
    const cyclicColorMaps = colorCETMaps.filter(
        (map) => map.id.type === 'cyclic' && map.id.variant !== 's'
    )
    const linearColorMaps = colorCETMaps.filter(
        (map) => map.id.type === 'linear'
    )
    const defaultCyclicColorMap = findColorCETMap({ type: 'cyclic', id: 3 })!
    const defaultDivergingColorMap = defaultCyclicColorMap
    const defaultLinearColorMap = findColorCETMap({ type: 'linear', id: 16 })!

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

    let selectableColorMaps = $state(cyclicColorMaps)
    let selectedColorMap = $state(defaultCyclicColorMap)
    let selectedClickAction = $state(clickActions[0])
    let selectedVisualizationMode = $state(visualizationModes[1])
    let sampledPendulumXY = $state(centerXY)
    let sampledPendulumLocation = $state([0, 0]) // Theta coordinates
    let sampledPendulum = $state([0, 0, 0, 0])
    let statesBuffer: TgpuBuffer<d.WgslArray<d.Vec4f>> & StorageFlag

    function getClickXYCoordinates(e: MouseEvent) {
        const rect = fractalCanvas.getBoundingClientRect()
        // I don't know why adding 2 makes it look right
        const x = 2 + Math.floor((e.x - rect.left) / (rect.width / gridSize))
        const y =
            gridSize -
            1 -
            Math.floor((e.y - rect.top) / (rect.height / gridSize))
        return { x, y }
    }

    function toThetaCoordinates(x: number, y: number) {
        const theta1 =
            (Math.PI / zoomAmount) * ((x / (gridSize - 1)) * 2 - 1) +
            zoomCenter.theta1
        const theta2 =
            (Math.PI / zoomAmount) * ((y / (gridSize - 1)) * 2 - 1) +
            zoomCenter.theta2
        return { theta1, theta2 }
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
            statesBuffer.buffer,
            i * d.sizeOf(d.vec4f),
            readBufferA.buffer,
            0,
            d.sizeOf(d.vec4f)
        )
        // Copy the state for the selected pixel (perturbed pendulum)
        commandEncoder.copyBufferToBuffer(
            statesBuffer.buffer,
            (i + pixelCount) * d.sizeOf(d.vec4f),
            readBufferB.buffer,
            0,
            d.sizeOf(d.vec4f)
        )
        device.queue.submit([commandEncoder.finish()])
        const mainPendulum = await readBufferA.read()
        const perturbedPendulum = await readBufferB.read()
        sampledPendulum = [...mainPendulum, ...perturbedPendulum]
        readBufferA.destroy()
        readBufferB.destroy()
    }

    function zoom(x: number, y: number, factor: number) {
        const { theta1, theta2 } = toThetaCoordinates(x, y)
        zoomAmount *= factor
        zoomCenter = { theta1, theta2 }

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

    function resetZoom() {
        zoomCenter = { theta1: 0, theta2: 0 }
        zoomAmount = 1.0
        resetPendulums()
    }

    function fractalCanvasClick(e: MouseEvent) {
        if (e.target !== fractalCanvas) {
            return
        }
        const { x, y } = getClickXYCoordinates(e)
        const { theta1, theta2 } = toThetaCoordinates(x, y)
        sampledPendulumXY = [x, y]
        sampledPendulumLocation = [theta1 / Math.PI, theta2 / Math.PI]
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
        // Setup WebGPU and canvas
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
            integrationTimestep = integrationTimestepTemp
            trace = []

            drawColormapPreview()

            // Setup compute shader
            const computeModule = device.createShaderModule({
                code: computeShaderCode,
            })

            const uniformBufferData = d.struct({
                // RK4 integration time step
                dt: d.f32,
                // Gravitational acceleration
                gravity: d.f32,
                // Length of first pendulum arm
                l1: d.f32,
                // Length of second pendulum arm
                l2: d.f32,
                // Mass of first pendulum bob
                m1: d.f32,
                // Mass of second pendulum bob
                m2: d.f32,
                gridSize: d.u32, // Grid size
            })

            const uniformBuffer = root
                .createBuffer(uniformBufferData, {
                    dt: integrationTimestep,
                    gravity: gravity,
                    l1: length1,
                    l2: length2,
                    m1: mass1,
                    m2: mass2,
                    gridSize,
                })
                .$usage('uniform')
            cleanupFns.push(() => uniformBuffer.destroy())

            const pixelsBufferData = d.arrayOf(
                d.struct({
                    // (initial_energy, kinetic_energy, potential_energy)
                    energy: d.vec3f,
                    // Distance between the 2 pendulums
                    distance: d.f32,
                }),
                pixelCount
            )
            const pixelsBuffer = root
                .createBuffer(pixelsBufferData)
                .$usage('storage', 'vertex')
            cleanupFns.push(() => pixelsBuffer.destroy())

            // Two states (theta1, omega1, theta2, omega2) per pixel
            // One is the main pendulum, and the other is slightly perturbed
            const statesBufferData = d.arrayOf(d.vec4f, pixelCount * 2)
            statesBuffer = root.createBuffer(statesBufferData).$usage('storage')
            cleanupFns.push(() => statesBuffer.destroy())

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
                            zoomCenter.theta1
                        const theta2 =
                            (Math.PI / zoomAmount) *
                                ((y / (gridSize - 1)) * 2 - 1) +
                            zoomCenter.theta2
                        initialStates[i] = d.vec4f(theta1, 0, theta2, 0)

                        // Perturb the second pendulum slightly in phase space
                        const r =
                            (perturbationAmount * (Math.PI / zoomAmount)) /
                            (gridSize - 1)
                        const angle = Math.random() * 2 * Math.PI
                        const deltaTheta1 = r * Math.cos(angle)
                        const deltaTheta2 = r * Math.sin(angle)
                        initialStates[i + pixelCount] = d.vec4f(
                            theta1 + deltaTheta1,
                            0,
                            theta2 + deltaTheta2,
                            0
                        )
                        const potential_energy =
                            -(mass1 + mass2) *
                                length1 *
                                gravity *
                                Math.cos(theta1) -
                            mass2 * length2 * gravity * Math.cos(theta2)
                        initialEnergies[i] = d.f32(potential_energy)
                    }
                }
                statesBuffer.write(initialStates)
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

            // Buffer for selected visualization mode
            visualizationModeBuffer = root
                .createBuffer(d.u32, selectedVisualizationMode.id)
                .$usage('uniform')
            cleanupFns.push(() => visualizationModeBuffer.destroy())

            // Buffer for selected color map
            const colorMapBufferData = d.arrayOf(
                d.vec3f,
                selectedColorMap.colors.length
            )
            const colorMapBuffer = root
                .createBuffer(colorMapBufferData, selectedColorMap.colors)
                .$usage('storage', 'vertex')
            cleanupFns.push(() => colorMapBuffer.destroy())

            const gridSizeBuffer = root
                .createBuffer(d.u32, gridSize)
                .$usage('uniform')
            cleanupFns.push(() => gridSizeBuffer.destroy())

            const computeBindGroupLayout = tgpu.bindGroupLayout({
                uniforms: { uniform: uniformBufferData },
                states: { storage: statesBufferData, access: 'mutable' },
                pixels: { storage: pixelsBufferData, access: 'mutable' },
            })

            const computeBindGroup = root.createBindGroup(
                computeBindGroupLayout,
                {
                    uniforms: uniformBuffer,
                    states: statesBuffer,
                    pixels: pixelsBuffer,
                }
            )

            const computePipeline = device.createComputePipeline({
                layout: device.createPipelineLayout({
                    bindGroupLayouts: [root.unwrap(computeBindGroupLayout)],
                }),
                compute: {
                    module: computeModule,
                    entryPoint: 'main',
                    constants: {
                        workgroupSize: workgroupSize,
                    },
                },
            })

            const vertexModule = device.createShaderModule({
                code: vertexShaderCode,
            })
            const fragmentModule = device.createShaderModule({
                code: fragmentShaderCode,
            })

            const renderBindGroupLayout = tgpu.bindGroupLayout({
                gridSize: { uniform: d.u32 },
                states: { storage: statesBufferData },
                pixels: { storage: pixelsBufferData },
                colorMap: { storage: colorMapBufferData },
                visualizationMode: { uniform: d.u32 },
            })

            const renderBindGroup = root.createBindGroup(
                renderBindGroupLayout,
                {
                    gridSize: gridSizeBuffer,
                    states: statesBuffer,
                    pixels: pixelsBuffer,
                    colorMap: colorMapBuffer,
                    visualizationMode: visualizationModeBuffer,
                }
            )

            const renderPipeline = device.createRenderPipeline({
                layout: device.createPipelineLayout({
                    bindGroupLayouts: [root.unwrap(renderBindGroupLayout)],
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

            function runComputePass() {
                const encoder = device.createCommandEncoder()
                const computePass = encoder.beginComputePass()
                computePass.setPipeline(computePipeline)
                computePass.setBindGroup(0, root.unwrap(computeBindGroup))
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
                renderPass.setBindGroup(0, root.unwrap(renderBindGroup))
                // Draw the pixels as squares defined by the 4 corner vertices
                renderPass.draw(4, pixelCount)
                renderPass.end()
                device.queue.submit([encoder.finish()])
            }

            let rollingAverageTps = new RollingAverage(5 * 60)
            let rollingAverageFps = new RollingAverage(5 * 60)

            function computeLoop() {
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

                // Run again as fast as possible
                simulationTimerId = window.setTimeout(computeLoop, 0)
            }

            let lastRenderTime = 0
            async function renderLoop() {
                const now = performance.now()
                const elapsed = now - lastRenderTime
                lastRenderTime = now

                // Draw fractal
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
                drawSampledPendulum(sampledPendulum[0], sampledPendulum[2])

                animationFrameId = requestAnimationFrame(renderLoop)
            }

            // Start the compute and render loops
            computeLoop()
            animationFrameId = requestAnimationFrame(renderLoop)
        }

        // Compile and run the simulation
        resetShaders()
    })

    // Destroy WebGPU resources before unmounting
    onDestroy(() => {
        root?.destroy()
    })

    // Trace state for the sampled pendulum visualization
    // For sensitivity mode, trace is an array of lines between bobs with color and alpha
    // For other modes, trace is an array of bob positions with alpha
    interface TraceSegment {
        x1: number
        y1: number
        x2: number
        y2: number
        color: string
    }

    let trace: TraceSegment[] = []
    let baseContentColor: string

    // TODO Is this inefficient?
    function getCssValue(variable: string) {
        return getComputedStyle(document.documentElement)
            .getPropertyValue(variable)
            .trim()
    }

    function drawArmAndBob(
        context: CanvasRenderingContext2D,
        x1: number,
        y1: number,
        x2: number,
        y2: number,
        radius: number,
        color: string,
        width: number = 3
    ) {
        context.strokeStyle = color
        context.lineWidth = width
        context.beginPath()
        context.moveTo(x1, y1)
        context.lineTo(x2, y2)
        context.stroke()
        context.fillStyle = color
        context.beginPath()
        context.arc(x2, y2, radius, 0, 2 * Math.PI)
        context.fill()
    }

    function drawTrace(
        context: CanvasRenderingContext2D,
        segments: TraceSegment[]
    ) {
        for (let i = 0; i < segments.length; ++i) {
            const segment = segments[i]
            context.save()
            // Fade alpha based on segment position
            context.globalAlpha = i / (segments.length - 1)
            context.strokeStyle = segment.color
            context.lineWidth = traceWidth
            context.beginPath()
            context.moveTo(segment.x1, segment.y1)
            context.lineTo(segment.x2, segment.y2)
            context.stroke()
            context.restore()
        }
    }

    function arrayToRGB(values: number[]): string {
        return `rgb(${Math.round(values[0] * 255)},${Math.round(values[1] * 255)},${Math.round(values[2] * 255)})`
    }

    function angleToColor(theta: number, colorMap: ColorCETMap): string {
        const normalizedTheta = (((theta / (2 * Math.PI)) % 1) + 1) % 1
        const index = Math.floor(normalizedTheta * (colorMap.colors.length - 1))
        return arrayToRGB(colorMap.colors[index])
    }

    function drawSampledPendulum(theta1: number, theta2: number): void {
        const context = sampledCanvas.getContext('2d')
        if (!context) return

        context.clearRect(0, 0, sampledCanvas.width, sampledCanvas.height)

        const margin = 20
        const maxLength =
            Math.min(sampledCanvas.width, sampledCanvas.height) / 2 - margin
        // TODO link to the actual arm lengths and bob masses
        const l1 = maxLength / 2
        const l2 = maxLength / 2
        const m1 = 8
        const m2 = 8
        const origin = {
            x: sampledCanvas.width / 2,
            y: sampledCanvas.height / 2,
        }

        // First pendulum positions
        const x1 = origin.x + l1 * Math.sin(theta1)
        const y1 = origin.y + l1 * Math.cos(theta1)
        const x2 = x1 + l2 * Math.sin(theta2)
        const y2 = y1 + l2 * Math.cos(theta2)

        if (selectedVisualizationMode.id === 2) {
            // Sensitivity mode: draw both pendulums and trace line between bobs
            const theta1b = sampledPendulum[4]
            const theta2b = sampledPendulum[6]
            const x1b = origin.x + l1 * Math.sin(theta1b)
            const y1b = origin.y + l1 * Math.cos(theta1b)
            const x2b = x1b + l2 * Math.sin(theta2b)
            const y2b = y1b + l2 * Math.cos(theta2b)
            const dist = Math.sqrt((x2 - x2b) ** 2 + (y2 - y2b) ** 2)
            const normDist = Math.min(dist / (2 * maxLength), 1)
            const colorIndex = Math.floor(
                normDist * (selectedColorMap.colors.length - 1)
            )
            const lineColor = arrayToRGB(selectedColorMap.colors[colorIndex])

            // Add line trace between bobs
            trace.push({
                x1: x2,
                y1: y2,
                x2: x2b,
                y2: y2b,
                color: lineColor,
            })
            if (trace.length > maxTraceLength) trace.shift()
            drawTrace(context, trace)

            // Draw both pendulums
            drawArmAndBob(context, x1, y1, x2, y2, m2, baseContentColor)
            drawArmAndBob(
                context,
                origin.x,
                origin.y,
                x1,
                y1,
                m1,
                baseContentColor
            )
            drawArmAndBob(context, x1b, y1b, x2b, y2b, m2, baseContentColor)
            drawArmAndBob(
                context,
                origin.x,
                origin.y,
                x1b,
                y1b,
                m1,
                baseContentColor
            )
        } else {
            // Angle and energy loss modes
            let color1 = baseContentColor
            let color2 = baseContentColor
            if (selectedVisualizationMode.id === 0) {
                // Theta 1: Color the first arm
                color1 = angleToColor(theta1, selectedColorMap)
            } else if (selectedVisualizationMode.id === 1) {
                // Theta 2: Color the second arm
                color2 = angleToColor(theta2, selectedColorMap)
            }

            trace.push({
                x1: x2,
                y1: y2,
                x2: x2,
                y2: y2,
                color: baseContentColor,
            })
            if (trace.length > maxTraceLength) {
                trace.shift()
            }

            // Draw trace as bob path
            for (let i = 1; i < trace.length; ++i) {
                const prev = trace[i - 1]
                const curr = trace[i]
                context.save()
                context.globalAlpha = i / (trace.length - 1)
                context.strokeStyle = baseContentColor
                context.lineWidth = traceWidth
                context.beginPath()
                context.moveTo(prev.x1, prev.y1)
                context.lineTo(curr.x1, curr.y1)
                context.stroke()
                context.restore()
            }
            drawArmAndBob(context, x1, y1, x2, y2, m2, color2)
            drawArmAndBob(context, origin.x, origin.y, x1, y1, m1, color1)
        }
    }

    function drawColormapPreview() {
        const context = gradientCanvas.getContext('2d')
        if (!context) return

        const w = gradientCanvas.width
        const h = gradientCanvas.height
        for (let x = 0; x < w; x++) {
            const colorIndex = Math.floor(
                (x / (w - 1)) * (selectedColorMap.colors.length - 1)
            )
            const color = selectedColorMap.colors[colorIndex]
            context.fillStyle = arrayToRGB(color)
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
            // Theta 1 or Theta 2
            selectedColorMap = defaultCyclicColorMap
            selectableColorMaps = cyclicColorMaps
        } else if (selectedVisualizationMode.id === 2) {
            // Sensitivity
            selectedColorMap = defaultLinearColorMap
            selectableColorMaps = linearColorMaps
        } else if (selectedVisualizationMode.id === 3) {
            // Energy loss
            selectedColorMap = defaultDivergingColorMap
            selectableColorMaps = cyclicColorMaps
        }

        resetShaders()
    }

    function onSelectColormap() {
        // TODO Can the colormap be changed without resetting the simulation?
        resetShaders()
    }

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
        context.lineWidth = crosshairWidth

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
        context.fillStyle = baseContentColor
        context.fill()
        context.restore()
    }
</script>

<main class="md:m-2 md:overflow-x-scroll">
    <div class="flex flex-col md:flex-row md:gap-4 md:p-2">
        <div class="flex-none" style="position:relative;">
            <!-- Fractal canvas -->
            <canvas
                class="skeleton z-0 w-full rounded-lg border border-gray-700"
                width={gridSize}
                height={gridSize}
                bind:this={fractalCanvas}
                onclick={fractalCanvasClick}
                style="position:relative;z-index:1;"
            ></canvas>
            <!-- Crosshair canvas -->
            <canvas
                class="pointer-events-none absolute top-0 left-0 z-10 w-full"
                width={gridSize}
                height={gridSize}
                bind:this={crosshairCanvas}
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
                    bind:value={selectedColorMap}
                    onchange={onSelectColormap}
                >
                    {#each selectableColorMaps as colorMap}
                        <option value={colorMap}>{colorMap.displayName}</option>
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
                <div class="stat">
                    <div class="stat-title">Ticks / sec</div>
                    <div class="stat-value font-mono">
                        {Math.abs(measuredTps).toFixed(0)}
                    </div>
                </div>
                <div class="stat">
                    <div class="stat-title">Frames / sec</div>
                    <div class="stat-value font-mono">
                        {#if measuredFps >= 30}
                            {measuredFps.toFixed(0)}
                        {:else}
                            <span class="text-warning"
                                >{measuredFps.toFixed(0)}</span
                            >
                        {/if}
                    </div>
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
                max={maxTicksPerSecond}
                bind:value={targetTicksPerSecond}
            />
            <p class="label w-full">Simulation steps per second</p>

            <!-- Time step slider -->
            <legend class="fieldset-legend">
                Time step
                <span class="font-mono"
                    >{integrationTimestepTemp.toFixed(4)}</span
                >
            </legend>
            <input
                type="range"
                class="range w-full"
                min="0.0001"
                max="0.01"
                step="0.0001"
                onmouseup={resetShaders}
                bind:value={integrationTimestepTemp}
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
