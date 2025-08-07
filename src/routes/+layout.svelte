<script lang="ts">
    import '../app.css'
    import { onMount } from 'svelte'
    import tgpu from 'typegpu'

    let { children } = $props()

    onMount(async () => {
        if (!navigator.gpu) {
            throw new Error('WebGPU is not supported on this browser.')
        } else {
            const root = await tgpu.init()
            let device: GPUDevice = root.device

            console.log('WebGPU supported')
            console.log(device.limits)
        }
    })
</script>

<div class="flex flex-col">
    <div class="navbar bg-base-100 flex flex-row shadow-sm">
        <div class="navbar-start">
            <a class="btn btn-ghost text-xl" href="/">WebGPU Experiments</a>
        </div>
        <div class="navbar-center hidden lg:flex">
            <a class="btn btn-ghost text-lg" href="/pendulumFractal"
                >Double Pendulum</a
            >
        </div>
        <div class="navbar-end">
            <a
                class="btn text-sm"
                href="https://github.com/ajs1998/webgpu-experiments"
                target="_blank">GitHub</a
            >
        </div>
    </div>

    {@render children()}
</div>
