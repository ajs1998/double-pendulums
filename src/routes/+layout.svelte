<script lang="ts">
    import '../app.css'
    import { base } from '$app/paths';
    import { onMount } from 'svelte'
    import tgpu from 'typegpu'

    let { children } = $props()

    let showWebgpuAlert = $state(false)

    onMount(async () => {
        if (!navigator.gpu) {
            showWebgpuAlert = true
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
            <a class="btn btn-ghost text-xl" href="{base}/">WebGPU Experiments</a>
        </div>
        <div class="navbar-center hidden lg:flex">
            <a class="btn btn-ghost text-lg" href="{base}/pendulumFractal"
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

{#if showWebgpuAlert}
    <div class="toast toast-top toast-center my-10">
        <div class="alert alert-error">
            <div class="gap-12">
                <h3 class="font-bold">Error</h3>
                <span class="text-sm">
                    This browser does not support WebGPU
                </span>
            </div>
            <a
                class="btn btn-sm btn-soft"
                href="https://caniuse.com/webgpu"
                target="_blank"
            >
                See supported browsers
            </a>
            <button class="btn btn-sm btn-soft">Dismiss</button>
        </div>
    </div>
{/if}
