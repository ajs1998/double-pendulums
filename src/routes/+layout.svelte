<script lang="ts">
    import '../app.css'
    import { base } from '$app/paths'
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

<!-- Navbar -->
<div class="navbar bg-base-200 px-3">
    <div class="navbar-start gap-3">
        <a class="btn btn-soft" href="{base}/">Home</a>
        <a class="btn btn-soft" href="{base}/pendulumFractal">Double Pendulum</a
        >
    </div>
    <div class="navbar-end gap-3">
        <a
            class="btn btn-soft"
            href="https://github.com/ajs1998/webgpu-experiments"
            target="_blank">GitHub</a
        >
        <button class="btn btn-soft" popovertarget="popover-1" style="anchor-name:--anchor-1">
            Theme
            <svg
                width="12px"
                height="12px"
                class="inline-block h-2 w-2 fill-current opacity-60"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 2048 2048">
                <path d="M1799 349l242 241-1017 1017L7 590l242-241 775 775 775-775z"></path>
            </svg>
        </button>
        <ul class="dropdown menu rounded-box bg-base-300 shadow-sm"
            popover id="popover-1" style="position-anchor:--anchor-1">
            <li>
                <input
                    type="radio"
                    name="theme-dropdown"
                    class="theme-controller w-full btn btn-sm justify-start"
                    aria-label="Mocha"
                    value="mocha"
                    checked />
            </li>
            <li>
                <input
                    type="radio"
                    name="theme-dropdown"
                    class="theme-controller w-full btn btn-sm justify-start"
                    aria-label="Macchiato"
                    value="macchiato" />
            </li>
            <li>
                <input
                    type="radio"
                    name="theme-dropdown"
                    class="theme-controller w-full btn btn-sm justify-start"
                    aria-label="Frappe"
                    value="frappe" />
            </li>
            <li>
                <input
                    type="radio"
                    name="theme-dropdown"
                    class="theme-controller w-full btn btn-sm justify-start"
                    aria-label="Latte"
                    value="latte" />
            </li>
        </ul>
    </div>
</div>

{@render children()}

<footer class="footer footer-center bg-base-300">
    <aside>
        <p>
            Made by
            <a
                href="https://github.com/ajs1998"
                class="link"
                target="_blank"
            >
                Alex Sweeney
            </a>
        </p>

        <!-- <p>
            Inspired by 2swap's YouTube video
            <a
                href="https://www.youtube.com/watch?v=dtjb2OhEQcU"
                class="link"
                target="_blank"
            >
                Double Pendulums are Chaoticn't
            </a>
        </p> -->
    </aside>
</footer>

<!-- WebGPU alert -->
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
