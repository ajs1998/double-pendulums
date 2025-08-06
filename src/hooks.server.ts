import type { Handle } from '@sveltejs/kit'

export const handle: Handle = async ({ event, resolve }) => {
    // This is necessary to prevent an error with Chrome DevTools
    // https://github.com/sveltejs/kit/issues/13743
    if (
        event.url.pathname.startsWith(
            '/.well-known/appspecific/com.chrome.devtools'
        )
    ) {
        // Return empty response with 204 No Content
        return new Response(null, { status: 204 })
    }

    return await resolve(event)
}
