export type ColorCETMap = {
    displayName: string
    csv: string
}

// Load all colormaps as a map from "CET-C3s" etc to csv string
const colormapModules: Record<string, string> = Object.fromEntries(
    Object.entries(
        import.meta.glob('$lib/colorcet-maps/*.csv', {
            query: '?raw',
            import: 'default',
            eager: true,
        })
    ).map(([path, csv]) => [
        path.split('/').pop()!.replace('.csv', ''),
        csv as string,
    ])
)

// Shift a colormap by 50%
function shiftColormap50(csv: string): string {
    const lines = csv.split('\n')
    const half = Math.floor(lines.length / 2)
    return [...lines.slice(half), ...lines.slice(0, half)].join('\n')
}

// Cyclic colormaps (CET-C*)
export const cyclicColorMaps: ColorCETMap[] = Object.entries(colormapModules)
    .filter(([key]) => /^CET-C\d(s)?$/.test(key))
    .flatMap(([key, csv]) => {
        const displayName =
            'Cyclic ' +
            key.replace(/^CET-C(\d)(s)?$/, (_, num, s) =>
                s ? `${num} shift 25%` : num
            )
        return [{ displayName, csv }]
    })
    .concat([
        {
            displayName: 'Cyclic 3 shift 50%',
            csv: shiftColormap50(colormapModules['CET-C3']),
        },
    ])
    .sort((a, b) => a.displayName.localeCompare(b.displayName))

// Linear colormaps (CET-L*)
export const linearColorMaps: ColorCETMap[] = Object.entries(colormapModules)
    .filter(([key]) => /^CET-L\d{2}(s)?$/.test(key))
    .map(([key, csv]) => ({
        displayName:
            'Linear ' +
            key.replace(/^CET-L(\d{2})(s)?$/, (_, num, s) =>
                s ? `${num} shift 25%` : num
            ),
        csv,
    }))

// Example usage:
// cyclicColorMaps.find(m => m.displayName === "Cyclic 3 shift 50%")
