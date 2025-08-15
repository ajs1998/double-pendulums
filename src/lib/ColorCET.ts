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

export const cyclicColorMaps = Object.entries(colormapModules)
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

export const linearColorMaps = Object.entries(colormapModules)
    .filter(([path]) => {
        return /CET-L\d{2}s?\.csv$/.test(path)
    })
    .map(([path, csv]) => {
        const fileName = path.split('/').pop()!
        const displayName =
            'Linear ' +
            fileName.replace(/CET-L(\d{2})(s?)\.csv/, (_, num, s) =>
                s ? `${num} shift 25%` : num
            )
        return { fileName, displayName, csv } as ColorCETMap
    })
