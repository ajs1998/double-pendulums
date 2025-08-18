import * as d from 'typegpu/data'

export type ColorCETMapType =
    | 'cyclic'
    | 'diverging'
    | 'isoluminant'
    | 'linear'
    | 'rainbow'
export type ColorCETIdentifier = {
    type: ColorCETMapType
    id: number
    variant?: 'CB' | 's' | 'A'
}
export type ColorCETMap = {
    id: ColorCETIdentifier
    displayName: string
    colors: d.v3f[]
}

export function getId(idString: string): ColorCETIdentifier {
    const matches = idString.match(/^(CB)?(C|D|I|L|R)(\d+)(s|A)?$/)
    if (!matches) throw new Error(`Invalid ColorCET id string: ${idString}`)
    const type =
        matches[2] === 'C'
            ? 'cyclic'
            : matches[2] === 'D'
              ? 'diverging'
              : matches[2] === 'I'
                ? 'isoluminant'
                : matches[2] === 'L'
                  ? 'linear'
                  : 'rainbow'
    const id = Number(matches[3])
    const variant =
        matches[1] === 'CB'
            ? 'CB'
            : matches[4] === 's'
              ? 's'
              : matches[4] === 'A'
                ? 'A'
                : undefined
    return {
        type,
        id,
        variant,
    }
}

export function getDisplayName(id: ColorCETIdentifier): string {
    let displayName = `${id.type.charAt(0).toUpperCase() + id.type.slice(1)} ${id.id}`
    if (id.variant === 'CB') {
        displayName = `Colorblind ${displayName}`
    } else if (id.variant === 's') {
        displayName = `${displayName} (shift 25%)`
    } else if (id.variant === 'A') {
        displayName = `${displayName} (high contrast)`
    }
    return displayName
}

const modules = Object.entries(
    import.meta.glob('$lib/colorcet-maps/*.csv', {
        query: '?raw',
        import: 'default',
        eager: true,
    })
)

// Load all colormaps as a map from "CET-C3s" etc to csv string
export const colorCETMaps: ColorCETMap[] =
    modules.map(([path, csv]) => {
        const idString = path.split('-').pop()!.replace('.csv', '')
        const colors = (csv as string)
            .split('\n')
            .filter((line) => line.trim() !== '')
            .map((line) => {
                const [r, g, b] = line.split(',').map(Number)
                return d.vec3f(r / 255, g / 255, b / 255)
            })
        const id = getId(idString)
        const displayName = getDisplayName(id)
        return { id, displayName, colors }
    }).sort((a, b) => a.displayName.localeCompare(b.displayName));

export function findColorCETMap(id: ColorCETIdentifier): ColorCETMap | undefined {
    return colorCETMaps.find(entry =>
        entry.id.type === id.type &&
        entry.id.id === id.id &&
        entry.id.variant === id.variant
    );
}
