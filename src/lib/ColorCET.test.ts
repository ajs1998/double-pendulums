import { expect, test } from 'vitest'
import { getId, getDisplayName, findColorCETMap } from '$lib/ColorCET'

test('get id', () => {
    expect(getId('C3')).toEqual({ type: 'cyclic', id: 3 })
    expect(getId('C3s')).toEqual({ type: 'cyclic', id: 3, variant: 's' })
    expect(getId('CBC1')).toEqual({ type: 'cyclic', id: 1, variant: 'CB' })
    expect(getId('CBD1')).toEqual({ type: 'diverging', id: 1, variant: 'CB' })
    expect(getId('CBL1')).toEqual({ type: 'linear', id: 1, variant: 'CB' })
    expect(getId('D03')).toEqual({ type: 'diverging', id: 3 })
    expect(getId('D01A')).toEqual({ type: 'diverging', id: 1, variant: 'A' })
    expect(getId('I3')).toEqual({ type: 'isoluminant', id: 3 })
    expect(getId('L03')).toEqual({ type: 'linear', id: 3 })
    expect(getId('R3')).toEqual({ type: 'rainbow', id: 3 })
})

test('get display name', () => {
    expect(getDisplayName({ type: 'cyclic', id: 3 })).toEqual('Cyclic 3')
    expect(getDisplayName({ type: 'cyclic', id: 3, variant: 's' })).toEqual(
        'Cyclic 3 (shift 25%)'
    )
    expect(getDisplayName({ type: 'cyclic', id: 3, variant: 'CB' })).toEqual(
        'Colorblind Cyclic 3'
    )
    expect(getDisplayName({ type: 'diverging', id: 1, variant: 'CB' })).toEqual(
        'Colorblind Diverging 1'
    )
    expect(getDisplayName({ type: 'linear', id: 1, variant: 'CB' })).toEqual(
        'Colorblind Linear 1'
    )
    expect(getDisplayName({ type: 'diverging', id: 1, variant: 'A' })).toEqual(
        'Diverging 1 (high contrast)'
    )
    expect(getDisplayName({ type: 'isoluminant', id: 3 })).toEqual(
        'Isoluminant 3'
    )
    expect(getDisplayName({ type: 'linear', id: 3 })).toEqual('Linear 3')
    expect(getDisplayName({ type: 'rainbow', id: 3 })).toEqual('Rainbow 3')
})

test('get colormap', () => {
    expect(findColorCETMap({ type: 'cyclic', id: 1 })).toBeDefined()
})
