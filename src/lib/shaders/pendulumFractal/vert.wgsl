struct Pixel {
    energy: vec2f, // (kinetic_energy, potential_energy)
    initial_energy: f32, // initial kinetic_energy, initial potential_energy
    distance: f32, // distance between the 2 pendulums
}

@group(0) @binding(0) var<uniform> grid_size: u32;
@group(0) @binding(1) var<storage, read> states: array<vec4f>;
@group(0) @binding(2) var<storage, read> pixels: array<Pixel>;
@group(0) @binding(3) var<storage, read> color_map: array<vec3f>;
@group(0) @binding(4) var<uniform> visualization_mode: u32;

struct Out {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
};

const PI: f32 = 3.141592653589793;

@vertex
fn main(
    @builtin(instance_index) pixel_index: u32, 
    @location(0) corner: vec2u
) -> Out {
    // Convert corner position and pixel index to normalized device coordinates [-1, 1]
    let x = (f32(corner.x + pixel_index % grid_size)) / f32(grid_size) * 2.0 - 1.0;
    let y = (f32(corner.y + pixel_index / grid_size)) / f32(grid_size) * 2.0 - 1.0;
    let state = states[pixel_index];
    let theta1 = state[0];
    let omega1 = state[1];
    let theta2 = state[2];
    let omega2 = state[3];

    let pixel = pixels[pixel_index];

    var color_index: u32;
    if (visualization_mode == 0u) {
        // Theta1 visualization
        color_index = u32(fract(theta1 / (2 * PI)) * 255);
    } else {
        // Energy deviation
        let difference = abs(pixel.energy[0] + pixel.energy[1] - pixel.initial_energy);
        color_index = (u32(log(1 + difference) * 255)) % 256;
    }

    // Map distance to a color
    // let color_index = u32(pixel.distance * 255);

    let color = color_map[color_index];
    return Out(vec4f(x, y, 0., 1.), vec4f(color.rgb, 1.));
}
