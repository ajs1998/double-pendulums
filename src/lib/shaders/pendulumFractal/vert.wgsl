struct Pixel {
    energy: vec3f,
    distance: f32,
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

fn floormod(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

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
        // Theta1
        color_index = u32(fract(theta1 / (2 * PI)) * 255);
    } else if (visualization_mode == 1u) {
        // Theta2
        color_index = u32(fract(theta2 / (2 * PI)) * 255);
    } else if (visualization_mode == 2u) {
        // Sensitivity
        color_index = (u32(pixel.distance * 255)) % 256;
    } else if (visualization_mode == 3u) {
        // Energy loss
        let energy = pixel.energy[1] + pixel.energy[2];
        let initial_energy = pixel.energy[0];
        color_index = u32(floormod((initial_energy - energy) / initial_energy * 256.0, 256.0));
    }

    let color = color_map[color_index];
    return Out(vec4f(x, y, 0., 1.), vec4f(color.rgb, 1.));
}

