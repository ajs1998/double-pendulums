struct Uniforms {
    dt: f32,
    gravity: f32,
    l1: f32,
    l2: f32,
    m1: f32,
    m2: f32,
    grid_size: u32,
};

struct Pixel {
    energy: vec3f, // initial_energy, kinetic_energy, potential_energy
    distance: f32, // distance to the initially perturbed pendulum
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> states: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> pixels: array<Pixel>;

const PI: f32 = 3.141592653589793;

// Math adapted from https://scipython.com/blog/the-double-pendulum/
fn double_pendulum_derivatives(s: vec4f) -> vec4f {
    let theta1 = s[0];
    let omega1 = s[1];
    let theta2 = s[2];
    let omega2 = s[3];

    let m1 = uniforms.m1;
    let m2 = uniforms.m2;
    let l1 = uniforms.l1;
    let l2 = uniforms.l2;
    let g = uniforms.gravity;

    let deltaTheta = theta1 - theta2;
    let sinDeltaTheta = sin(deltaTheta);
    let cosDeltaTheta = cos(deltaTheta);

    let denom = m1 + m2 * sinDeltaTheta * sinDeltaTheta;

    let domega1_dt = (
        m2 * g * sin(theta2) * cosDeltaTheta
        - m2 * sinDeltaTheta * (l1 * omega1 * omega1 * cosDeltaTheta + l2 * omega2 * omega2)
        - (m1 + m2) * g * sin(theta1)
    ) / (l1 * denom);

    let domega2_dt = (
        (m1 + m2) * (l1 * omega1 * omega1 * sinDeltaTheta - g * sin(theta2) + g * sin(theta1) * cosDeltaTheta)
        + m2 * l2 * omega2 * omega2 * sinDeltaTheta * cosDeltaTheta
    ) / (l2 * denom);

    return vec4<f32>(omega1, domega1_dt, omega2, domega2_dt);
}

fn rk4_step(s: vec4f) -> vec4f {
    let k1 = double_pendulum_derivatives(s);
    let k2 = double_pendulum_derivatives(s + k1 * uniforms.dt / 2.);
    let k3 = double_pendulum_derivatives(s + k2 * uniforms.dt / 2.);
    let k4 = double_pendulum_derivatives(s + uniforms.dt * k3);

    return s + (uniforms.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
}

fn compute_energy(s: vec4f) -> vec2f {
    let theta1 = s[0];
    let omega1 = s[1];
    let theta2 = s[2];
    let omega2 = s[3];

    let m1 = uniforms.m1;
    let m2 = uniforms.m2;
    let l1 = uniforms.l1;
    let l2 = uniforms.l2;
    let g = uniforms.gravity;

    let kinetic_energy = 0.5 * m1 * (l1 * omega1) * (l1 * omega1)
        + 0.5 * m2 * ((l1 * omega1) * (l1 * omega1) + (l2 * omega2) * (l2 * omega2)
        + 2.0 * l1 * l2 * omega1 * omega2 * cos(theta1 - theta2));
    let potential_energy = -(m1 + m2) * l1 * g * cos(theta1) - m2 * l2 * g * cos(theta2);

    return vec2f(kinetic_energy, potential_energy);
}

override workgroupSize = 8;

// global_invocation_id = workgroup_id * workgroup_size + local_invocation_id
@compute @workgroup_size(workgroupSize, workgroupSize, 1)
fn main(@builtin(global_invocation_id) pixel_id: vec3u) {
    let grid_size = uniforms.grid_size;
    let pixel_count = grid_size * grid_size;
    let pixel_index = pixel_id.x + pixel_id.y * grid_size;
    let initialTheta0 = f32(pixel_id.x) / f32(grid_size) * 2.0 * PI - PI;
    let initialTheta1 = f32(pixel_id.y) / f32(grid_size) * 2.0 * PI - PI;

    // Run 2 pendulums per pixel then calculate their difference
    let next_state1 = rk4_step(states[pixel_index]);
    let next_state2 = rk4_step(states[pixel_index + pixel_count]);

    states[pixel_index] = next_state1;
    states[pixel_index + pixel_count] = next_state2;

    let x1 = uniforms.l1 * sin(next_state1[0]) + uniforms.l2 * sin(next_state1[2]);
    let y1 = -uniforms.l1 * cos(next_state1[0]) - uniforms.l2 * cos(next_state1[2]);
    let x2 = uniforms.l1 * sin(next_state2[0]) + uniforms.l2 * sin(next_state2[2]);
    let y2 = -uniforms.l1 * cos(next_state2[0]) - uniforms.l2 * cos(next_state2[2]);

    // Calculate and normalize the distance between the two pendulum bobs
    let maxDistance = 2 * (uniforms.l1 + uniforms.l2);
    let distance = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) / maxDistance;

    let initial_energy = pixels[pixel_index].energy[0];
    let energy = compute_energy(next_state1);
    pixels[pixel_index] = Pixel(vec3f(initial_energy, energy), distance);
}
