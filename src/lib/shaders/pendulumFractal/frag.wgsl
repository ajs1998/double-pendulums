// Fragment shaders compute the color of a pixel
// This has already been done in the compute shader
@fragment
fn main(@location(0) color: vec4f) -> @location(0) vec4f {
  return color;
}
