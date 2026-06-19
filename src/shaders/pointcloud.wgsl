// pointcloud.wgsl — 3D pointcloud vertex + fragment shaders
//
// Renders colored points as instanced quads with depth testing.
// Each point is a small billboard quad facing the camera.

struct Uniforms {
  viewProjection: mat4x4<f32>,
  cameraPos: vec3<f32>,
  pointSize: f32,
  numPoints: u32,
};

struct VertexInput {
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> points: array<f32>;  // [N, 3]
@group(0) @binding(2) var<storage, read> colors: array<f32>;  // [N, 3]

// Quad vertices (two triangles)
const quadPositions = array<vec2<f32>, 6>(
  vec2<f32>(-0.5, -0.5),
  vec2<f32>( 0.5, -0.5),
  vec2<f32>(-0.5,  0.5),
  vec2<f32>(-0.5,  0.5),
  vec2<f32>( 0.5, -0.5),
  vec2<f32>( 0.5,  0.5),
);

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;

  let pointIdx = input.instanceIndex;
  if (pointIdx >= uniforms.numPoints) {
    output.position = vec4<f32>(0.0, 0.0, -1.0, 1.0);
    return output;
  }

  let px = points[pointIdx * 3 + 0];
  let py = points[pointIdx * 3 + 1];
  let pz = points[pointIdx * 3 + 2];
  let worldPos = vec3<f32>(px, py, pz);

  // Billboard offset in clip space
  let quadOffset = quadPositions[input.vertexIndex];
  let clipPos = uniforms.viewProjection * vec4<f32>(worldPos, 1.0);

  // Scale point size based on distance
  let dist = length(worldPos - uniforms.cameraPos);
  let size = uniforms.pointSize / max(dist, 0.1);

  output.position = clipPos + vec4<f32>(quadOffset * size * 0.01, 0.0, 0.0);
  output.color = vec3<f32>(
    colors[pointIdx * 3 + 0],
    colors[pointIdx * 3 + 1],
    colors[pointIdx * 3 + 2],
  );
  output.uv = quadOffset + 0.5;

  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  // Circular point with soft edge
  let dist = length(input.uv - vec2<f32>(0.5));
  if (dist > 0.5) {
    discard;
  }
  let alpha = smoothstep(0.5, 0.3, dist);
  return vec4<f32>(input.color, alpha);
}
