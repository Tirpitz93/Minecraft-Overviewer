#version 330

uniform mat4 Mvp;
uniform vec3 pos;
uniform vec3 scale;
uniform vec2 model_rot;
//uniform vec3 dir_light;
in vec3 in_vert;
in vec3 in_normal;
in vec2 in_texcoord_0;
out vec2 texCoordV;
out vec4 color;
//out float lum;

void main() {
    texCoordV = in_texcoord_0 + vec2(0, 0.0001) * in_normal.xy;
    color = vec4((in_vert * scale) + pos, 1.0);
    gl_Position = Mvp * vec4((in_vert * scale) + pos, 1.0);
}
