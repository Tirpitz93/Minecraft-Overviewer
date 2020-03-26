#version 330
uniform mat4 Mvp;
in vec3 in_vert;
out vec3 color;
void main() {
    gl_Position = Mvp * vec4(in_vert, 1.0);
    color = vec3(0.0, gl_Position.z, 1.0);
}