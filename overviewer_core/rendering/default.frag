#version 330

uniform sampler2DArray Textures;
in vec4 color;
in vec3 texCoord;
in float lum;
out vec4 fragColor;

void main() {
    fragColor = vec4(lum, lum, lum, 1.0) * vec4(texture(Textures, texCoord));
}
