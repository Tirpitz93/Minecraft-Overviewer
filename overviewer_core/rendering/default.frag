#version 330

uniform sampler2DArray Textures;
in vec4 color;
in vec3 texCoord;
out vec4 fragColor;

void main() {
    fragColor = vec4(texture(Textures, texCoord));
}
