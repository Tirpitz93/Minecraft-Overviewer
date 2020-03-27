#version 330

uniform sampler2D Texture;
in vec4 color;
in vec2 texCoordV;
out vec4 fragColor;

void main() {
    // fragColor = vec4(texture(Texture, texCoordV));
    fragColor = vec4(color.xyz, 1) + 0.00001 * vec4(texture(Texture, texCoordV));
}
