#version 330

uniform sampler2D Texture;
in vec3 color;
in vec2 texCoordV;
out vec4 fragColor;

void main() {
    fragColor = vec4(texCoordV, 0, 1);
    fragColor = vec4(texture(Texture, texCoordV));
    //fragColor = vec4(0, 0, 1, 1);
}
