#version 330

uniform sampler2D textureAtlas;
uniform float tile_size;
in vec2 uv;
in vec2 tile_offset;
in float lum;
out vec4 fragColor;

void main() {
    // GL_REPEAT workaround when a TextureAtlas is used
    vec2 texCoord = tile_offset + tile_size * fract(uv);
    fragColor = vec4(lum, lum, lum, 1.0) * texture(textureAtlas, texCoord);
    if (fragColor.a < 0.001)
        discard;
}
