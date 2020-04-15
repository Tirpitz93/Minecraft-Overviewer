#version 330

uniform sampler2D textureAtlas;
uniform float tile_size;
uniform bool is_colortint_pass;
in vec2 uv;
in vec2 tile_offset;
in float lum;
flat in int tintindex;
out vec4 fragColor;

void main() {
    // GL_REPEAT workaround when a TextureAtlas is used
    vec2 texCoord = tile_offset + tile_size * fract(uv);
    fragColor = vec4(lum, lum, lum, 1.0) * texture(textureAtlas, texCoord);

    // Discard invisible pixels (making transparent textures work correctly)
    if (fragColor.a < 0.001) {
        discard;
    }

    // In the colortint pass render all non tintindex faces black. Transparent pixels are already discarded
    if (is_colortint_pass && tintindex == -1) {
        fragColor = vec4(0, 0, 0, 1);
    }
}
