#version 330

uniform mat4 Mvp;
uniform vec3 pos;
uniform vec3 scale;
uniform uint face_texture_ids[6];
uniform vec4 face_uvs[6];
//uniform vec2 model_rot;
//uniform vec3 dir_light;
in vec3 in_vert;
in vec3 in_normal;
in vec2 in_texcoord_0;
in uint in_faceid;
out vec3 texCoord;
out vec4 color;
//out float lum;

void main() {
    texCoord = vec3(in_texcoord_0 * face_uvs[in_faceid].xy + (1-in_texcoord_0) * face_uvs[in_faceid].zw + vec2(0, 0.0001) * in_normal.xy, face_texture_ids[in_faceid]);
    //color = vec4(0.0001 * texCoordV.x, in_faceid * 0.125, 0, 1);
    // face_texture_ids[in_faceid] * 0.124
    gl_Position = Mvp * vec4((in_vert * scale) + pos, 1.0);
}
