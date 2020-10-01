#version 330

// These are always the same
uniform mat4 Mvp;
uniform vec3 dir_light;
uniform uint atlas_size;

// Model Transform and MC specific values
uniform vec3 pos;
uniform vec3 scale;
uniform int face_texture_ids[6];
uniform vec4 face_uvs[6];
uniform float face_rotation[6];
uniform vec2 model_rot;
uniform bool uvlock;
uniform vec3 rotation_origin;
uniform vec3 rotation_axis;
uniform float rotation_angle;
uniform int face_tintindex[6];

// Values for each vertex
in vec3 in_vert;
in vec3 in_normal;
in vec2 in_texcoord_0;
in uint in_faceid;

// Outputs passed to the fragment shader
out vec4 color;
out float lum;
out vec2 uv;
out vec2 tile_offset;
flat out int tintindex;

void main() {
    if (face_texture_ids[in_faceid] == -1)
    {
        // Don't render anything if there is no texture for this face
        gl_Position = vec4(0, 0, 0, 0);
    }
    else
    {

        //color = vec4(0.0001 * texCoordV.x, in_faceid * 0.125, 0, 1);
        // face_texture_ids[in_faceid] * 0.124

        // mat4 translate_mat;
        // translate_mat[0] = vec4(1.0, 0.0, 0.0, pos.x);
        // translate_mat[1] = vec4(0.0, 1.0, 0.0, pos.y);
        // translate_mat[2] = vec4(0.0, 0.0, 1.0, pos.z);
        // translate_mat[3] = vec4(0.0, 0.0, 0.0, 1.0);

        // mat4 scale_mat;
        // scale_mat[0] = vec4(scale.x, 0, 0, 0);
        // scale_mat[1] = vec4(0, scale.y, 0, 0);
        // scale_mat[2] = vec4(0, 0, scale.z, 0);
        // scale_mat[3] = vec4(0, 0, 0, 1);

        // Matrix for rotating the entire model (gets applied after scale and pos)
        mat4 model_rot_matrix;
        model_rot_matrix[0] = vec4(cos(model_rot.y), 0.0, sin(model_rot.y), 0.0);
        model_rot_matrix[1] = vec4(sin(model_rot.y)*sin(model_rot.x), cos(model_rot.x), -sin(model_rot.x)*cos(model_rot.y), 0.0);
        model_rot_matrix[2] = vec4(-sin(model_rot.y)*cos(model_rot.x), sin(model_rot.x), cos(model_rot.x)*cos(model_rot.y), 0.0);
        model_rot_matrix[3] = vec4(0.0, 0.0, 0.0, 1.0);

        // Matrix for rotating only the element (get's applied before pos but after scale)
        // Source: https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
        mat3 rot_matrix;
        rot_matrix[0][0] = cos(rotation_angle) + rotation_axis.x * rotation_axis.x * (1 - cos(rotation_angle));
        rot_matrix[0][1] = rotation_axis.y * rotation_axis.x * (1 - cos(rotation_angle)) + rotation_axis.z * sin(rotation_angle);
        rot_matrix[0][2] = rotation_axis.z * rotation_axis.x * (1 - cos(rotation_angle)) - rotation_axis.y * sin(rotation_angle);
        rot_matrix[1][0] = rotation_axis.x * rotation_axis.y * (1 - cos(rotation_angle)) - rotation_axis.z * sin(rotation_angle);
        rot_matrix[1][1] = cos(rotation_angle) + rotation_axis.y * rotation_axis.y * (1 - cos(rotation_angle));
        rot_matrix[1][2] = rotation_axis.z * rotation_axis.y * (1 - cos(rotation_angle)) + rotation_axis.x * sin(rotation_angle);
        rot_matrix[2][0] = rotation_axis.x * rotation_axis.z * (1 - cos(rotation_angle)) + rotation_axis.y * sin(rotation_angle);
        rot_matrix[2][1] = rotation_axis.y * rotation_axis.z * (1 - cos(rotation_angle)) - rotation_axis.x * sin(rotation_angle);
        rot_matrix[2][2] = cos(rotation_angle) + rotation_axis.z * rotation_axis.z * (1 - cos(rotation_angle));

        // Apply Transform
        vec4 pos_in_block = model_rot_matrix * (vec4(
        rot_matrix * ((in_vert * scale) - rotation_origin) + rotation_origin + pos,
        1.0)
        );
        gl_Position = Mvp * pos_in_block;
        vec4 rot_normals = normalize(model_rot_matrix * vec4(rot_matrix * in_normal, 0.0));

        // UV coordinates
        if (uvlock) {
            // Calculate the UVs from the world-position of the vertex
            uv = vec2(0.5 + pos_in_block.x, 0.5 - pos_in_block.y) * abs(rot_normals.z) +
                vec2(0.5 - pos_in_block.z, 0.5 + pos_in_block.y) * rot_normals.x +
                vec2(0.5 + pos_in_block.x, 0.5 + pos_in_block.z) * rot_normals.y;
        }
        else {
            // Calculate the UVs from texcoord and the UV given by the json files
            float face_rotation_angle = face_rotation[in_faceid];
            mat2 face_rotation_mat;
            face_rotation_mat[0] = vec2(cos(face_rotation_angle), -sin(face_rotation_angle));
            face_rotation_mat[1] = vec2(sin(face_rotation_angle), cos(face_rotation_angle));

            vec2 rotated_texcoord_0 = face_rotation_mat * (in_texcoord_0 - 0.5) + 0.5;
            uv = rotated_texcoord_0 * face_uvs[in_faceid].xy + (1-rotated_texcoord_0) * face_uvs[in_faceid].zw;
        }
        // Convert the texture ID to the tile offset. This could have been done in python, too
        uint texture_index = uint(face_texture_ids[in_faceid]);
        tile_offset = vec2(texture_index % atlas_size, texture_index / atlas_size) / atlas_size;

        // Light
        lum = max(dot(rot_normals.xyz, dir_light), 0.0);

        // tintindex
        tintindex = face_tintindex[in_faceid];
    }
}
