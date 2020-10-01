"""
Example headless EGL rendering on linux in a VM
"""
import numpy as np
from PIL import Image
import moderngl
from math import sin, cos, pi, asin, tan

ctx = moderngl.create_context(
    standalone=True,
    backend='egl',
    libgl='libGL.so.1',
    libegl='libEGL.so.1',
)
ctx.enable(moderngl.DEPTH_TEST|moderngl.CULL_FACE)
RESOLUTION = 5120
fbo = ctx.simple_framebuffer((5120, 5120), components=4)
fbo.use()

cube_vertices = np.array([
    #x  y   z
    -1, -1, -1,     #
    -1, -1, 1,      #
    -1, 1, -1,      #
    -1, 1, 1,       #
    1, -1, -1,      #
    1, -1, 1,       #
    1, 1, -1,       #
    1, 1, 1         #
], dtype="f4")
cube_indecies = np.array([
    0, 2, 1,        # West
    1, 2, 3,        # West
    4, 6, 5,        # East
    5, 6, 7,        # East
    0, 4, 1,        # Bottom
    1, 4, 5,        # Bottom
    2, 6, 3,        # Top
    3, 6, 7,        # Top
    0, 4, 2,        # South
    2, 4, 6,        # South
    1, 5, 3,        # North
    3, 5, 7         # North
], dtype="i4")


cube_prog = ctx.program(vertex_shader="""
#version 330
uniform mat4 Mvp;
in vec3 in_vert;
out vec3 color;
void main() {
    gl_Position = Mvp * vec4(in_vert, 1.0);
    color = vec3(0.0, gl_Position.z, 1.0);
}
""",
    fragment_shader="""
#version 330
out vec4 fragColor;
in vec3 color;
void main() {
    fragColor = vec4(color, 1.0);
}
""")

ctx.clear(1, 0, 0, 0, 1)

cube_vbo = ctx.buffer(cube_vertices.tobytes())
cube_ibo = ctx.buffer(cube_indecies.tobytes())

cube_vao = ctx.vertex_array(cube_prog, [(cube_vbo, "3f", "in_vert")], cube_ibo)

# vao.render(mode=moderngl.TRIANGLES)

scale_mat = np.array([
    [.707, 0,  0,  0],
    [0,  .6124, 0,  0],
    [0,  0,  .5, 0],
    [0,  0,  0,  1]
])
# Rotation matricies from Wikipedia: https://en.wikipedia.org/wiki/Rotation_matrix
alpha = asin(tan(pi/6))
s = sin(alpha)
c = cos(alpha)
rot_x = np.array([
    [1, 0, 0, 0],
    [0, c, -s, 0],
    [0, s, c, 0],
    [0, 0, 0, 1]
])
alpha = pi / 4
s = sin(alpha)
c = cos(alpha)
rot_y = np.array([
    [c, 0, s, 0],
    [0, 1, 0, 0],
    [-s, 0, c, 0],
    [0, 0, 0, 1]
])
# roty*rotx is the normal isometric view, the scale makes it fit into a sqare after applying the isometric view
mat = np.matmul(np.matmul(rot_y, rot_x), scale_mat)
cube_prog["Mvp"].write((mat).astype('f4').tobytes())

# ctx.wireframe = True
cube_vao.render(mode=moderngl.TRIANGLES)


image = Image.frombytes('RGBA', (RESOLUTION, RESOLUTION), fbo.read(components=4))
image = image.transpose(Image.FLIP_TOP_BOTTOM)
image.save('triangle.png', format='png')