from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .cython import mesh_core_cython


def rasterize_triangles(vertices, triangles, h, w):
    """
    raterize triangles
    Each triangle has 3 vertices & Each vertex has 3 coordinates x, y, z.
    h, w is the size of rendering

    :param: vertices: (n_vertex, 3)
    :param: triangles: (n_triangle, 3)
    :param: h: height
    :param: w: width
    :return: depth_buffer: [h, w] saves the depth, here, the bigger the z, the fronter the point.
        triangle_buffer: [h, w] saves the tri id(-1 for no triangle).
        barycentric_weight: [h, w, 3] saves corresponding barycentric weight.
    """

    # initial 
    depth_buffer = np.zeros([h, w]) - 999999.  # set the initial z to the farest position
    triangle_buffer = np.zeros([h, w], dtype=np.int32) - 1  # if tri id = -1, the pixel has no triangle correspondance
    barycentric_weight = np.zeros([h, w, 3], dtype=np.float32)  #

    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()

    mesh_core_cython.rasterize_triangles_core(
        vertices, triangles,
        depth_buffer, triangle_buffer, barycentric_weight,
        vertices.shape[0], triangles.shape[0],
        h, w)


def render_colors(vertices, triangles, colors, h, w, channels=3, background_img=None):
    """
    render mesh with colors
    :param: vertices: (nver, 3)
    :param: triangles: (n_triangle, 3)
    :param: colors: (n_vertex, 3)
    :param: h: height
    :param: w: width
    :param: channels: channel
    :param: background_img: background image
    :return: image: [h, w, c]. rendered image
    """

    # initial 
    if background_img is None:
        image = np.zeros((h, w, channels), dtype=np.float32)
    else:
        assert background_img.shape[0] == h and background_img.shape[1] == w and background_img.shape[2] == channels
        image = background_img
    depth_buffer = np.zeros([h, w], dtype=np.float32, order='C') - 999999.

    # change orders. --> C-contiguous order(column major)
    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()
    colors = colors.astype(np.float32).copy()

    mesh_core_cython.render_colors_core(
        image, vertices, triangles,
        colors,
        depth_buffer,
        vertices.shape[0], triangles.shape[0],
        h, w, channels)
    return image


def render_texture(vertices, triangles, texture, tex_coords, tex_triangles, h, w, channels=3, mapping_type='nearest',
                   background_img=None):
    """
    render mesh with texture map
    :param: vertices: (n_vertex, 3)
    :param: triangles: (n_triangle, 3)
    :param: texture: (tex_h, tex_w, 3)
    :param: tex_coords: (n_tex_coords, 3)
    :param: tex_triangles: (ntri, 3)
    :param: h: height of rendering
    :param: w: width of rendering
    :param: channels: channel
    :param: mapping_type: 'bilinear' or 'nearest'
    """
    # initial 
    if background_img is None:
        image = np.zeros((h, w, channels), dtype=np.float32)
    else:
        assert background_img.shape[0] == h and background_img.shape[1] == w and background_img.shape[2] == channels
        image = background_img

    depth_buffer = np.zeros([h, w], dtype=np.float32, order='C') - 999999.

    tex_h, tex_w, tex_c = texture.shape
    if mapping_type == 'nearest':
        mt = int(0)
    elif mapping_type == 'bilinear':
        mt = int(1)
    else:
        mt = int(0)

    # -> C order
    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()
    texture = texture.astype(np.float32).copy()
    tex_coords = tex_coords.astype(np.float32).copy()
    tex_triangles = tex_triangles.astype(np.int32).copy()

    mesh_core_cython.render_texture_core(
        image, vertices, triangles,
        texture, tex_coords, tex_triangles,
        depth_buffer,
        vertices.shape[0], tex_coords.shape[0], triangles.shape[0],
        h, w, channels,
        tex_h, tex_w, tex_c,
        mt)
    return image


def generate_vertex_norm(vertices, triangles, n_vertices, n_triangles):
    """
    generate vertex norm for each vertex using norms of triangles
    :param vertices: (n_vertex, 3)
    :param triangles: (n_triangle, 3)
    :param n_vertices: number of vertices
    :param n_triangles: number of triangles
    :return: norm: (n_vertex, 3)
    """
    # pt1 shape of (ntri, 3)
    pt1 = vertices[triangles[:, 0], :]
    pt2 = vertices[triangles[:, 1], :]
    pt3 = vertices[triangles[:, 2], :]

    # norm of triangle of shape (ntri, 3)
    norm_tri = np.cross(pt1 - pt2, pt1 - pt3)

    # norm of vertices
    N = np.zeros((n_vertices, 3))
    N = tnorm_to_vnorm(vertex_norm=N, n_vertex=n_vertices, triangle_norm=norm_tri, triangles=triangles,
                       n_triangle=n_triangles)
    # mag of shape (nver, 1)
    mag = np.sum(np.square(N), 1, keepdims=True)
    # deal with zero vector
    index = np.where(mag == 0)
    mag[index[0]] = 1
    N[index[0], 1] = 1
    N = np.divide(N, np.repeat(np.sqrt(mag), 3, 1))
    return -N


def tnorm_to_vnorm(vertex_norm, n_vertex, triangle_norm, triangles, n_triangle):
    """
    triangle norm to vertex norm
    vertex norm is the average of the triangle norms to which it's adjacent
    :param vertex_norm:
    :param n_vertex:
    :param triangle_norm:
    :param triangles:
    :param n_triangle:
    :return:
    """
    mesh_core_cython.tnorm_to_vnorm(vertex_norm, n_vertex, triangle_norm, triangles, n_triangle)

    return vertex_norm
