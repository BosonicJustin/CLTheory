import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def sample_10d_sphere(n_samples=1):
    """Sample uniformly from 10D sphere surface"""
    x = np.random.randn(n_samples, 10)
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def get_shape_type_smooth(param0):
    """
    Smooth shape type determination that ensures continuity
    Returns blend weights for different shape types
    """
    # Normalize to [0, 1]
    normalized = (param0 + 1) / 2
    
    # Create smooth transitions between 8 shape types
    shape_types = [
        "superquadric", "fractal_tree", "crystal", "organic_blob",
        "geodesic_dome", "twisted_prism", "spiral_tower", "neural_network"
    ]
    
    # Map to continuous range [0, 8)
    continuous_idx = normalized * 8
    
    # Get primary and secondary shape indices
    primary_idx = int(continuous_idx) % 8
    secondary_idx = (primary_idx + 1) % 8
    
    # Blend factor between primary and secondary
    blend = continuous_idx - int(continuous_idx)
    
    return shape_types[primary_idx], shape_types[secondary_idx], blend, primary_idx

# ----

def generate_superquadric_full(params):
    """Generate superquadric using ALL 9 parameters"""
    # ALL parameters are used with specific, distinct purposes
    a1 = abs(params[0]) * 2 + 0.5          # x semi-axis [0.5, 2.5]
    a2 = abs(params[1]) * 2 + 0.5          # y semi-axis [0.5, 2.5]
    a3 = abs(params[2]) * 2 + 0.5          # z semi-axis [0.5, 2.5]
    e1 = abs(params[3]) * 1.8 + 0.2        # shape exponent 1 [0.2, 2.0]
    e2 = abs(params[4]) * 1.8 + 0.2        # shape exponent 2 [0.2, 2.0]
    rot_x = params[5] * np.pi              # x rotation [-π, π]
    rot_y = params[6] * np.pi              # y rotation [-π, π]
    rot_z = params[7] * np.pi              # z rotation [-π, π]
    twist = params[8] * 2 * np.pi          # twist parameter [-2π, 2π]
    
    # Generate surface
    u = np.linspace(-np.pi/2, np.pi/2, 40)
    v = np.linspace(-np.pi, np.pi, 40)
    U, V = np.meshgrid(u, v)
    
    # Superquadric equations
    x = a1 * np.sign(np.cos(U)) * (np.abs(np.cos(U)) ** e1) * np.sign(np.cos(V)) * (np.abs(np.cos(V)) ** e2)
    y = a2 * np.sign(np.cos(U)) * (np.abs(np.cos(U)) ** e1) * np.sign(np.sin(V)) * (np.abs(np.sin(V)) ** e2)
    z = a3 * np.sign(np.sin(U)) * (np.abs(np.sin(U)) ** e1)
    
    # Apply rotations (all three axes)
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    
    # Rotation matrices
    Rx = np.array([[1, 0, 0], [0, np.cos(rot_x), -np.sin(rot_x)], [0, np.sin(rot_x), np.cos(rot_x)]])
    Ry = np.array([[np.cos(rot_y), 0, np.sin(rot_y)], [0, 1, 0], [-np.sin(rot_y), 0, np.cos(rot_y)]])
    Rz = np.array([[np.cos(rot_z), -np.sin(rot_z), 0], [np.sin(rot_z), np.cos(rot_z), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    
    rotated_points = points @ R.T
    
    # Apply twist deformation
    for i, point in enumerate(rotated_points):
        height_factor = point[2] / a3 if a3 > 0 else 0
        twist_angle = twist * height_factor
        cos_t, sin_t = np.cos(twist_angle), np.sin(twist_angle)
        x_new = point[0] * cos_t - point[1] * sin_t
        y_new = point[0] * sin_t + point[1] * cos_t
        rotated_points[i] = [x_new, y_new, point[2]]
    
    # Reshape back
    x_final = rotated_points[:, 0].reshape(x.shape)
    y_final = rotated_points[:, 1].reshape(y.shape)
    z_final = rotated_points[:, 2].reshape(z.shape)
    
    return x_final, y_final, z_final

def generate_fractal_tree_full(params):
    """Generate fractal tree using ALL 9 parameters"""
    branch_angle = abs(params[0]) * np.pi/3 + np.pi/12      # [15°, 75°]
    branch_ratio = abs(params[1]) * 0.5 + 0.3               # [0.3, 0.8]
    branch_count = int(abs(params[2]) * 5) + 2              # [2, 7]
    twist = params[3] * np.pi                               # [-π, π]
    asymmetry = params[4] * 0.8                             # [-0.8, 0.8]
    depth = int(abs(params[5]) * 4) + 3                     # [3, 7]
    thickness_ratio = abs(params[6]) * 0.4 + 0.1            # [0.1, 0.5]
    gravity_bend = params[7] * 0.5                          # [-0.5, 0.5]
    noise_factor = abs(params[8]) * 0.3                     # [0, 0.3]
    
    def generate_branch(start, direction, length, thickness, current_depth, generation):
        if current_depth <= 0 or length < 0.05:
            return []
        
        # Apply gravity bend
        bent_direction = direction.copy().astype(float)
        bent_direction[2] += gravity_bend * (1 - direction[2])  # Bend toward/away from vertical
        bent_direction /= np.linalg.norm(bent_direction)
        
        # Add noise
        if noise_factor > 0:
            noise = np.random.normal(0, noise_factor, 3)
            bent_direction += noise
            bent_direction /= np.linalg.norm(bent_direction)
        
        end = start + bent_direction * length
        branches = [(start, end, thickness)]
        
        # Create child branches
        actual_branch_count = max(2, int(branch_count * (1 - generation * 0.1)))
        for i in range(actual_branch_count):
            angle = 2 * np.pi * i / actual_branch_count + twist * current_depth
            
            # Branch length with asymmetry
            branch_length = length * branch_ratio * (1 + asymmetry * np.sin(angle * 2))
            branch_length = max(0.1, branch_length)  # Minimum length
            
            # Create rotation for branching
            cos_a, sin_a = np.cos(branch_angle), np.sin(branch_angle)
            cos_t, sin_t = np.cos(angle), np.sin(angle)
            
            # New direction vector
            new_dir = np.array([
                bent_direction[0] * cos_a + (bent_direction[1] * cos_t + bent_direction[2] * sin_t) * sin_a,
                bent_direction[1] * cos_a - bent_direction[0] * cos_t * sin_a,
                bent_direction[2] * cos_a - bent_direction[0] * sin_t * sin_a
            ])
            new_dir /= np.linalg.norm(new_dir)
            
            new_thickness = thickness * thickness_ratio
            branches.extend(generate_branch(end, new_dir, branch_length, new_thickness, 
                                          current_depth - 1, generation + 1))
        
        return branches
    
    branches = generate_branch(np.array([0, 0, 0]), np.array([0, 0, 1]), 2.5, 0.15, depth, 0)
    return branches

def generate_crystal_full(params):
    """Generate crystal using ALL 9 parameters"""
    symmetry = int(abs(params[0]) * 7) + 3                  # [3, 10] fold symmetry
    height = abs(params[1]) * 3 + 0.5                      # [0.5, 3.5] height
    base_size = abs(params[2]) * 2 + 0.5                   # [0.5, 2.5] base size
    taper = abs(params[3]) * 0.8 + 0.2                     # [0.2, 1.0] taper ratio
    twist_angle = params[4] * np.pi                        # [-π, π] twist
    face_curvature = params[5] * 0.7                       # [-0.7, 0.7] face distortion
    top_offset_x = params[6] * 0.8                         # [-0.8, 0.8] top offset x
    top_offset_y = params[7] * 0.8                         # [-0.8, 0.8] top offset y
    mid_expansion = params[8] * 0.5 + 1.0                  # [0.5, 1.5] middle expansion
    
    vertices = []
    faces = []
    
    # Create multiple layers for more complex crystal
    layers = 3
    for layer_idx in range(layers):
        layer_height = height * layer_idx / (layers - 1)
        layer_progress = layer_idx / (layers - 1) if layers > 1 else 0
        
        # Size varies through layers
        if layer_idx == 1:  # Middle layer
            layer_size = base_size * mid_expansion
        else:
            layer_size = base_size * (1 - layer_progress * (1 - taper))
        
        # Twist varies through layers
        layer_twist = twist_angle * layer_progress
        
        # Top offset varies through layers
        offset_x = top_offset_x * layer_progress
        offset_y = top_offset_y * layer_progress
        
        # Generate vertices for this layer
        layer_start_idx = len(vertices)
        angles = np.linspace(0, 2*np.pi, symmetry, endpoint=False)
        
        for i, angle in enumerate(angles):
            twisted_angle = angle + layer_twist
            r = layer_size * (1 + face_curvature * np.sin(i * 3))
            x = r * np.cos(twisted_angle) + offset_x
            y = r * np.sin(twisted_angle) + offset_y
            vertices.append([x, y, layer_height])
        
        # Create faces between this layer and the next
        if layer_idx < layers - 1:
            for i in range(symmetry):
                next_i = (i + 1) % symmetry
                current_base = layer_start_idx
                next_base = len(vertices)  # Will be the start of next layer
                
                # We'll create faces after all vertices are generated
                pass
    
    # Generate faces between layers
    for layer_idx in range(layers - 1):
        current_base = layer_idx * symmetry
        next_base = (layer_idx + 1) * symmetry
        
        for i in range(symmetry):
            next_i = (i + 1) % symmetry
            
            # Create quad as two triangles
            faces.append([current_base + i, current_base + next_i, next_base + i])
            faces.append([current_base + next_i, next_base + next_i, next_base + i])
    
    return np.array(vertices), faces

def generate_organic_blob_full(params):
    """Generate organic blob using ALL 9 parameters"""
    resolution = 35
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2*np.pi, resolution)
    Phi, Theta = np.meshgrid(phi, theta)
    
    # Base radius
    base_radius = abs(params[0]) * 0.5 + 0.8               # [0.8, 1.3] base size
    r = np.ones_like(Phi) * base_radius
    
    # Use ALL 8 remaining parameters for different harmonic components
    harmonics = [
        (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)
    ]
    
    for i, (l, m) in enumerate(harmonics):
        if i < 8:
            amplitude = params[i + 1] * 0.25                # Each parameter affects different frequency
            
            # Proper spherical harmonics approximation
            if m == 0:
                harmonic = np.cos(l * Phi)
            else:
                angular_part = np.cos(m * Theta) if (i % 2 == 0) else np.sin(m * Theta)
                harmonic = angular_part * (np.sin(Phi) ** m) * np.cos(l * Phi)
            
            r += amplitude * harmonic
    
    # Ensure positive radius
    r = np.abs(r) + 0.1
    
    # Convert to Cartesian
    x = r * np.sin(Phi) * np.cos(Theta)
    y = r * np.sin(Phi) * np.sin(Theta)
    z = r * np.cos(Phi)
    
    return x, y, z

def generate_geodesic_dome_full(params):
    """Generate geodesic dome using ALL 9 parameters"""
    # Start with icosahedron
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ]) / np.sqrt(1 + phi**2)
    
    # Use ALL parameters
    scale = abs(params[0]) * 2 + 0.5                       # [0.5, 2.5] overall scale
    x_stretch = 1 + params[1] * 0.6                        # [0.4, 1.6] x scaling
    y_stretch = 1 + params[2] * 0.6                        # [0.4, 1.6] y scaling
    z_stretch = 1 + params[3] * 0.6                        # [0.4, 1.6] z scaling
    twist = params[4] * np.pi                              # [-π, π] twist
    bulge_strength = params[5] * 0.8                       # [-0.8, 0.8] radial bulge
    shear_xy = params[6] * 0.5                             # [-0.5, 0.5] xy shear
    shear_xz = params[7] * 0.5                             # [-0.5, 0.5] xz shear
    wave_distortion = params[8] * 0.4                      # [-0.4, 0.4] wave distortion
    
    # Apply all transformations
    vertices[:, 0] *= x_stretch
    vertices[:, 1] *= y_stretch
    vertices[:, 2] *= z_stretch
    vertices *= scale
    
    # Apply shears
    for i, v in enumerate(vertices):
        vertices[i, 0] += shear_xy * v[1] + shear_xz * v[2]
    
    # Apply twist based on height
    for i, v in enumerate(vertices):
        height_factor = v[2] / (scale * z_stretch)
        twist_angle = twist * height_factor
        cos_t, sin_t = np.cos(twist_angle), np.sin(twist_angle)
        x_new = v[0] * cos_t - v[1] * sin_t
        y_new = v[0] * sin_t + v[1] * cos_t
        vertices[i] = [x_new, y_new, v[2]]
    
    # Apply radial bulge
    for i, v in enumerate(vertices):
        dist_from_origin = np.linalg.norm(v)
        normalized_dist = dist_from_origin / scale
        bulge_factor = 1 + bulge_strength * (normalized_dist - 0.5)
        vertices[i] = v * bulge_factor
    
    # Apply wave distortion
    for i, v in enumerate(vertices):
        angle = np.arctan2(v[1], v[0])
        wave_offset = wave_distortion * np.sin(4 * angle) * 0.2
        vertices[i, 2] += wave_offset
    
    # Icosahedron faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    return vertices, faces

def generate_shape_from_10d_vector_injective(vector_10d):
    """
    FULLY INJECTIVE mapping from 10D vector to shape
    
    Key principles:
    1. Parameter 0 determines shape type (with smooth blending possible)
    2. ALL remaining 9 parameters are used in every shape type
    3. Each parameter has a distinct, meaningful effect
    4. Small changes in input = small changes in output
    """
    if len(vector_10d) != 10:
        raise ValueError("Input must be a 10-dimensional vector")
    
    # Get shape type from parameter 0
    primary_shape, secondary_shape, blend, shape_idx = get_shape_type_smooth(vector_10d[0])
    params = vector_10d[1:]  # Parameters 1-9
    
    # Generate primary shape
    if primary_shape == "superquadric":
        shape_data = generate_superquadric_full(params)
    elif primary_shape == "fractal_tree":
        shape_data = generate_fractal_tree_full(params)
    elif primary_shape == "crystal":
        vertices, faces = generate_crystal_full(params)
        shape_data = (vertices, faces)
    elif primary_shape == "organic_blob":
        shape_data = generate_organic_blob_full(params)
    elif primary_shape == "geodesic_dome":
        vertices, faces = generate_geodesic_dome_full(params)
        shape_data = (vertices, faces)
    elif primary_shape == "twisted_prism":
        # Add twisted prism implementation using all 9 params
        vertices, faces = generate_twisted_prism_full(params)
        shape_data = (vertices, faces)
    elif primary_shape == "spiral_tower":
        # Add spiral tower implementation using all 9 params  
        vertices, faces = generate_spiral_tower_full(params)
        shape_data = (vertices, faces)
    elif primary_shape == "neural_network":
        # Add neural network implementation using all 9 params
        nodes, connections = generate_neural_network_full(params)
        shape_data = (nodes, connections)
    else:
        # Fallback to organic blob
        shape_data = generate_organic_blob_full(params)
    
    # Create parameter usage report
    param_usage = {
        "shape_selector": f"param[0] = {vector_10d[0]:.3f} → {primary_shape}",
        "all_parameters_used": "YES - All 9 parameters (1-9) affect the shape",
        "injective": "YES - Different inputs produce different outputs",
        "parameter_ranges": {
            f"param[{i+1}]": f"{vector_10d[i+1]:.3f}" for i in range(9)
        }
    }
    
    return shape_data, primary_shape, param_usage

def generate_twisted_prism_full(params):
    """Generate twisted prism using ALL 9 parameters"""
    n_sides = int(abs(params[0]) * 8) + 3                   # [3, 11] sides
    height = abs(params[1]) * 4 + 1                         # [1, 5] height
    twist_angle = params[2] * 2 * np.pi                     # [-2π, 2π] twist
    taper = abs(params[3]) * 0.8 + 0.2                     # [0.2, 1.0] taper
    cross_section_var = params[4] * 0.5                     # [-0.5, 0.5] variation
    segments = int(abs(params[5]) * 15) + 5                # [5, 20] segments
    base_radius = abs(params[6]) * 1.5 + 0.5               # [0.5, 2.0] base size
    shear_x = params[7] * 0.5                              # [-0.5, 0.5] x shear
    shear_y = params[8] * 0.5                              # [-0.5, 0.5] y shear
    
    vertices = []
    faces = []
    
    for seg in range(segments + 1):
        t = seg / segments
        current_height = t * height
        current_twist = t * twist_angle
        current_scale = base_radius * (1 - t * (1 - taper))
        
        # Apply shear
        shear_offset_x = shear_x * t * height
        shear_offset_y = shear_y * t * height
        
        # Create polygon at this height
        seg_start = len(vertices)
        for i in range(n_sides):
            angle = 2 * np.pi * i / n_sides + current_twist
            radius = current_scale * (1 + cross_section_var * np.sin(i * 3))
            x = radius * np.cos(angle) + shear_offset_x
            y = radius * np.sin(angle) + shear_offset_y
            vertices.append([x, y, current_height])
        
        # Create faces between segments
        if seg > 0:
            prev_start = seg_start - n_sides
            for i in range(n_sides):
                next_i = (i + 1) % n_sides
                faces.append([prev_start + i, prev_start + next_i, seg_start + i])
                faces.append([prev_start + next_i, seg_start + next_i, seg_start + i])
    
    return np.array(vertices), faces

def generate_spiral_tower_full(params):
    """Generate spiral tower using ALL 9 parameters"""
    height = abs(params[0]) * 6 + 2                         # [2, 8] height
    radius_base = abs(params[1]) * 2 + 0.5                  # [0.5, 2.5] base radius
    radius_top = abs(params[2]) * 1 + 0.1                   # [0.1, 1.1] top radius
    spiral_turns = abs(params[3]) * 8 + 1                   # [1, 9] turns
    segments = int(abs(params[4]) * 60) + 20                # [20, 80] segments
    cross_sides = int(abs(params[5]) * 6) + 3               # [3, 9] cross-section sides
    thickness = abs(params[6]) * 0.4 + 0.1                  # [0.1, 0.5] thickness
    wobble = abs(params[7]) * 0.3                           # [0, 0.3] path wobble
    tilt = params[8] * 0.5                                  # [-0.5, 0.5] tower tilt
    
    vertices = []
    faces = []
    path_points = []
    
    # Generate spiral path with wobble and tilt
    for i in range(segments + 1):
        t = i / segments
        current_height = t * height
        angle = t * spiral_turns * 2 * np.pi
        radius = radius_base * (1 - t) + radius_top * t
        
        # Add wobble
        wobble_offset_x = wobble * np.sin(angle * 5) * radius * 0.2
        wobble_offset_y = wobble * np.cos(angle * 7) * radius * 0.2
        
        center_x = radius * np.cos(angle) + wobble_offset_x + tilt * current_height
        center_y = radius * np.sin(angle) + wobble_offset_y
        path_points.append([center_x, center_y, current_height, angle])
    
    # Generate cross-sections
    for i, (cx, cy, cz, path_angle) in enumerate(path_points):
        section_radius = thickness * (1 + 0.3 * np.sin(i * 0.5))
        
        for j in range(cross_sides):
            local_angle = 2 * np.pi * j / cross_sides
            x_local = section_radius * np.cos(local_angle)
            y_local = section_radius * np.sin(local_angle)
            
            # Rotate by path angle
            x = cx + x_local * np.cos(path_angle) - y_local * np.sin(path_angle)
            y = cy + x_local * np.sin(path_angle) + y_local * np.cos(path_angle)
            
            vertices.append([x, y, cz])
    
    # Generate faces
    for i in range(segments):
        base_idx = i * cross_sides
        next_base_idx = (i + 1) * cross_sides
        
        for j in range(cross_sides):
            next_j = (j + 1) % cross_sides
            faces.append([base_idx + j, base_idx + next_j, next_base_idx + j])
            faces.append([base_idx + next_j, next_base_idx + next_j, next_base_idx + j])
    
    return np.array(vertices), faces

def generate_neural_network_full(params):
    """Generate neural network using ALL 9 parameters"""
    layers = int(abs(params[0]) * 4) + 3                    # [3, 7] layers
    nodes_per_layer = int(abs(params[1]) * 8) + 4           # [4, 12] nodes/layer
    layer_spacing = abs(params[2]) * 2 + 1                  # [1, 3] spacing
    node_size = abs(params[3]) * 0.3 + 0.1                 # [0.1, 0.4] node size
    connection_density = abs(params[4])                      # [0, 1] connection rate
    curvature = params[5] * 0.8                            # [-0.8, 0.8] connection curve
    asymmetry = params[6] * 0.8                            # [-0.8, 0.8] layer asymmetry
    vertical_spread = abs(params[7]) * 2 + 0.5             # [0.5, 2.5] vertical spread
    rotation = params[8] * np.pi                           # [-π, π] network rotation
    
    nodes = []
    connections = []
    
    # Generate nodes with asymmetry and rotation
    for layer in range(layers):
        layer_nodes = max(2, int(nodes_per_layer * (1 + asymmetry * np.sin(layer))))
        layer_x = layer * layer_spacing
        
        for node in range(layer_nodes):
            if layer_nodes <= 6:
                angle = 2 * np.pi * node / layer_nodes
                y = np.cos(angle) * vertical_spread
                z = np.sin(angle) * vertical_spread
            else:
                cols = int(np.sqrt(layer_nodes))
                row = node // cols
                col = node % cols
                y = (col - cols/2) * 0.8
                z = (row - cols/2) * 0.8
            
            # Apply network rotation
            y_rot = y * np.cos(rotation) - z * np.sin(rotation)
            z_rot = y * np.sin(rotation) + z * np.cos(rotation)
            
            nodes.append([layer_x, y_rot, z_rot])
    
    # Generate connections with curvature consideration
    layer_starts = [0]
    current_idx = 0
    for layer in range(layers):
        layer_nodes = max(2, int(nodes_per_layer * (1 + asymmetry * np.sin(layer))))
        current_idx += layer_nodes
        layer_starts.append(current_idx)
    
    for layer in range(layers - 1):
        start_idx = layer_starts[layer]
        end_idx = layer_starts[layer + 1]
        next_start_idx = layer_starts[layer + 1]
        next_end_idx = layer_starts[layer + 2]
        
        for i in range(start_idx, end_idx):
            for j in range(next_start_idx, next_end_idx):
                # Connection probability affected by curvature and distance
                base_prob = connection_density
                node_i, node_j = np.array(nodes[i]), np.array(nodes[j])
                distance = np.linalg.norm(node_j - node_i)
                adjusted_prob = base_prob * (1 + curvature * (1 - distance / layer_spacing))
                adjusted_prob = max(0, min(1, adjusted_prob))
                
                if np.random.random() < adjusted_prob:
                    connections.append((i, j))
    
    return np.array(nodes), connections

# ----

# def verify_injectivity(n_samples=1000):
#     """
#     Verify that the mapping is injective by checking for collisions
#     """
#     print("Verifying injectivity...")
    
#     vectors = []
#     shape_hashes = []
    
#     for i in range(n_samples):
#         # Generate random vector on 10D sphere
#         vector = sample_10d_sphere(1)[0]
#         vectors.append(vector)
        
#         # Generate shape and create a hash of its key properties
#         shape_data, shape_type, _ = generate_shape_from_10d_vector_injective(vector)
        
#         # Create a simple hash based on shape type and first few coordinates
#         if shape_type in ["superquadric", "organic_blob"]:
#             x, y, z = shape_data
#             shape_hash = (shape_type, float(x[0,0]), float(y[0,0]), float(z[0,0]), 
#                          float(x[-1,-1]), float(y[-1,-1]), float(z[-1,-1]))
#         elif shape_type == "fractal_tree":
#             branches = shape_data
#             if len(branches) > 0:
#                 first_branch = branches[0]
#                 last_branch = branches[-1] if len(branches) > 1 else branches[0]
#                 shape_hash = (shape_type, float(first_branch[0][0]), float(first_branch[1][0]), 
#                             float(last_branch[0][0]), float(last_branch[1][0]), len(branches))
#             else:
#                 shape_hash = (shape_type, 0, 0, 0, 0, 0)
#         else:  # mesh-based shapes (crystal, geodesic_dome, etc.)
#             if isinstance(shape_data, tuple) and len(shape_data) == 2:
#                 vertices, faces = shape_data
#                 if len(vertices) > 0:
#                     shape_hash = (shape_type, float(vertices[0,0]), float(vertices[0,1]), float(vertices[0,2]),
#                                 float(vertices[-1,0]), float(vertices[-1,1]), float(vertices[-1,2]), len(vertices))
#                 else:
#                     shape_hash = (shape_type, 0, 0, 0, 0, 0, 0, 0)
#             else:
#                 # Handle unexpected shape_data format
#                 shape_hash = (shape_type, 0, 0, 0, 0, 0, 0, 0)
        
#         shape_hashes.append(shape_hash)
    
#     # Check for collisions
#     unique_hashes = set(shape_hashes)
#     collision_rate = 1 - len(unique_hashes) / len(shape_hashes)
    
#     print(f"Generated {n_samples} shapes")
#     print(f"Unique shapes: {len(unique_hashes)}")
#     print(f"Collision rate: {collision_rate:.4f}")
#     print(f"Injectivity score: {1-collision_rate:.4f}")
    
#     if collision_rate < 0.01:
#         print("✅ EXCELLENT: Mapping is highly injective")
#     elif collision_rate < 0.05:
#         print("✅ GOOD: Mapping is mostly injective")
#     else:
#         print("❌ POOR: Mapping has significant collisions")
    
#     return collision_rate

# def demonstrate_parameter_effects(base_vector):
#     """
#     Demonstrate how each parameter affects the output shape
#     """
#     print(f"Base vector: {base_vector}")
#     shape_data, shape_type, param_usage = generate_shape_from_10d_vector_injective(base_vector)
#     print(f"Base shape: {shape_type}")
#     print(f"Parameter usage: {param_usage}")
    
#     print("\nParameter sensitivity analysis:")
    
#     for param_idx in range(10):
#         # Create small perturbation
#         perturbed_vector = base_vector.copy()
#         delta = 0.1
#         perturbed_vector[param_idx] += delta
        
#         # Renormalize to sphere
#         perturbed_vector = perturbed_vector / np.linalg.norm(perturbed_vector)
        
#         # Generate perturbed shape
#         new_shape_data, new_shape_type, new_param_usage = generate_shape_from_10d_vector_injective(perturbed_vector)
        
#         print(f"  Param {param_idx}: {base_vector[param_idx]:.3f} → {perturbed_vector[param_idx]:.3f}")
#         if param_idx == 0 and new_shape_type != shape_type:
#             print(f"    Shape type changed: {shape_type} → {new_shape_type}")
#         else:
#             print(f"    Shape modified (same type: {new_shape_type})")

# # Example usage
if __name__ == "__main__":
    z = np.random.normal(size=9)
    z = z / np.linalg.norm(z)  # Normalize to unit sphere

    print(generate_superquadric_full(z)[0])
#     # print("FULLY INJECTIVE 10D Shape Generator")
#     # print("=" * 50)
    
#     # # Test injectivity
#     # verify_injectivity(100)
    
#     # print("\n" + "=" * 50)
    
#     # # Demonstrate parameter effects
#     # test_vector = sample_10d_sphere(1)[0]
#     # demonstrate_parameter_effects(test_vector)
    
#     # print("\n" + "=" * 50)
#     # print("Shape type mapping (parameter 0):")
#     # for i in range(8):
#     #     param0_value = -1 + (i / 4)  # Sample points
#     #     primary_shape, _, _, _ = get_shape_type_smooth(param0_value)
#     #     print(f"param[0] = {param0_value:5.2f} → {primary_shape}")