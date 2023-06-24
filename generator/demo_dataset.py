import os
import open3d as o3d
from utils import *


NO_EACH_CLASS = 15
# DATASET_FOLDER = "dataset"
DATASET_FOLDER = "smooth_dataset"
# SHUFFLE = False
SHUFFLE = True
# SAVE_PCD = False
SAVE_PCD = True
COLORFUL = True

to_be_drawn = []
dx, dy = 100, 100
classes = ['medium-can', 'food-can', 'small-cup', 'medium-cup', \
           'large-cup', 'medium-ball', 'tiny-ball', 'small-ball']
            # 'tiny-ball', 'small-ball', 'medium-ball']

for class_i in range(len(classes)):
    cls = classes[class_i]
    pcd_foder = f"{DATASET_FOLDER}/{cls}"

    # Randomly selected the files in the deformed dataset
    deformed_files = os.listdir(f'{pcd_foder}')
    random_selected_files = random.sample(deformed_files, NO_EACH_CLASS)
    
    count = 0
    for random_file in random_selected_files:
        pcd_path = f"{pcd_foder}/{random_file}"
                
        mesh = o3d.io.read_triangle_mesh(pcd_path)
        mesh.compute_vertex_normals()

        if COLORFUL == True:
            color = [round(random.uniform(0, 1), 1), \
                    round(random.uniform(0, 1), 1), \
                    round(random.uniform(0, 1), 1)]
            mesh.paint_uniform_color(color)
        
        # Move to the origin of coordinate systems
        shift_x, shift_y, _ = move_obj_to_coord_for_mesh(mesh, [0, 0, 0])
        mesh = translate_pcd(mesh, [shift_x, shift_y, 0])
        
        # Arrange mesh for visualization
        if SHUFFLE == False:
            mesh = translate_pcd(mesh, [class_i*dx, count*dy, 0])
            count += 1

        to_be_drawn.append(mesh)


if SHUFFLE == True:
    # Shuffle the list and divide it into a nested list
    random.shuffle(to_be_drawn)
    to_be_drawn = [to_be_drawn[n : n + NO_EACH_CLASS] \
                   for n in range(0, len(to_be_drawn), NO_EACH_CLASS)]

    # Arrange mesh for visualization
    for i in range(len(classes)):
        for j in range(NO_EACH_CLASS):
            to_be_drawn[i][j] = translate_pcd(to_be_drawn[i][j], [i*dx, j*dy, 0])
    
    # Flatten a nested list
    to_be_drawn = sum(to_be_drawn, [])          


# Create and save an integrated point cloud
if SAVE_PCD == True:
    demo_pcd = o3d.geometry.TriangleMesh()
    for pcd in range(len(to_be_drawn)):
        demo_pcd += to_be_drawn[pcd]
    o3d.io.write_triangle_mesh(filename=f"demo_dataset_{NO_EACH_CLASS}_1.ply", mesh=demo_pcd, write_ascii=True)

o3d.visualization.draw_geometries(to_be_drawn)
