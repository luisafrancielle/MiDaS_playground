# Dependencies
import cv2
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

# Download the model
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

transform = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transform.small_transform

import os

def matching_files(folder1: str, folder2: str):
    names_folder1 = {os.path.splitext(f)[0].split('_')[0]: os.path.join(folder1, f) for f in os.listdir(folder1)}
    names_folder2 = {os.path.splitext(f)[0].split('_')[0]: os.path.join(folder2, f) for f in os.listdir(folder2)}
  
    common_names = set(names_folder1.keys()) & set(names_folder2.keys())

    matching_files = [(names_folder1[name], names_folder2[name]) for name in common_names]
    
    return matching_files


def get_heatmap_img(image_path):
    name, _ = os.path.splitext(os.path.basename(image_path))
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img_rgb).to('cpu')

    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()
        output = prediction.cpu().numpy()

    plt.imshow(output, cmap='gray')  
    plt.axis('off') 
    plt.savefig(f'heatmap_imgs\\{name}_heatmap.jpg', bbox_inches='tight', pad_inches=0)
    cv2.imshow('frame', output)
    cv2.waitKey(0)

def get_heatmap_folder(folder_path: str):
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        get_heatmap_img(image_path)

def create_RGBD_all(matching_files:list, folder:str):
    for matching in matching_files:
        rgb_img = cv2.imread(matching[0])
        heatmap = cv2.imread(matching[1], cv2.IMREAD_GRAYSCALE)

        if rgb_img is None or heatmap is None:
            continue

        heatmap_resized = cv2.resize(heatmap, (rgb_img.shape[1], rgb_img.shape[0]))

        depth = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX)

        rgbd = np.dstack((rgb_img, depth)).astype(np.uint8)
        
        name = os.path.basename(matching[0]).split('.')[0]
        
        cv2.imwrite(f'{folder}\\{name}_rgbd.png', rgbd)

# def save_rgbd_as_pcd(matching_files:list):

#     for matching in matching_files:
#         rgb_img = cv2.imread(matching[0])
#         depth_img = cv2.imread(matching[1], cv2.IMREAD_GRAYSCALE)
        
#         filename = os.path.basename(matching[0]).split('.')[0] + '_rgbd'

#         cloud = pcl.PointCloud()

#         height, width = depth_img.shape

#         points = []
#         for y in range(height):
#             for x in range(width):
#                 depth_value = depth_img[y, x]

#                 if depth_value == 0:
#                     continue

#                 point = [x, y, depth_value]

#                 point += list(rgb_img[y, x])

#                 points.append(point)

#         cloud.from_array(np.array(points, dtype=np.float32))

#         pcl.save(cloud, filename)
        
def visualize_rgbd_as_3d(rgbd_image_path):

    rgbd_img = cv2.imread(rgbd_image_path, cv2.IMREAD_UNCHANGED)
    
    rgb = rgbd_img[:, :, :3]
    depth = rgbd_img[:, :, 3]


    height, width, _ = rgb.shape
    fx, fy = 256, 256  
    cx, cy = width // 2, height // 2  

    # Create mesh grid for pixel coordinates
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)

    # Back-project to 3D space
    x3D = (x - cx) * depth / fx
    y3D = (y - cy) * depth / fy
    z3D = depth

    # Mask to remove zero depth values
    mask = depth > 0

    # Visualize using matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x3D[mask], y3D[mask], -z3D[mask], c=rgb[mask].reshape(-1, 3) / 255.0, s=0.5)
    plt.show()


rgb_folder = "C:\\Users\\jpedr\\Desktop\\projetos_maneiros_git\\midas_ai\\RGB_imgs"
heatmap_folder = "C:\\Users\\jpedr\\Desktop\\projetos_maneiros_git\\midas_ai\\heatmap_imgs"
rgbd_folder = "C:\\Users\\jpedr\\Desktop\\projetos_maneiros_git\\midas_ai\\RGBD_imgs"

# get_heatmap_folder(folder_path=rgb_folder)
# list_matching = matching_files(rgb_folder, heatmap_folder)
# #print(list_matching)
# create_RGBD_all(list_matching,rgbd_folder)
rgbd_img_path = "C:\\Users\\jpedr\\Desktop\\projetos_maneiros_git\\midas_ai\\RGBD_imgs\\foto_rgbd.png"
visualize_rgbd_as_3d(rgbd_img_path)