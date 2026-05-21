#!/usr/bin/env python3
import argparse
import os
import polyscope as ps
import numpy as np
from symmetries import *
import lzma
import open3d as o3d

parser = argparse.ArgumentParser()
#parser.add_argument('--path', type=str, default='', help='Path of dataset')
parser.add_argument('--file', type=str, default=1, help='Number of shape to plot')
parser.add_argument('--sym_file', type=str, default=1, help='Number of shape to plot')
parser.add_argument("--normalize", action='store_true', help='Center and normalize object and symmetry planes to unit sphere')

#Add boolean argument
parser.add_argument("--no_sym", action='store_true', help='Use this option if you only want to plot the shape without symmetries')

opt = parser.parse_args()
name = opt.file
#opt.file = os.path.join('/data/Software/BT3D_rot2/symmetry_axis_features/pot', name + '.npz')
extension = opt.file.split('.')[-1]
print(extension)
if extension == 'xz' or extension == 'npz' or extension == 'npy' or extension == 'xyz':
    type_ext = 'points'
elif extension in ('obj', 'ply', 'off'):
    type_ext = 'mesh'
else:
    print('Extension not supported')
    exit()

sym = opt.sym_file
prefix = opt.file.split('/')[-1].split('.')[0]

if type_ext== 'points' and extension == 'xyz':
    points = np.loadtxt(opt.file)
    points = points[:,:3]

if type_ext == 'points' and extension == 'xz':
    with lzma.open(opt.file, 'rb') as fhandle:
        points = np.loadtxt(fhandle)


    print(points.shape)

elif type_ext == 'points' and extension == 'npz':
    points= np.load(opt.file)
    #Cheking if the file contains 'points' or 'features'
    if 'points' in points:
        points = points['points']
    else:
        points = points[:,:3]
    
elif type_ext == 'points' and extension == 'npy':
    points = np.load(opt.file)
    print(np.mean(points, axis=0))
    points2 = np.load(opt.file.replace('original','selected'))

elif type_ext == 'mesh':
    import trimesh
    mesh = trimesh.load(opt.file, force='mesh')
    vertices  = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.faces)
else:
    print('Extension not supported')
    exit()

if opt.normalize:
    if type_ext == 'mesh':
        centroid = np.mean(vertices, axis=0)
        vertices -= centroid
        scale = np.max(np.linalg.norm(vertices, axis=1))
        vertices /= scale
    elif type_ext == 'points':
        centroid = np.mean(points, axis=0)
        points -= centroid
        scale = np.max(np.linalg.norm(points, axis=1))
        points /= scale

if not opt.no_sym:
    symmetry_list = []
    #Read symmetries
    #with open(os.path.join(opt.path, sym)) as f:
    with open(sym) as f:
        num_symmetries = int(f.readline().strip())
        print(num_symmetries)
    
        for i in range(num_symmetries):
            L = f.readline().strip().split()
            if L[0]=="plane":
                L = L[1:]
                L = [float(x) for x in L]
                print(L[0:3])
                print(L[3:6])
                pt = np.array(L[3:6])
                if opt.normalize:
                    pt -= centroid
                    pt /= scale
                if len(L) > 6:
                    if L[6] > 0.5:
                        symmetry_list.append(SymmetryPlane(point=pt, normal=np.array(L[0:3])))
                else:
                    symmetry_list.append(SymmetryPlane(point=pt, normal=np.array(L[0:3])))
                
            elif L[0]=="axis":
                L = L[1:]
                L = [float(x) for x in L]
                symmetry_list.append(SymmetryAxis(point=np.array(L[3:6]), normal=np.array(L[0:3]), angle=float(L[6])))

        
ps.init()
ps.set_up_dir("y_up")
ps.set_background_color((1, 1, 1))
ps.look_at((2.,2.,2.),(0.,0.,0.))
ps.set_ground_plane_mode("shadow_only")



if type_ext == 'points':
    #center and normalize points
    #points -= np.mean(points, axis=0)
    #points /= np.max(np.linalg.norm(points, axis=1))
    ps.register_point_cloud("PC", points, color=(97/255,149/255,243/255))
    #ps.register_point_cloud("PC2", points2)
    auxPoints = np.concatenate((points.T, np.ones((1,points.shape[0]))), axis=0)
elif type_ext == 'mesh':
    #p3 = np.load('/data/Software/e3sym/sample.npy')
    #p4 = np.load('/data/Software/e3sym/planepoints.npy')
   
    #vertices -= np.mean(vertices, axis=0)
    #vertices /= np.max(np.linalg.norm(vertices, axis=1))    
    ps.register_surface_mesh("mesh", vertices, triangles)
    #ps.register_point_cloud("PC2", p3)
    #ps.register_point_cloud("PC3", p4)
    

if not opt.no_sym:
    for i, sym in enumerate(symmetry_list):
        if isinstance(sym, SymmetryPlane):
            mesh = ps.register_surface_mesh("sym_plane_"+str(i), sym.coords, sym.trianglesBase)
            mesh.set_transparency(0.8)
        elif isinstance(sym, SymmetryAxis):
            name = "sym_axis_"+str(i)
            ps.register_curve_network(name, np.array([-0.7*sym.normal+sym.point,0.7*sym.normal+sym.point]), np.array([[0,1]]), radius=0.01, color=(239/255,8/255,8/255))

        #    if sym.angle == np.inf:
        #        continue
        #    print('Angle: ', np.degrees(sym.angle))
        #    rotPoints = rotationAxis(sym.angle, sym.point, sym.normal)@auxPoints
            #ps.register_point_cloud("Rot_PC_"+str(i), rotPoints[:3,:].T)

#ps.set_screenshot_extension('.jpg')
#ps.screenshot('images/'+prefix+'*.jpg')       
ps.show()