import numpy as np

#Matrix multiplication
def matmul(mats):
    out = mats[0]

    for i in range(1, len(mats)):
        out = np.matmul(out, mats[i])
    
    return out

#Returns the rotation matrix in X
def rotationX(theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([
        [1,0,0,0],
        [0,cos_theta,-sin_theta,0],
        [0,sin_theta,cos_theta,0],
        [0,0,0,1]], dtype = np.float32)

#Returns the rotation matrix in Y
def rotationY(theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([
        [cos_theta,0,sin_theta,0],
        [0,1,0,0],
        [-sin_theta,0,cos_theta,0],
        [0,0,0,1]], dtype = np.float32)

#Returns the rotation matrix in Y
def rotationZ(theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([
        [cos_theta,-sin_theta,0,0],
        [sin_theta,cos_theta,0,0],
        [0,0,1,0],
        [0,0,0,1]], dtype = np.float32)

#Returns the translation matrix
def translate(tx, ty, tz):
    return np.array([
        [1,0,0,tx],
        [0,1,0,ty],
        [0,0,1,tz],
        [0,0,0,1]], dtype = np.float32)

def random_rotation_matrix():
    angx = np.random.randint(-180,180)
    angy = np.random.randint(-180,180)
    angz = np.random.randint(-180,180)

    rx = rotationX(np.radians(angx))
    ry = rotationY(np.radians(angy))
    rz = rotationZ(np.radians(angz))

    return matmul([rx, ry, rz])

def rotationAxis(theta, point, normal):
    axis = normal / np.linalg.norm(normal)
    a,b,c = axis
    h = np.sqrt(a**2 + c**2)

    T = translate(-point[0], -point[1], -point[2])
    Tinv = translate(point[0], point[1], point[2])

    Ry = np.array([
        [a/h, 0, c/h, 0],
        [0,1,0,0],
        [-c/h, 0, a/h, 0],
        [0,0,0,1]], dtype=np.float32)
    
    Ryinv = np.array([
        [a/h, 0, -c/h, 0],
        [0,1,0,0],
        [c/h, 0, a/h, 0],
        [0,0,0,1]], dtype=np.float32)
    
    Rz = np.array([
        [h, b, 0, 0],
        [-b, h, 0, 0],
        [0,0,1,0],
        [0,0,0,1]], dtype=np.float32)
    
    Rzinv = np.array([
        [h, -b, 0, 0],
        [b, h, 0, 0],
        [0,0,1,0],
        [0,0,0,1]], dtype=np.float32)
    
    Rx = rotationX(theta)

    return matmul([Tinv,Ryinv,Rzinv,Rx,Rz,Ry,T])