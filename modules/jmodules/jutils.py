import numpy as np
import json
# import scipy
from math import atan2, asin
from datetime import datetime
from pytz import timezone
from scipy.spatial.transform import Rotation as Rot


class SynJSON:
    def __init__(self, jfile):
        self.jfile = jfile
        with open(jfile) as f:
            data = json.load(f)
            self.data = data
    def landmarks(self):
        ## return the landmarks as an np array
        landmarks_positions = []
        for ldmk in self.data['landmarks']['ibug68']:
            landmarks_position = ldmk['screen_space_pos']
            landmarks_positions.append(landmarks_position)
        landmarks_positions = np.array(landmarks_positions)

        return landmarks_positions
    
    def face_center(self):
        """    return the face center as np array (x, y)  """
        landmarks = self.landmarks()
        return landmarks.mean(axis=0)

    def pitch_yaw(self, radian=True):
        """ get the pitch and yaw angles in radian as a 2 element numpy array
        was in world coordinates """
        yaw = self.data['gaze_values']['eye_right']['horizontal_angle']
        pitch = self.data['gaze_values']['eye_right']['vertical_angle']
        pitchyaw = np.array([pitch, yaw])*np.pi/180.0
        # print("in function:", pitch, yaw)
        # print(pitchyaw)
        if radian:
            return float(pitchyaw[0]), float(pitchyaw[1])
        else:
            return pitch, yaw

    def cam_gaze_vectors(self):
        """ get the gaze vector in camera frame, transformed form world frame"""
        world2cam = self.world2cam()
        gvectors = self.gaze_vectors()  #left and right
        r = Rot.from_matrix(world2cam[:3,:3])
        cam_gaze_vectors = r.apply(gvectors)
        assert np.isclose(1.0 ,np.linalg.norm(cam_gaze_vectors[1]))
        assert np.isclose(1.0 ,np.linalg.norm(cam_gaze_vectors[0]))
        return cam_gaze_vectors   #left and right
        

    def pitch_yaw_roll(self, radian=True):
        """ get the pitch. yaw, roll angles of of gaze from gaze vector """
        std_vecs=np.array([[0,0,1], [0,0,1]])  #left and right standard vectors, looking stright from cam
        cam_gaze_vectors = self.cam_gaze_vectors()
        # print(f'cam_gaze_vectors={cam_gaze_vectors}')
        # print(f'std_vecs = {std_vecs}')
        # print(type(Rot))
        r = Rot.align_vectors(cam_gaze_vectors, std_vecs)
        yaw, pitch, roll = r[0].as_euler("YXZ", degrees=True) * [1, -1, 1]
        # print(f'done as euler')
        if radian:
            return pitch*np.pi/180, yaw*np.pi/180, roll*np.pi/180
        else:
            return pitch, yaw, roll
    
    def world2cam(self):

        mat =self.data['camera']["transform_world2cam"]["mat_4x4"]
        mat = np.array(mat)
        return mat

    def cam2world(self):
        mat =self.data['camera']["transform_cam2world"]["mat_4x4"]
        mat = np.array(mat)
        return mat

    def gaze_vectors(self):  #return the 3d gaze vectors, left and right
        vecr = self.data['gaze_values']['eye_right']['gaze_vector']
        vec_right = np.array(vecr)
        vecl = self.data['gaze_values']['eye_left']['gaze_vector']
        vec_left = np.array(vecl)
        return vec_left, vec_right

    def d3tod2(self, vec, radian=True):
        x, y, z = vec
        theta = asin(x)
        phi = atan2(y, z)
        if radian:
            return phi, theta  #pitch and yaw
        else:
            return phi*180*np.pi, theta*180/np.pi
        

    def pitchyaw2d(self, radian=True, average=True):
        cam_gaze_vectors = self.cam_gaze_vectors()
        pitch_left, yaw_left = self.d3tod2(cam_gaze_vectors[0])
        pitch_right, yaw_right = self.d3tod2(cam_gaze_vectors[1])
        results = np.array([[pitch_left, yaw_left], [pitch_right, yaw_right]])
        if not radian:
            results = results*180/np.pi 
        if average:
            return  results.mean(axis=0)
        else:
            return results  #pitch and then yaw
        

def stamp_angles(ifile, jfile, resize_to=(224, 224)):
    """  given a synthetic json file and image file, 
    stamped image with pitch and yaw
    """    
    js = SJ(jfile) 
    pitch, yaw = js.pitch_yaw(radius=False)
    im = Image.open(ifile)
    im = im.resize(resize_to)
    text = f'pitch:{pitch:<5.1f} yaw:{yaw:<5.1f}'
#     font = ImageFont.load("arial.pil")
    font = ImageFont.truetype("Arial Unicode.ttf", 20)
    draw = ImageDraw.Draw(im)
    draw.text((10, 10),text,(255,250,255), font=font)
    return im

def get_now():
    now = datetime.utcnow()
    now = now.astimezone(timezone('US/Pacific'))
    date_format='%m/%d/%Y %H:%M:%S'
    now = now.strftime(date_format)  
    return now    