import bpy
import numpy as np
from math import *
from mathutils import * 
import csv
import pickle

VIZ_FOLDER = # Folder containing motion.csv and task_dict.pkl to visualize
'''
Blender script for visualizing Cassie motion. Provide 
VIZ_FOLDER containing the absolute filepath
to the directory containing the motion.csv (one col per
timestamp with all DOF values) and the task_dict.pkl
containing information on the goal location and obstacles.
'''

#--- Blender Utils ----------------------------------------------------------------------+

class BlenderUtils:
    """docstring for BlenderUtils"""
    def __init__(self,ob,sce):
        self.ob = ob
        self.sce = sce
        
    def get_unit_scale(self):
        # get unit_scale
        unit_settings = self.sce.unit_settings
        if unit_settings.system in {"METRIC", "IMPERIAL"}:
            # The units used in modelling are for display only. 
            # Behind the scenes everything is in meters.
            scale = unit_settings.scale_length
        else:
            # No unit system in use
            scale = 1
        return scale 

    def quaternionRotation(self,armatureName, boneName):
        bone        = bpy.data.objects[armatureName].pose.bones[boneName].bone
        bone_ml     = bone.matrix_local
        bone_pose   = bpy.data.objects[armatureName].pose.bones[boneName]
        bone_pose_m = bone_pose.matrix
        
        if bone.parent:
            parent        = bone.parent
            parent_ml     = parent.matrix_local
            parent_pose   = bone_pose.parent
            parent_pose_m = parent_pose.matrix

            object_diff = parent_ml.inverted() * bone_ml
            pose_diff   = parent_pose_m.inverted() * bone_pose_m
            local_diff  = object_diff.inverted() * pose_diff
        else:
            local_diff = bone_ml.inverted() * bone_pose_m
        
        return local_diff.to_quaternion()

    def xAxisRotation(self,armatureName, boneName):
        q = self.quaternionRotation(armatureName, boneName)
        return atan2(2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[1]*q[1]+q[2]*q[2]))

    def yAxisRotation(self,armatureName, boneName):
        q = self.quaternionRotation(armatureName, boneName)
        return asin(2*(q[0]*q[2]-q[3]*q[1]))

    def zAxisRotation(self,armatureName, boneName):
        q = self.quaternionRotation(armatureName, boneName)
        return atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]*q[2]+q[3]*q[3]))


    # get local rotation matrix in euler
    def get_loc_rot_mat(self,name):
        pbone = self.ob.pose.bones[name]
        local_mat = self.ob.convert_space(pbone,pbone.matrix,from_space='POSE',to_space='LOCAL')
        local_mat = local_mat.to_euler()
        return local_mat

    # get global position of bone head
    def get_global_pos_head(self,name):
        pbone = self.ob.pose.bones[name]
        pos = (self.ob.location + pbone.head) * self.get_unit_scale() #global xyz position
        #print("ob location:{}, bone head:{}".format(self.ob.location,pbone.head))
        return pos

    # get global position of bone head
    def get_global_pos_tail(self,name):
        pbone = self.ob.pose.bones[name]
        pos = (self.ob.location + pbone.tail) * self.get_unit_scale() #global xyz position
        #print("ob location:{}, bone tail:{}".format(self.ob.location,pbone.tail))
        return pos

    # set angle on blender 
    def set_angle_x(self,name,radian):
        pbone = self.ob.pose.bones[name]
        pbone.rotation_euler[0] = radian

    def set_angle_y(self,name,radian):
        pbone = self.ob.pose.bones[name]
        pbone.rotation_euler[1] = radian

    def set_angle_z(self,name,radian):
        pbone = self.ob.pose.bones[name]
        pbone.rotation_euler[2] = radian
    
    def set_pos_x(self,name,pos):
        pbone = self.ob.pose.bones[name]
        pbone.location[0] = pos
    
    def set_pos_y(self,name,pos):
        pbone = self.ob.pose.bones[name]
        pbone.location[1] = pos
    
    def set_pos_z(self,name,pos):
        pbone = self.ob.pose.bones[name]
        pbone.location[2] = pos

    def set_pos_3d(self,name,pos_vec):
        vec = Vector((pos_vec[0], pos_vec[1], pos_vec[2]))
        pbone = self.ob.pose.bones[name]
        mat_rot = Matrix.Identity(4)
        mat_trans = Matrix.Translation(vec)
        mat = mat_trans @ mat_rot     
        pbone.matrix = self.ob.convert_space(pose_bone=pbone, 
                                             matrix=mat, 
                                             from_space='WORLD', 
                                             to_space='POSE')
        # set scale
        pbone.scale = Vector((1.0,1.0,1.0))                                       

    def set_pose(self,name,pos,rot):
        pos_vec = Vector((pos[0], pos[1], pos[2]))
        eul = Euler((rot[0], rot[1], rot[2]), 'XYZ')
        pbone = self.ob.pose.bones[name]
        mat_rot = eul.to_matrix().to_4x4()
        mat_trans = Matrix.Translation(pos_vec)
        mat = mat_trans * mat_rot
        pbone.matrix = self.ob.convert_space(pose_bone=pbone, 
                                             matrix=mat, 
                                             from_space='WORLD', 
                                             to_space='POSE')
        # set scale
        pbone.scale = Vector((1.0,1.0,1.0))  

    def mute_copy_rotation(self,names):
        for name in names:
            self.ob.pose.bones[name].constraints["Copy Rotation"].mute = True
        self.sce.update()

    def open_copy_rotation(self,names):
        for name in names:
            self.ob.pose.bones[name].constraints["Copy Rotation"].mute = False
        self.sce.update()

    # update joints and get fk results
    def update_joints(self,q):
        hip = q[0]
        knee = q[1]
        knee_to_shin = 0 
        ankle = radians(13) - knee 

        # update joints on blender/ order is important!!!
        if left_right == 0:
            names = ['hip_flexion_left','knee_joint_left','knee_to_shin_left','ankle_joint_left']
        else:
            names = ['hip_flexion_right','knee_joint_right','knee_to_shin_right','ankle_joint_right']

        self.set_angle_z(names[0],hip)
        self.set_angle_x(names[1],knee)
        self.set_angle_x(names[2],knee_to_shin)
        self.set_angle_x(names[3],ankle)
        self.sce.update()
        # get pos of the tail of ankle
        fk = self.get_global_pos_tail(names[3])
        return fk

    # get keyframes of object list
    def check_keyframe(self, obj, curr_frame):
        is_keyframe = False
        anim = obj.animation_data
        if anim is not None and anim.action is not None:
            for fcu in anim.action.fcurves:
                for keyframe in fcu.keyframe_points:
                    x, y = keyframe.co
                    #print("x:{}, curr_frame:{}".format(x,curr_frame))
                    if abs(x - curr_frame) < 0.01:
                        is_keyframe = True
                        break
        
        return is_keyframe

    def copy_keyframe(self, obj, from_i, to_j):
        anim = obj.animation_data
        if anim is not None and anim.action is not None:
            for fcurve in anim.action.fcurves:
                fcurve.keyframe_points.insert(to_j, fcurve.keyframe_points[from_i].co.y)


def read_lines(file):
    with open(file, 'rU') as data:
        reader = csv.reader(data)
        for row in reader:
            yield [ float(i) for i in row ]

#--- RUN ----------------------------------------------------------------------+

if __name__ == '__main__':
    global left_right #left = 0 right = 1 
    global butils
    global tg

    motion = list(read_lines(VIZ_FOLDER+"motion.csv"))
    task_dict = pickle.load(open(VIZ_FOLDER+"task_dict.pkl", 'rb'))
        
    time = motion[0]

    ob = bpy.context.scene.objects['cassie_bone']
    ob.animation_data_clear()
    eye_ob = bpy.context.scene.objects['eye_bone']
    eye_latticeL = bpy.context.scene.objects['EyeLattice.L']
    eye_latticeR = bpy.context.scene.objects['EyeLattice.R']
    eye_ob_count = 0
    eye_latticeL_count = 0
    eye_latticeR_count = 0
    
    sce = bpy.context.scene
    for i in range(2):
        pos = list(task_dict['cones'][i]) + [0.03]
        bpy.context.scene.objects[f'obs{i+1}'].location = pos
    bpy.context.scene.objects['target'].location = list(task_dict['target']) + [0.03]

    names = ['hip_abduction_left','hip_rotation_left','hip_flexion_left','knee_joint_left','knee_to_shin_left','ankle_joint_left','toe_joint_left',
            'hip_abduction_right','hip_rotation_right','hip_flexion_right','knee_joint_right','knee_to_shin_right','ankle_joint_right','toe_joint_right']
    
    ori_start = 0
    f_start = 0
    last_frame = f_start
    T = 1/24.0
    last_time = time[0]
    for i in range(len(time)):
        #curr_frame = round((time[i] - last_time)/T) + last_frame
        curr_frame = round((time[i] - time[0])/T)
        sce.frame_set(curr_frame)
        butils = BlenderUtils(ob,sce)
        
        pos_vec = [motion[1][i],motion[2][i],motion[3][i]]
        #pos_vec = [motion[1][i],0,motion[3][i]]
        butils.set_pos_3d('pelvis_target', pos_vec=pos_vec) # only this function is global
        butils.set_angle_x('pelvis_target',motion[4][i])
        butils.set_angle_z('pelvis_target',-1*motion[5][i])
        butils.set_angle_y('pelvis_target',motion[6][i])
        
        butils.set_angle_y(names[0],-1*motion[7][i])
        butils.set_angle_y(names[1],-1*motion[8][i])
        butils.set_angle_z(names[2],-1*motion[9][i])
        butils.set_angle_x(names[3],motion[10][i])
        butils.set_angle_x(names[4],motion[11][i])
        butils.set_angle_x(names[5],motion[12][i])
        butils.set_angle_y(names[6],motion[13][i])
        
        butils.set_angle_y(names[7],-1*motion[14][i])   
        butils.set_angle_y(names[8],-1*motion[15][i])
        butils.set_angle_z(names[9],-1*motion[16][i])
        butils.set_angle_x(names[10],motion[17][i])
        butils.set_angle_x(names[11],motion[18][i])
        butils.set_angle_x(names[12],motion[19][i])
        butils.set_angle_y(names[13],motion[20][i])
        
        #butils.sce.update() 
        
        bpy.ops.pose.select_all(action='TOGGLE')
        bpy.ops.pose.select_all(action='TOGGLE')
        bpy.ops.anim.keyframe_insert_menu(type='BUILTIN_KSI_LocRot')               
        
        
        last_time = time[i]
        last_frame = curr_frame
        print("Progress: {} in total {}s".format(time[i],time[-1]))