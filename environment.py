import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import numpy as np
import cv2
import math


DISPLAY_CAMERA_IMG = False
SECONDS_PER_EPISODE = 10

class env:
    DISPLAY_CAM = DISPLAY_CAMERA_IMG
    STEER = 1.0 # steering amount: [-1,1]
    front_camera = None

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.nissan_micra_bp = self.blueprint_library.find('vehicle.nissan.micra')[0]

    def reset(self):
        self.collision_history = []
        self.actor_list = []
        # spawning the vehicle actor choosing a random spawn point from the map's list of recommended points
        self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.nissan_micra_bp, self.spawn_point)
        self.actor_list.append(self.vehicle)
        # Initializing camera blueprint
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', '640')
        self.camera_bp.set_attribute('image_size_y', '480')
        self.camera_bp.set_attribute('fov', '110') # field of viewself.rgb_cam = self.blueprint_library.find

        # spawning the rgb camera relative to the car, 2 meters forward and 1 meter above
        relative_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        self.rgb_camera = self.world.spawn_actor(self.camera_bp, relative_transform, attach_to=self.vehicle)
        self.actor_list.append(self.rgb_camera)
        self.rgb_camera.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0))
        time.sleep(5)

        collision_sensor = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_sensor, relative_transform, attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.collision_data(event))

        # must return an observation, which is the image of the front facing camera
        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0))

        return self.front_camera


    def collision_data(self, event):
        self.collision_history.append(event)

    def process_img(self, image):
        img = np.array(image.raw_data)
        img_temp = img.reshape((480, 640, 4))
        img_reshaped = img_temp[:, :, :3] # getting rgb values instead of rgba

        if self.DISPLAY_CAM:
            cv2.imshow("", img_reshaped) # displaying camera input
            cv2.waitKey(1)
        
        self.front_camera = img_reshaped

        # return img_reshaped/255.0 # normalize image data to 0-1 rather than 0-255 for input to neural networks 

    def step(self, action):
        if action == 0: # left
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER))
        elif action == 1: # straight
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 2: # right
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER))

        velocity = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))

        done = False
        reward = 0
        if len(self.collision_history) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1
        
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        
        # return obs, reward, done, info
        return self.front_camera, reward, done, None
