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
        self.spectator = self.world.get_spectator()
        self.town_map = self.world.get_map()

        self.generate_unique_waypoints()
        self.current_waypoint = ... 

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

        # get closest waypoint
        self.set_closest_waypoint()

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

    # def reward(self, )

    def step(self, actions):
        # apply actions
        self.vehicle.apply_control(carla.VehicleControl(throttle=actions[0], steer=actions[1], brake=actions[2]))
        
        # compute reward
        velocity = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))

        done = False
        r_collision = 0
        if len(self.collision_history) != 0:
            done = True
            r_collision = -200
        
        # restricting to 48.3 kmh = 30 mph
        r_speed = kmh if kmh <= 48.3 else 100-kmh
        
        theta = self.get_car_and_lane_angle()
        speed_along_lane = velocity * math.cos(theta)
        v_perpendicular_to_orbit = velocity * math.sin(theta)

        r_out = 0
        distance_to_wp = self.get_car_deviation_from_waypoint()
        if distance_to_wp > 2:
            r_out = -50
        
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        
        # return obs, reward, done, info
        return self.front_camera, reward, done, None
    
    """
    Run only at beginning of training
    """
    def generate_unique_waypoints(self):
        print("Generating unique waypoints...")

        all_waypoints = self.town_map.generate_waypoints(0.3) # 0.3 meters between waypoints

        # find unique waypoints
        self.unique_waypoints = []
        for wp in all_waypoints:
            if len(self.unique_waypoints) == 0:
                self.unique_waypoints.append(wp)
            else:
                # getting waypoints that are not located at the same position
                found = False
                for uwp in self.unique_waypoints:
                    # if waypoints are within 0.1 meters of x and y positions and within 20 degrees of yaw, they are the same
                    if abs(uwp.transform.location.x - wp.transform.location.x) < 0.1 \
                            and abs(uwp.transform.location.y - wp.transform.location.y) < 0.1 \
                            and abs(uwp.transform.rotation.yaw - uwp.rotation.yaw) < 20:
                        found = True
                        break
                
                if not found:
                    self.unique_waypoints.append(wp)
        
        # draw all waypoints for 60 seconds
        for wp in self.unique_waypoints:
            self.world.debug.draw_string(wp.transform.location, '^', draw_shadow=False, color = carla.Color(r=0, g=0, b=255), life_time=60.0, persistent_lines=True)
        
        # move spectator to top down view
        spectator_pos = carla.Transform(carla.Location(x=0, y=30, z=200), carla.Rotation(pitch=-90, yaw=-90))
        self.spectator.set_transform(spectator_pos)

    
    def set_closest_waypoint(self):
        my_waypoint = self.vehicle.get_transform().location
        self.current_waypoint = min(self.unique_waypoints, key=lambda wp: my_waypoint.distance(wp.transform.location))

        # draw the waypoint
        self.world.debug.draw_string(self.current_waypoint.transform.location, '^', draw_shadow=False, color=carla.Color(r=0, g=0, b=255), life_time=60.0, persistent_lines=True)

    def get_car_and_lane_angle(self):
        vehicle_transform = self.vehicle.get_transform()
        theta = 360 - ((vehicle_transform.rotation.yaw - self.current_waypoint.transform.rotation.yaw) % 360)

        return theta
    
    def get_car_deviation_from_waypoint(self):
        vehicle_transform = self.vehicle.get_transform()
        distance_to_wp = self.current_waypoint.transform.location.distance(vehicle_transform.location)

        return distance_to_wp
    


