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
import pygame
import json


DISPLAY_CAMERA_IMG = False
SECONDS_PER_EPISODE = 200

class environment:
    DISPLAY_CAM = DISPLAY_CAMERA_IMG
    STEER = 1.0 # steering amount: [-1,1]
    front_camera = None

    def __init__(self, draw_waypoints=True, display_img=True, lookahead_steps=5):
        # initialize pygame screen
        # pygame.init()
        # self.display_size = (640, 480)
        # self.screen = pygame.display.set_mode(self.display_size)
        # pygame.display.set_caption("RGB Camera View")

        self.image_saved = False

        # intialize CARLA components
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.spectator = self.world.get_spectator()
        self.town_map = self.world.get_map()

        self.generate_unique_waypoints(draw_waypoints)
        self.current_waypoint = None # updates later in reset method
        self.lookahead_steps = lookahead_steps # num steps to lookahead and collect waypoints
        self.future_waypoint = None
        self.stuck_steps = 0


        self.front_camera = None
        self.DISPLAY_CAM = display_img

        self.blueprint_library = self.world.get_blueprint_library()
        self.bus_bp = self.blueprint_library.find('vehicle.mitsubishi.fusorosa')

    def destroy_all_actors(self):
        """
        Safely destroys all the actors spawned in the current episode.
        """
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
                print(f"{actor} destroyed", end=", ")
                time.sleep(0.5)
        self.actor_list.clear()
        print()

    # setting up sensors for separate sensor state input
    def setup_other_sensors(self):
        # GPS sensor
        gps_bp = self.world.get_blueprint_library().find('sensor.other.gnss')
        self.gps_sensor = self.world.spawn_actor(gps_bp, carla.Transform(), attach_to=self.vehicle)
        self.gps_sensor.listen(lambda data: self.process_gps_data(data))

        # IMU sensor
        imu_bp = self.world.get_blueprint_library().find('sensor.other.imu')
        self.imu_sensor = self.world.spawn_actor(imu_bp, carla.Transform(), attach_to=self.vehicle)
        self.imu_sensor.listen(lambda data: self.process_imu_data(data))

        self.actor_list.append(self.gps_sensor)
        self.actor_list.append(self.imu_sensor)

    def process_gps_data(self, data):
        self.gps_data = np.array([data.latitude, data.longitude, data.altitude])

    def process_imu_data(self, data):
        self.imu_data = np.array([
            data.accelerometer.x, data.accelerometer.y, data.accelerometer.z,
            data.gyroscope.x, data.gyroscope.y, data.gyroscope.z,
            data.compass
        ])

    def try_spawn_vehicle(self, attempts=10):
        """
        Try to spawn the vehicle at multiple spawn points to avoid collision.
        """
        for _ in range(attempts):
            spawn_point = random.choice(self.world.get_map().get_spawn_points())
            vehicle = self.world.try_spawn_actor(self.bus_bp, spawn_point)
            if vehicle:
                self.actor_list.append(vehicle)
                print(f"Spawned vehicle at {spawn_point}.")
                return vehicle
            print("Spawn attempt failed, trying another location.")
            time.sleep(0.1)  # Small delay before next attempt
        return None

    def reset(self):
        self.collision_history = []
        self.actor_list = []

        # attempting to spawn vehicle
        self.vehicle = self.try_spawn_vehicle()
        if not self.vehicle:
            raise RuntimeError("Failed to spawn vehicle after multiple attempts.")

        self.actor_list.append(self.vehicle)

        # Initializing camera blueprint
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', '640')
        self.camera_bp.set_attribute('image_size_y', '480')
        self.camera_bp.set_attribute('fov', '110') # field of viewself.rgb_cam = self.blueprint_library.find

        # spawning the rgb camera relative to the car, 2 meters forward and 1 meter above
        relative_transform = carla.Transform(carla.Location(x=4, z=3.0))
        self.rgb_camera = self.world.spawn_actor(self.camera_bp, relative_transform, attach_to=self.vehicle)
        self.actor_list.append(self.rgb_camera)
        self.rgb_camera.listen(lambda data: self.process_img(data))
        print("Camera spawned.")

        collision_sensor = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_sensor, relative_transform, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_data(event))
        print("Collision sensor spawned.")

        # get closest waypoint
        self.set_closest_waypoint()

        # set up sensors
        self.setup_other_sensors()

        # must return an observation, which is the image of the front facing camera
        print("Waiting for image observation to load...")
        while self.front_camera is None:
            time.sleep(0.01)

        print("Done. Training begins.")
        self.episode_start = time.time()

        return self.front_camera


    def collision_data(self, event):
        self.collision_history.append(event)

    def process_img(self, image):
        img = np.array(image.raw_data)
        img_temp = img.reshape((480, 640, 4))
        img_reshaped = img_temp[:, :, :3] # getting rgb channels instead of rgba

        # resize image for less computation
        resized_img = cv2.resize(img_reshaped, (320, 160), interpolation=cv2.INTER_AREA)

        # self.front_camera = img_reshaped
        self.front_camera = resized_img

        if self.DISPLAY_CAM:
            # create pygame surface from the raw data
            img_surface = pygame.surfarray.make_surface(np.transpose(img_reshaped, (1,0,2)))
            self.screen.blit(img_surface, (0,0))
            pygame.display.flip()

         # Save the first image
        if not self.image_saved:
            cv2.imwrite('first_image.png', cv2.cvtColor(img_reshaped, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR as OpenCV expects
            self.image_saved = True
            print("First image saved as 'first_image.png'.")

        # return img_reshaped/255.0 # normalize image data to 0-1 rather than 0-255 for input to neural networks 

    
    def step(self, actions):
        # apply actions
        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=float(actions[0]), 
                steer=float(actions[1]), 
                brake=float(actions[2])
            )
        )

        self.set_closest_waypoint()  # Update waypoints based on the vehicle's new position
        
        # compute Reward
        velocity_vector = self.vehicle.get_velocity()
        speed = math.sqrt(velocity_vector.x**2 + velocity_vector.y**2 + velocity_vector.z**2)
        # kmh = 3.6 * velocity

        # get vehicle heading and waypoint heading in radians
        vehicle_transform = self.vehicle.get_transform()
        vehicle_yaw_rad = math.radians(vehicle_transform.rotation.yaw)
        waypoint_yaw_rad = math.radians(self.current_waypoint.transform.rotation.yaw)

        # calculate the angle between the vehicle's heading and the waypoint's direction
        theta = vehicle_yaw_rad - waypoint_yaw_rad
        theta = (theta + math.pi) % (2 * math.pi) - math.pi  # normalize angle to [-pi, pi]

        speed_along_lane = speed * math.cos(theta)
        speed_perpendicular_to_lane = speed * math.sin(theta)
        deviation_from_lane = self.get_car_deviation_from_waypoint()

        # reward calculations
        alpha, beta, eta = 0.33, 0.33, 0.33  # adjust these weights as necessary
        # restricting to 13.5 m/s = 30 mph
        r_speed = speed if speed <= 13.5 else 27-speed
        r_speed /= 13.5 # normalize to between [0,1]
        r_center = speed_along_lane - abs(speed_perpendicular_to_lane) - deviation_from_lane - (speed_along_lane * deviation_from_lane)
        r_out = 0 if deviation_from_lane < 2.0 else -50

        reward = alpha * r_speed + beta * r_center + eta * r_out 

        collision_occurred = False
        angle_threshold = math.radians(30)  # 45 degrees threshold
        angle_deviation = math.atan2(math.sin(vehicle_yaw_rad - waypoint_yaw_rad), 
                                 math.cos(vehicle_yaw_rad - waypoint_yaw_rad))
        episode_length = time.time() - self.episode_start
        stuck_threshold = 5  # number of consecutive low-speed steps to consider stuck

        # Check for low velocity indicating possible stuck condition
        if speed < 0.001:  # Threshold speed to determine if stuck
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0  # Reset if speed is above the threshold

        done = False
        if self.stuck_steps >= stuck_threshold:
            print("Episode ended: Vehicle is stuck.")
            done = True
            self.stuck_steps = 0
        if episode_length >= SECONDS_PER_EPISODE:
            print("Episode ended: Maximum amount of time per episode reached.")
            done = True
        if angle_deviation > angle_threshold:
            print(f"Episode ended: Vehicle deviated more than {angle_threshold} radians away from the center lane.")
            done = True
        if deviation_from_lane > 10:
            print(f"Episode ended: Vehicle deviated more than {deviation_from_lane} meters away from the center lane.")
            done = True
        if len(self.collision_history) > 0:
            print(f"Episode ended: Vehicle encountered a collision")
            done = True
            collision_occurred = True


        # normalize image to [0,1]
        normalized_camera = self.front_camera / 255.0

        # return observation, reward, done, info
        return normalized_camera, reward, done, {
            'episode_length': episode_length, 
            'lane_deviation': deviation_from_lane, 
            'collision_occurred': collision_occurred
        }


    """
    Run only at beginning of training
    """
    def generate_unique_waypoints(self, draw_waypoints):
        print("Generating unique waypoints...")
        spectator_pos = carla.Transform(carla.Location(x=0, y=30, z=200), carla.Rotation(pitch=-90, yaw=-90))
        self.spectator.set_transform(spectator_pos)
        all_waypoints = self.town_map.generate_waypoints(0.3)
        self.unique_waypoints = []
        i = 1
        num_wp = len(all_waypoints)
        for wp in all_waypoints:
            print(f"{i}/{num_wp} waypoints")
            if not self.unique_waypoints or not any(wp.transform.location.distance(uwp.transform.location) < 0.1 for uwp in self.unique_waypoints):
                self.unique_waypoints.append(wp)
                if draw_waypoints:
                    self.world.debug.draw_string(wp.transform.location + carla.Location(z=1), 'O', color=carla.Color(r=255, g=0, b=0), life_time=120.0)

            i += 1

        print("Unique waypoints are generated and drawn.")
    # def generate_unique_waypoints(self, draw_waypoints):
    #     print("Generating unique waypoints...")

    #     all_waypoints = self.town_map.generate_waypoints(0.3) # 0.3 meters between waypoints

    #     # find unique waypoints
    #     self.unique_waypoints = []
    #     for wp in all_waypoints:
    #         if len(self.unique_waypoints) == 0:
    #             self.unique_waypoints.append(wp)
    #         else:
    #             # getting waypoints that are not located at the same position
    #             found = False
    #             for uwp in self.unique_waypoints:
    #                 # if waypoints are within 0.1 meters of x and y positions and within 20 degrees of yaw, they are the same
    #                 if abs(uwp.transform.location.x - wp.transform.location.x) < 0.1 \
    #                         and abs(uwp.transform.location.y - wp.transform.location.y) < 0.1 \
    #                         and abs(uwp.transform.rotation.yaw - wp.transform.rotation.yaw) < 20:
    #                     found = True
    #                     break
                
    #             if not found:
    #                 self.unique_waypoints.append(wp)
        
    #     # draw all waypoints for 60 seconds
    #     if draw_waypoints:
    #         for wp in self.unique_waypoints:
    #             hover_location = carla.Location(
    #                 x=wp.transform.location.x,
    #                 y=wp.transform.location.y,
    #                 z=wp.transform.location.z + 5 # add 5 meters of hover offset so that agent doesn't learn wp as part of env
    #             )
    #             self.world.debug.draw_string(hover_location, '^', draw_shadow=False, color = carla.Color(r=0, g=0, b=255), life_time=60.0, persistent_lines=True)
        
    #         # move spectator to top down view
    #         spectator_pos = carla.Transform(carla.Location(x=0, y=30, z=200), carla.Rotation(pitch=-90, yaw=-90))
    #         self.spectator.set_transform(spectator_pos)

    #     # with open('unique_waypoints.json', 'w') as f:
    #     #     json.dump({'unique_waypoints': self.unique_waypoints}, f)
        

    #     print("Unique waypoints are generated and drawn.")
    
    """Run only at the beginning of each episode"""
    def set_closest_waypoint(self):
        current_location = self.vehicle.get_transform().location
        # Calculate distances to all waypoints
        distances = [current_location.distance(wp.transform.location) for wp in self.unique_waypoints]
        # Find the closest waypoint
        min_distance_index = np.argmin(distances)
        self.current_waypoint = self.unique_waypoints[min_distance_index]

        # Lookahead mechanism
        lookahead_index = min(min_distance_index + self.lookahead_steps, len(self.unique_waypoints) - 1)
        self.future_waypoint = self.unique_waypoints[lookahead_index]

        # Draw the current and future waypoints
        self.world.debug.draw_string(self.current_waypoint.transform.location + carla.Location(z=1), 'X', color=carla.Color(r=0, g=255, b=0), life_time=10.0)
        self.world.debug.draw_string(self.future_waypoint.transform.location + carla.Location(z=1), 'X', color=carla.Color(r=0, g=0, b=255), life_time=10.0)
    # def set_closest_waypoint(self):
    #     my_waypoint = self.vehicle.get_transform().location
    #     self.current_waypoint = min(self.unique_waypoints, key=lambda wp: my_waypoint.distance(wp.transform.location))

    #     # draw the waypoint
    #     hover_location = carla.Location(
    #         x=self.current_waypoint.transform.location.x,
    #         y=self.current_waypoint.transform.location.y,
    #         z=self.current_waypoint.transform.location.z + 5 # add 5 meters of hover offset so that agent doesn't learn wp as part of env
    #     )
    #     self.world.debug.draw_string(hover_location, '^', draw_shadow=False, color=carla.Color(r=0, g=0, b=255), life_time=10.0, persistent_lines=True)

    def get_car_and_lane_angle(self):
        vehicle_transform = self.vehicle.get_transform()
        theta = 360 - ((vehicle_transform.rotation.yaw - self.current_waypoint.transform.rotation.yaw) % 360)

        return theta
    
    def get_car_deviation_from_waypoint(self):
        vehicle_transform = self.vehicle.get_transform()
        distance_to_wp = self.current_waypoint.transform.location.distance(vehicle_transform.location)

        return distance_to_wp
    


