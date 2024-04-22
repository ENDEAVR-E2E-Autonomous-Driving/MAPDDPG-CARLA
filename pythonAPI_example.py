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

def process_img(image):
    img = np.array(image.raw_data)
    img_temp = img.reshape((480, 640, 4))
    img_reshaped = img_temp[:, :, :3] # getting rgb values instead of rgba
    cv2.imshow("", img_reshaped) # displaying camera input
    cv2.imwrite('testImage.jpg', img_reshaped)
    cv2.waitKey(1)

    return img_reshaped/255.0 # normalize image data to 0-1 rather than 0-255 for input to neural networks    

actor_list = []

try:
    # connect to the local simulator by creating a client object
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0) # time limit for all networking operations

    # retrieve the world
    world = client.get_world()

    """
    Initializing blueprints, blueprints contain necessary information about a new actor (such as a car or pedestrian)
    Blueprints allow the user to change the color of the car, establish sensors and how many channels, etc.
    """
    # retrieving all available blueprints
    blueprint_library = world.get_blueprint_library()

    # initializing vehicle blueprint
    nissan_micra_bp = blueprint_library.find('vehicle.nissan.micra')

    # spawning the vehicle actor choosing a random spawn point from the map's list of recommended points
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle_actor = world.spawn_actor(nissan_micra_bp, spawn_point)

    # test controlling the car
    vehicle_actor.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(vehicle_actor)

    # Initializing camera blueprint
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_bp.set_attribute('fov', '90') # field of view

    # spawning the rgb camera relative to the car, 2 meters forward and 1 meter above
    relative_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
    rgb_camera = world.spawn_actor(camera_bp, relative_transform, attach_to=vehicle_actor)
    rgb_camera.listen(lambda data: process_img(data))
    actor_list.append(rgb_camera)
    
    time.sleep(10)

finally:
    # Destroying all actors
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up")

