from gym.envs.registration import register

register(
    id="GymEnviroments/camera-v1",
    entry_point="GymEnviroments.v1_camera_env:CameraEnv",
)
