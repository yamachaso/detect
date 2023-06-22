from os.path import expanduser

ROOT_PATH = f"{expanduser('~')}/catkin_ws/src/detect"
OUTPUTS_PATH = f"{ROOT_PATH}/outputs"
DATASETS_PATH = f"{ROOT_PATH}/resources/datasets"
CONFIGS_PATH = f"{ROOT_PATH}/resources/configs"
SAMPLES_PATH = f"{ROOT_PATH}/resources/samples"
SEGNET_DATASETS_PATH = f"{ROOT_PATH}/resources/segnet_datasets"


UINT16MAX = 2 ** 16
