import rospy
from nodes.service_servers.unit_convert_server import SERVICE_NAMES_AND_TYPES


class UnitConvertClient:
    def __init__(self):
        for name, type in SERVICE_NAMES_AND_TYPES.items():
            rospy.wait_for_service(name)
            setattr(self, name, rospy.ServiceProxy(name, type))
