import random
from enum import Enum
from math import cos, sin
from shapely import affinity
from glm import vec2, rotate, atan2, vec3, mat4, vec4, radians, degrees
from shapely.geometry import Polygon, Point
import math


class Actions(Enum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    MOVE_ORIENTATION = 2

    def __eq__(self, other):
        return self.value == other


class AgentState:
    position: vec2
    orientation: float  # Angle
    name: str
    forward_velocity: float
    turn_velocity: float

    def __init__(self, start_position: vec2, orientation: float, name: str,
                 body_width=0.08,
                 body_height=0.15,
                 forward_velocity=0.01,
                 turn_velocity=10.0,
                 use_random_position=False, ):
        self.start_position = start_position
        self.start_orientation = radians(orientation)
        self.position = start_position
        self.orientation = radians(orientation)
        self.body_width = body_width
        self.body_height = body_height

        self.name = name
        self.forward_velocity = forward_velocity
        self.turn_velocity = radians(turn_velocity)
        self.use_random_position = use_random_position

        self.current_target = None
        self.avoiding_obstacle = False

        if self.use_random_position:
            self.set_rand_pos()
        self.previous_pos = self.position

    def set_rand_pos(self):
        x = random.uniform(-0.8, 0.8)
        y = random.uniform(-0.8, 0.8)
        self.position = vec2(x, y)

    def get_body(self, position=None) -> Polygon:
        if position is None:
            position = self.position
        # Create arrow
        tl = Point(-self.body_height / 2, self.body_width / 2)
        mr = Point(self.body_width / 2, 0)
        bl = Point(-self.body_height / 2, -self.body_width / 2)
        arrow = Polygon([*map(Point, [tl, mr, bl])])
        arrow = affinity.rotate(arrow, angle=self.orientation, use_radians=True)
        arrow = affinity.translate(arrow, xoff=position.x, yoff=position.y)

        return arrow

    def get_orientation_vector(self):
        return vec2(cos(self.orientation), sin(self.orientation))

    def do_action(self, action: Actions):
        if action == Actions.MOVE_ORIENTATION:
            self.move_orientation()

        elif action == Actions.TURN_LEFT:
            self.turn_left()

        elif action == Actions.TURN_RIGHT:
            self.turn_right()

    def move_orientation(self):
        new_pos = self.create_new_position()

        self.previous_pos = self.position

        self.position = new_pos

    def create_new_position(self):
        return self.position + self.get_orientation_vector() * self.forward_velocity

    def turn_left(self):
        orientation_vector = self.get_orientation_vector()
        rotation_matrix = rotate(mat4(1.0), -self.turn_velocity, vec3(0, 0, 1))
        orientation_vector = vec2(vec4(orientation_vector, 1, 1) * rotation_matrix)
        self.orientation = atan2(orientation_vector.y, orientation_vector.x)

    def turn_right(self):
        orientation_vector = self.get_orientation_vector()
        rotation_matrix = rotate(mat4(1.0), self.turn_velocity, vec3(0, 0, 1))
        orientation_vector = vec2(vec4(orientation_vector, 1, 1) * rotation_matrix)
        self.orientation = atan2(orientation_vector.y, orientation_vector.x)

    def reset_state(self):
        if self.use_random_position:
            self.set_rand_pos()
        else:
            self.position = self.start_position
        self.previous_pos = self.position
        self.orientation = 0

    def check_bounds(self):
        p1 = Point(-1, -1)
        p2 = Point(-1, 1)
        p3 = Point(1, -1)
        p4 = Point(1, 1)
        poly_1 = Polygon([p2, p4, p3, p1])
        new_pos = self.create_new_position()
        body = self.get_body(new_pos)
        if poly_1.contains(body):
            return 1
        return -1


if __name__ == "__main__":
    agent = AgentState(start_position=vec2(-2, 0.9), orientation=2 * math.pi, name="e")
    p1 = Point(-1, -1)
    p2 = Point(-1, 1)
    p3 = Point(1, -1)
    p4 = Point(1, 1)
    poly_1 = Polygon([p2, p4, p3, p1])
