import random

import numpy as np
from glm import vec2
import cv2
from object_collector.envs.AgentState import AgentState
from shapely.geometry import Polygon, Point, LineString
from rtree import Index

random.seed(0)


class MapState:
    objectives: Index = Index()
    objective_dict: {int: Polygon} = {}

    def __init__(self, map_size, n_cells_per_unit=20, n_objectives=11):
        self.n_objectives = n_objectives
        height, width = map_size
        self.map_shape = (height, width, 3)
        self.map_state = np.ones(shape=self.map_shape, dtype=np.uint8) * 255
        self.cell_width = width // n_cells_per_unit
        self.cell_height = height // n_cells_per_unit
        self.cell_width_ndc = 2 / n_cells_per_unit
        self.cell_height_ndc = 2 / n_cells_per_unit
        self.width = width
        self.height = height

        self.generate_objectives()

    def reset(self):
        self.objectives = Index(interleaved=True)
        self.master_tree = Index()
        self.generate_objectives()
        self.setup_map()

    def all_objectives_collected(self):
        return self.objectives.get_size() == 0

    def try_collect(self, agent: AgentState):
        collected = 0

        agent_pos = agent.position
        previous_agent_pos = agent.previous_pos

        line_of_movement = LineString([previous_agent_pos, agent_pos])

        _, indices, bounds = self.objectives.leaves()[0]
        for i in indices:
            obj = self.objective_dict.get(i)
            if obj.intersects(line_of_movement):
                collected += 1
                self.objectives.delete(i, obj.bounds)

        if collected > 0:
            self.setup_map()

        return collected

    def setup_map(self):
        self.map_state = np.ones(shape=self.map_shape, dtype=np.uint8) * 255

        _, indices, bounds = self.objectives.leaves()[0]
        for j in indices:
            obj = self.objective_dict.get(j)
            tl, tr, br, bl = [Point(*self.from_ndc(vec2(*p))) for p in list(obj.exterior.coords)[:-1]]

            cv2.rectangle(img=self.map_state, pt1=(int(tl.x), int(tl.y)), pt2=(int(br.x), int(br.y)), color=(0, 255, 0),
                          thickness=-1)

    def get_current_state(self, agent: AgentState):
        points = [self.from_ndc(vec2(p[0], p[1])) for p in list(agent.get_body().exterior.coords)[:-1]]
        points = np.array(points, dtype=np.int32)
        points = points.reshape((-1, 1, 2))

        map_copy = np.array(self.map_state, copy=True)
        map_copy = cv2.fillPoly(map_copy, [points], (255, 0, 0))

        return map_copy.astype(dtype=np.float32)

    def generate_object(self, offset=None):
        if offset is None:
            offset_x = random.uniform(-0.9, 0.9)
            offset_y = random.uniform(-0.9, 0.9)
            offset = vec2(offset_x, offset_y)
        center_row, center_col = (offset.x, offset.y)
        top_left = (center_row - self.cell_height_ndc / 2, center_col - self.cell_width_ndc / 2)
        top_right = (center_row - self.cell_height_ndc / 2, center_col + self.cell_width_ndc / 2)
        bottom_left = (center_row + self.cell_height_ndc / 2, center_col - self.cell_width_ndc / 2)
        bottom_right = (center_row + self.cell_height_ndc / 2, center_col + self.cell_width_ndc / 2)
        obj = Polygon([*map(Point, [top_left, top_right, bottom_right, bottom_left])])

        return obj

    def to_ndc(self, pos: vec2):

        ndc_x = pos.x / self.width * 2 - 1
        ndc_y = pos.y / self.height * 2 - 1

        return vec2(ndc_x, ndc_y)

    def from_ndc(self, pos: vec2):
        pixelX = round((pos.x + 1.0) * 0.5 * self.width, 0)
        pixelY = round((1.0 - pos.y) * 0.5 * self.height, 0)

        return vec2(pixelX, pixelY)

    def generate_objectives(self):
        for i in range(self.n_objectives):
            obs = self.generate_object()
            self.objectives.insert(i, obs.bounds)
            self.objective_dict[i] = obs