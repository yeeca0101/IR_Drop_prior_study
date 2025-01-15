import os
import gzip
import numpy as np
import re
import heapq
from collections import defaultdict
import multiprocessing
import logging
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm

# Define via layer maps as constants
VIA_LAYERS_NANGATE45 = {
    ('m1', 'm4'): 'm14',
    ('m4', 'm7'): 'm47',
    ('m7', 'm8'): 'm78',
    ('m8', 'm9'): 'm89'
}

VIA_LAYERS_ASAP7 = {
    ('m2', 'm5'): 'm25',
    ('m5', 'm6'): 'm56',
    ('m6', 'm7'): 'm67',
    ('m7', 'm8'): 'm78'
}

VIA_LAYERS_ALL = {
    ('m1', 'm2'): 'm12',
    ('m1', 'm3'): 'm13',
    ('m1', 'm4'): 'm14',
    ('m1', 'm5'): 'm15',
    ('m1', 'm6'): 'm16',
    ('m1', 'm7'): 'm17',
    ('m1', 'm8'): 'm18',
    ('m1', 'm9'): 'm19',
    ('m2', 'm3'): 'm23',
    ('m2', 'm4'): 'm24',
    ('m2', 'm5'): 'm25',
    ('m2', 'm6'): 'm26',
    ('m2', 'm7'): 'm27',
    ('m2', 'm8'): 'm28',
    ('m2', 'm9'): 'm29',
    ('m3', 'm4'): 'm34',
    ('m3', 'm5'): 'm35',
    ('m3', 'm6'): 'm36',
    ('m3', 'm7'): 'm37',
    ('m3', 'm8'): 'm38',
    ('m3', 'm9'): 'm39',
    ('m4', 'm5'): 'm45',
    ('m4', 'm6'): 'm46',
    ('m4', 'm7'): 'm47',
    ('m4', 'm8'): 'm48',
    ('m4', 'm9'): 'm49',
    ('m5', 'm6'): 'm56',
    ('m5', 'm7'): 'm57',
    ('m5', 'm8'): 'm58',
    ('m5', 'm9'): 'm59',
    ('m6', 'm7'): 'm67',
    ('m6', 'm8'): 'm68',
    ('m6', 'm9'): 'm69',
    ('m7', 'm8'): 'm78',
    ('m7', 'm9'): 'm79',
    ('m8', 'm9'): 'm89'
}

class EffectiveDistanceMapGenerator:
    def __init__(self, sp_file_path: str, output_dir: str, voltage_source_layer: str = 'auto',
                 target_layers: Optional[List[str]] = None, resolution_factor: int = 2000,
                 via_layer_map_preset: Optional[Dict[Tuple[str, str], str]] = None):
        self.sp_file_path = sp_file_path
        self.output_dir = output_dir
        self.voltage_source_layer = voltage_source_layer.lower() if voltage_source_layer != 'auto' else None
        self.via_layer_map = via_layer_map_preset if via_layer_map_preset else VIA_LAYERS_ALL
        self.target_layers = target_layers if target_layers else list(set([layer for pair in self.via_layer_map.keys() for layer in pair]))
        self.graph = defaultdict(dict)
        self.voltage_sources = []
        self.resolution_factor = resolution_factor
        
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Compile regex patterns
        self.resistance_pattern = re.compile(r"R\d+\s+n1_(\w+)_(\d+)_(\d+)\s+n1_(\w+)_(\d+)_(\d+)\s+([\d.]+)", re.IGNORECASE)

    def parse_spice_file(self) -> None:
        open_func = gzip.open if self.sp_file_path.endswith('.gz') else open
        mode = 'rt' if self.sp_file_path.endswith('.gz') else 'r'
        max_layer = None

        try:
            with open_func(self.sp_file_path, mode) as file:
                for line in file:
                    line = line.replace('met', 'm')
                    match = self.resistance_pattern.match(line)
                    if match:
                        layer1, layer2 = match.group(1).lower(), match.group(4).lower()
                        x1, y1, x2, y2 = map(int, match.group(2, 3, 5, 6))
                        resistance = float(match.group(7))

                        if self.target_layers and (layer1 not in self.target_layers or layer2 not in self.target_layers):
                            continue

                        node1, node2 = (layer1, x1, y1), (layer2, x2, y2)
                        self.graph[node1][node2] = resistance
                        self.graph[node2][node1] = resistance

                        if self.voltage_source_layer is None:
                            max_layer = max(max_layer or layer1, layer1, layer2)

            if self.voltage_source_layer is None and max_layer is not None:
                self.voltage_source_layer = max_layer

            self.voltage_sources = [node for node in self.graph if node[0] == self.voltage_source_layer]
        except IOError as e:
            logging.error(f"Error reading file: {e}")
            raise

    def dijkstra(self, start_node: Tuple[str, int, int]) -> Dict[Tuple[str, int, int], float]:
        distances = {node: float('inf') for node in self.graph}
        distances[start_node] = 0
        priority_queue = [(0, start_node)]
        visited = set()

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            if current_node in visited:
                continue
            visited.add(current_node)

            for neighbor, weight in self.graph[current_node].items():
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

        return distances

    def process_voltage_source(self, source_node: Tuple[str, int, int]) -> Dict[Tuple[str, int, int], float]:
        return self.dijkstra(source_node)

    def generate_effective_distance_map(self):
        self.parse_spice_file()
        effective_distances = defaultdict(lambda: float('inf'))

        for source_node in tqdm(self.voltage_sources, desc="Processing voltage sources", unit="source"):
            distances = self.dijkstra(source_node)
            for node, distance in distances.items():
                effective_distances[node] = min(effective_distances[node], distance)

        layer_resolution = self.get_layer_resolution()
        distance_matrices = {layer: np.full(layer_resolution, np.inf) for layer in self.target_layers}

        for (layer, x, y), distance in tqdm(effective_distances.items(), desc="Creating distance matrices", unit="node"):
            if layer in distance_matrices:
                matrix_x, matrix_y = x // self.resolution_factor, y // self.resolution_factor
                if 0 <= matrix_x < layer_resolution[1] and 0 <= matrix_y < layer_resolution[0]:
                    distance_matrices[layer][matrix_y, matrix_x] = distance

        for layer, matrix in tqdm(distance_matrices.items(), desc="Saving distance maps", unit="layer"):
            output_file_path = os.path.join(self.output_dir, f'{layer}_effective_distance_map.npy')
            np.save(output_file_path, matrix)
            print(f"Saved effective distance map for layer {layer} at {output_file_path}")

    def get_layer_resolution(self) -> Tuple[int, int]:
        open_func = gzip.open if self.sp_file_path.endswith('.gz') else open
        mode = 'rt' if self.sp_file_path.endswith('.gz') else 'r'
        max_x, max_y = 0, 0

        try:
            with open_func(self.sp_file_path, mode) as file:
                for line in file:
                    line = line.replace('met', 'm')
                    match = self.resistance_pattern.match(line)
                    if match:
                        x1, y1, x2, y2 = map(int, match.group(2, 3, 5, 6))
                        max_x = max(max_x, x1, x2)
                        max_y = max(max_y, y1, y2)
        except IOError as e:
            logging.error(f"Error reading file for resolution: {e}")
            raise

        return (max_y // self.resolution_factor + 1, max_x // self.resolution_factor + 1)

if __name__ == '__main__':
    def test_generate_effective_distance():
        sp_file_path = '/data/BeGAN-circuit-benchmarks/nangate45/set1/data/BeGAN_0599.sp.gz'
        output_dir = './effective_distance_output'
        voltage_source_layer = 'auto'
        target_layers = None
        via_layer_map_preset = VIA_LAYERS_NANGATE45

        generator = EffectiveDistanceMapGenerator(sp_file_path, output_dir, voltage_source_layer, target_layers, via_layer_map_preset=via_layer_map_preset)
        generator.generate_effective_distance_map()

    test_generate_effective_distance()
