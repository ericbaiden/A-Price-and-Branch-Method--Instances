"""
Instance generator for LEO Constellation Cross-Domain Collaborative Scheduling.

Based on intermodalTransportationNetworksCG instance_generator.py architecture.
Builds time-expanded graph with nodes and arcs for column generation.

Reference: README Section 5.3-5.5 core data structures and instance building
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field

from .tle_parser import TLESatellite, TLEParser
from .visibility_calculator import VisibilityCalculator, VisibilityWindow


@dataclass
class TimeWindow:
    """Time window for scheduling horizon."""
    start: float  # minutes from epoch
    end: float    # minutes from epoch

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __hash__(self):
        return hash((self.start, self.end))

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SatelliteParams:
    """Satellite physical and energy parameters.

    Reference: README Section 4.1.2 Real Satellite Parameters
    """
    sat_id: str
    sat_type: str  # 'RS' or 'CS'

    # Storage (GB)
    storage_capacity: float = 8.0  # GB

    # Energy parameters (Wh)
    battery_capacity: float = 3000.0  # Wh
    battery_min: float = 500.0       # Wh (safety margin)

    # Power consumption (W)
    power_img: float = 50.0     # Imaging
    power_tx: float = 25.0       # ISL transmit
    power_rcv: float = 18.0      # ISL receive
    power_std: float = 5.0       # Standby
    power_charge: float = 60.0   # Solar charging

    # Link parameters
    num_inter_links: int = 2     # ISL links (RS: 2, CS: 4)
    num_intra_links: int = 4     # ISL links for CS
    downlink_bandwidth: float = 1000.0  # Mbps

    # Orbital parameters (from TLE)
    altitude: float = 0.0        # km
    inclination: float = 0.0     # degrees

    # Computed energy per slot (200s slot)
    @property
    def energy_img(self) -> float:
        """Energy consumed for one imaging operation."""
        return self.power_img * (200 / 3600)  # Wh

    @property
    def energy_tx(self) -> float:
        """Energy consumed per GB transmitted."""
        return self.power_tx * (200 / 3600) / self.downlink_bandwidth * 1000  # Wh/GB

    @property
    def energy_rcv(self) -> float:
        """Energy consumed per GB received."""
        return self.power_rcv * (200 / 3600) / self.downlink_bandwidth * 1000

    @property
    def energy_std(self) -> float:
        """Standby energy per slot."""
        return self.power_std * (200 / 3600)

    @property
    def energy_charge_per_slot(self) -> float:
        """Energy charged per slot."""
        return self.power_charge * (200 / 3600)


@dataclass
class Task:
    """Imaging task request.

    Reference: README Section 2.3 decision variables (λ_{t,p}, z_t)
    """
    task_id: str
    target_lat: float
    target_lon: float
    data_volume: float      # GB
    priority: float         # weight (π_t)
    release_slot: int       # earliest slot to image
    deadline_slot: int      # latest slot to image
    duration: float = 10.0  # imaging duration, seconds

    def __eq__(self, other):
        return self.task_id == other.task_id

    def __hash__(self):
        return hash(self.task_id)

    @property
    def window_slots(self) -> range:
        """Range of slots within the task's imaging window."""
        return range(self.release_slot, self.deadline_slot + 1)


@dataclass
class FlatNode:
    """Static network node (satellite at a given position).

    Reference: intermodalTransportationNetworksCG FlatNode
    """
    satellite: str
    mode: str  # 'RS' or 'CS'
    coordinate: Tuple[float, float, float]  # (lat, lon, alt)

    def __str__(self):
        return f'{self.satellite}, {self.mode}, {self.coordinate}'

    def __eq__(self, other):
        return self.satellite == other.satellite and self.mode == other.mode

    def __hash__(self):
        return hash((self.satellite, self.mode))


@dataclass
class FlatArc:
    """Static network arc (link between satellites)."""
    start: FlatNode
    end: FlatNode

    def __hash__(self):
        return hash((self.start.satellite, self.end.satellite,
                     self.start.mode, self.end.mode))


class FlatNetwork:
    """Static flat network of satellite nodes and links.

    Reference: intermodalTransportationNetworksCG FlatNetwork
    """
    def __init__(self):
        self.nodes: Set[FlatNode] = set()
        self.arcs: Set[FlatArc] = set()

    def add_node(self, node: FlatNode):
        self.nodes.add(node)

    def add_arc(self, arc: FlatArc):
        self.arcs.add(arc)

    def __eq__(self, other):
        return self.nodes == other.nodes and self.arcs == other.arcs

    def __hash__(self):
        return hash((tuple(sorted(self.nodes)), tuple(sorted(self.arcs))))


@dataclass
class TempNode:
    """Temporal network node (satellite state at a time slot).

    Reference: README Section 5.3.1 TempNode (adapted from intermodalTransportationNetworksCG)

    Node types:
    - 'state': satellite state node (s, k)
    - 'imaging': task imaging entry node (t, k)
    - 'ground': ground station node g
    """
    __slots__ = ('satellite', 'time_slot', 'node_type', 'route', 'coordinate', 'hash')

    satellite: str       # satellite ID
    time_slot: int       # time slot k
    node_type: str       # 'state', 'imaging', 'ground'
    route: str           # path identifier
    coordinate: Tuple[float, float, float]

    def __init__(self, satellite: str, time_slot: int, node_type: str,
                 route: str, coordinate: Tuple[float, float, float]):
        self.satellite = satellite
        self.time_slot = time_slot
        self.node_type = node_type
        self.route = route
        self.coordinate = coordinate
        self.hash = hash((self.satellite, self.time_slot, self.node_type, self.route))

    def __str__(self):
        return f'{self.satellite}, {self.time_slot}, {self.node_type}, {self.route}'

    def __eq__(self, other):
        return (self.satellite == other.satellite and
                self.time_slot == other.time_slot and
                self.node_type == other.node_type and
                self.route == other.route)

    def __hash__(self):
        return self.hash


@dataclass
class TempArc:
    """Temporal network arc (with energy consumption).

    Reference: README Section 5.3.2 TempArc (adapted from intermodalTransportationNetworksCG)

    Arc types:
    - 'imaging_arc': task → satellite (consumes E_img)
    - 'storage_arc': (s,k) → (s,k+1) (consumes E_std)
    - 'isl_arc': (s,k) → (s',k) (consumes E_tx + E_rcv)
    - 'downlink_arc': (s,k) → ground (consumes E_tx)
    - 'source_arc': task source → first node
    - 'sink_arc': last node → ground station
    """
    start: TempNode
    end: TempNode
    arc_type: str
    cost: float = 0.0
    capacity: Optional[float] = None
    energy_consumption: float = 0.0

    def __str__(self):
        return f'({self.start}) --> {self.end}, type={self.arc_type}, cost={self.cost}'

    def __hash__(self):
        return hash((self.start.satellite, self.start.time_slot,
                     self.end.satellite, self.end.time_slot,
                     self.arc_type, self.cost))

    def set_capacity(self, arc_type: str, params: Dict):
        """Set arc capacity based on arc type and satellite params.

        For cross_domain arcs, capacity is pre-calculated based on ISL bandwidth
        and should not be overwritten by this method.
        """
        if arc_type == 'isl_arc':
            self.capacity = params.get('isl_capacity', 25)
        elif arc_type == 'downlink_arc':
            self.capacity = params.get('downlink_bandwidth', 1000.0) * 200.0 / 8000.0
        elif arc_type == 'storage_arc':
            self.capacity = params.get('storage_capacity', 8.0)
        elif arc_type == 'imaging_arc':
            self.capacity = 10  # enough for multiple tasks per slot
        elif arc_type == 'imaging_convergence_arc':
            self.capacity = 1  # one acquisition per RS per slot
        elif arc_type == 'cross_domain_arc':
            # Use pre-calculated capacity if available, otherwise default
            self.capacity = params.get('cross_domain_cap', 10)
        elif arc_type == 'cross_domain_convergence_arc':
            # Use pre-calculated capacity if available, otherwise default
            self.capacity = params.get('cross_domain_conv_cap', 8)
        elif arc_type in ['source_arc', 'sink_arc']:
            self.capacity = 10
        else:
            self.capacity = 9999

    def set_energy(self, arc_type: str, sat_params: SatelliteParams):
        """Set energy consumption based on arc type."""
        if arc_type == 'imaging_arc':
            self.energy_consumption = sat_params.energy_img
        elif arc_type == 'isl_arc':
            self.energy_consumption = sat_params.energy_tx + sat_params.energy_rcv
        elif arc_type == 'downlink_arc':
            self.energy_consumption = sat_params.energy_tx
        elif arc_type == 'storage_arc':
            self.energy_consumption = sat_params.energy_std
        else:
            self.energy_consumption = 0.0


class TempNetwork:
    """Temporal network with nodes and arcs for each time slot.

    Reference: intermodalTransportationNetworksCG TempNetwork
    """
    def __init__(self):
        self.nodes: Set[TempNode] = set()
        self.arcs: Set[TempArc] = set()

    def add_node(self, node: TempNode):
        self.nodes.add(node)

    def add_arc(self, arc: TempArc):
        self.arcs.add(arc)

    def __eq__(self, other):
        return self.nodes == other.nodes and self.arcs == other.arcs

    def __hash__(self):
        return hash((tuple(sorted(self.nodes)), tuple(sorted(self.arcs))))


class Instance:
    """Scheduling instance containing all data for column generation.

    Reference: README Section 5.5 instance building flow
    """

    def __init__(
        self,
        time_window: TimeWindow,
        rs_count: int,
        cs_count: int,
        gs_config: Dict,
        tle_data: List[TLESatellite],
        subset: float = 1.0,
        task_data_volumes: Optional[List[float]] = None
    ):
        """Initialize instance.

        Args:
            time_window: Scheduling time horizon
            rs_count: Number of RS satellites
            cs_count: Number of CS satellites
            gs_config: Ground station configurations
            tle_data: TLE satellite data
            subset: Fraction of tasks to use (for scaling)
            task_data_volumes: Optional per-task data volume in GB, assigned in
                increasing task-ID order. When None (the default, and the only
                behaviour used by every historical result record), every task
                keeps the original unit volume of 1.0 GB.
        """
        self.time_window = time_window
        self.rs_count = rs_count
        self.cs_count = cs_count
        self.gs_config = gs_config
        self.tle_data = tle_data
        self.subset = subset
        self.task_data_volumes = list(task_data_volumes) if task_data_volumes is not None else None

        self.flat_network = FlatNetwork()
        self.temp_network = TempNetwork()

        self.sat_params: Dict[str, SatelliteParams] = {}
        self.tasks: List[Task] = []
        self.sources: Dict[str, TempNode] = {}  # task_id -> source node
        self.sinks: Dict[str, TempNode] = {}    # task_id -> sink node

        self.visibility: Dict = {}  # (rs_id, task_id, slot) -> bool
        self.task_visibility: Dict = {}  # task_id -> {target, sats, all_slots}
        self.heuristic: Dict = {}    # for A* search

        # Compute number of slots
        self.slot_duration = 200 / 60  # 200 seconds = 3.33 minutes
        self.num_slots = int(time_window.duration / self.slot_duration)

        # Hop-based cost constants (跳数代价)
        # 同一时隙内，跳数越多路径越长，存储/等待代价更大
        # 汇聚弧不算跳数，只有容量限制
        self.HOP_COST = {
            'isl_arc': 1.0,                      # ISL跳转代价
            'cross_domain_arc': 1.0,             # 跨域代价
            'downlink_arc': 1.0,                 # 下行代价
            'storage_arc': 10.0,                 # 存储代价高（10单位）
            'wait_arc': 10.0,                    # 等待代价高（10单位）
            'imaging_arc': 1.0,                  # 成像代价
            'imaging_convergence_arc': 0.0,      # 汇聚弧，不算跳数
            'cross_domain_convergence_arc': 0.0, # 汇聚弧，不算跳数
            'source_arc': 0.0,                    # 源弧无代价
            'sink_arc': 0.0,                      # 汇弧无代价
            'vgs_arc': 0.0,                       # VGS弧无代价
        }

    def build_instance(self, seed: int = 42):
        """Build the complete scheduling instance.

        Reference: README Section 5.5.1 instance building flow
        """
        # Step 1: Get satellite network
        self.get_satellite_network()

        # Step 2: Calculate visibility windows (RS-target and inter-satellite)
        self.calculate_visibility()

        # Step 2b: Precompute inter-satellite visibility for ISL
        self.calculate_inter_sat_visibility()

        # Step 2c: Precompute ground-satellite visibility for downlink
        self.calculate_ground_sat_visibility()

        # Step 3: Build temporal network
        self.build_temporal_network()

        # Step 4: Add storage layers (waiting arcs)
        self.add_storage_layers()

        # Step 5: Add imaging arcs (creates imaging arcs from waiting nodes)
        # Note: generate_tasks also creates imaging arcs with visibility checks,
        # so this may create duplicate arcs. The arcs from generate_tasks are the authoritative ones.
        self.add_imaging_arcs()

        # Step 6: Add ISL arcs (only RS→CS based on visibility)
        self.add_isl_arcs()

        # Step 7: Add downlink arcs (based on visibility)
        self.add_downlink_arcs()

        # Step 7b: Add cross-domain arcs RS->Virtual_CS->CS (capacity=1 per slot)
        self.add_cross_domain_arcs()

        # Step 8: Generate tasks
        self.generate_tasks(seed)

        # Step 9: Set arc capacities
        self.set_arc_capacities()

        # Step 10: Calculate incidence matrix
        self.calculate_incidence_matrix()

        # Step 11: Build heuristic
        self.build_heuristic_dist()

        # Step 12: Precompute A* heuristic dictionary (optimization)
        self.get_heuristic_fast()

    def get_satellite_network(self):
        """Build static network of satellite nodes.

        Reference: README Section 5.3 network structure
        """
        # Create satellite parameters
        for sat in self.tle_data:
            sat_type = sat.satellite_type
            params = SatelliteParams(
                sat_id=sat.norad_id,
                sat_type=sat_type,
                altitude=sat.altitude,
                inclination=sat.inclination,
            )

            # Set type-specific defaults (from README Section 4.1.2)
            if sat_type == 'RS':
                params.storage_capacity = 8.0  # GB (Dove)
                params.num_inter_links = 2     # ISL links for RS
                params.power_img = 50.0        # W
            else:  # CS (Starlink)
                params.storage_capacity = 10.0  # GB
                params.num_inter_links = 4      # ISL links for CS

            self.sat_params[sat.norad_id] = params

            # Add flat node
            coord = (0.0, 0.0, sat.altitude)  # Simplified coordinates
            flat_node = FlatNode(sat.norad_id, sat_type, coord)
            self.flat_network.add_node(flat_node)

        # Add ISL arcs between RS and CS satellites (simplified - all can connect)
        for sat1 in self.tle_data:
            for sat2 in self.tle_data:
                if sat1 != sat2:
                    # Create bidirectional ISL
                    node1 = FlatNode(sat1.norad_id, sat1.satellite_type,
                                     (0.0, 0.0, sat1.altitude))
                    node2 = FlatNode(sat2.norad_id, sat2.satellite_type,
                                     (0.0, 0.0, sat2.altitude))
                    self.flat_network.add_arc(FlatArc(node1, node2))

    def calculate_visibility(self):
        """Calculate visibility windows for RS satellites to targets.

        Reference: README Section 5.5.2 visibility window calculation
        """
        calc = VisibilityCalculator()

        # Generate target locations
        targets = []
        lats = np.random.uniform(-60, 60, 20)
        lons = np.random.uniform(-180, 180, 20)
        for lat, lon in zip(lats, lons):
            targets.append((lat, lon))

        # Compute visibility matrix
        self.visibility = calc.build_visibility_matrix(
            [s for s in self.tle_data if s.satellite_type == 'RS'],
            targets,
            (self.time_window.start, self.time_window.end),
            self.slot_duration
        )

    def calculate_inter_sat_visibility(self):
        """Precompute inter-satellite visibility for ISL arcs.

        Computes:
        - RS→CS visibility (data flows from RS to CS)
        - CS↔CS visibility (bidirectional, for multi-hop routing)

        Uses simplified distance-based check for speed.

        Sets self.inter_sat_visibility: Dict[(sat1_id, sat2_id, slot)] -> bool
        """
        from .visibility_calculator import VisibilityCalculator

        calc = VisibilityCalculator()

        rs_sats = [s for s in self.tle_data if s.satellite_type == 'RS']
        cs_sats = [s for s in self.tle_data if s.satellite_type == 'CS']

        self.inter_sat_visibility = {}

        # Precompute positions for all satellites at all slots
        # This avoids repeated SGP4 propagation during arc creation
        sat_positions = {}
        for sat in self.tle_data:
            sat_positions[sat.norad_id] = {}
            for k in range(self.num_slots):
                slot_time = self.time_window.start + k * self.slot_duration
                pos = calc._satellite_position(sat, slot_time)
                sat_positions[sat.norad_id][k] = pos

        max_isl_range = 2500.0  # km

        # Omit false entries: all callers use .get(key, False). Computing
        # each slot's distance matrix in NumPy avoids ~100M Python tuples.
        for k in range(self.num_slots):
            rs_valid = [sat for sat in rs_sats if sat_positions[sat.norad_id][k] is not None]
            cs_valid = [sat for sat in cs_sats if sat_positions[sat.norad_id][k] is not None]
            if not cs_valid:
                continue
            cs_xyz = np.asarray([sat_positions[sat.norad_id][k] for sat in cs_valid])
            if rs_valid:
                rs_xyz = np.asarray([sat_positions[sat.norad_id][k] for sat in rs_valid])
                delta = rs_xyz[:, None, :] - cs_xyz[None, :, :]
                distance = np.sqrt(np.sum(delta * delta, axis=2))
                for i, j in zip(*np.where(distance <= max_isl_range)):
                    self.inter_sat_visibility[rs_valid[i].norad_id, cs_valid[j].norad_id, k] = True
            delta = cs_xyz[:, None, :] - cs_xyz[None, :, :]
            distance = np.sqrt(np.sum(delta * delta, axis=2))
            visible = np.triu(distance <= max_isl_range, k=1)
            for i, j in zip(*np.where(visible)):
                first, second = cs_valid[i].norad_id, cs_valid[j].norad_id
                self.inter_sat_visibility[first, second, k] = True
                self.inter_sat_visibility[second, first, k] = True

    def calculate_ground_sat_visibility(self):
        """Precompute ground-station visibility for downlink arcs.

        Sets self.ground_sat_visibility: Dict[(sat_id, gs_name, slot)] -> bool
        """
        from .visibility_calculator import VisibilityCalculator

        calc = VisibilityCalculator()

        self.ground_sat_visibility = {}

        for sat in self.tle_data:
            for gs_name, gs_data in self.gs_config.items():
                for k in range(self.num_slots):
                    slot_time = self.time_window.start + k * self.slot_duration
                    windows = calc.compute_satellite_ground_visibility(
                        sat,
                        gs_data['lat'],
                        gs_data['lon'],
                        slot_time,
                        slot_time + self.slot_duration,
                        num_points=10
                    )
                    # Visible if there's any window in this slot
                    self.ground_sat_visibility[(sat.norad_id, gs_name, k)] = len(windows) > 0

    def build_temporal_network(self):
        """Build time-expanded network with state nodes.

        Reference: README Section 2.1 Time-Expanded Graph
        """
        # Create virtual + state nodes for each satellite at each time slot
        # Each satellite has 2 nodes per slot: virtual (ingress) + state (actual)
        for sat in self.tle_data:
            for k in range(self.num_slots):
                vtype = 'virtual_rs' if sat.satellite_type == 'RS' else 'virtual_cs'
                # Virtual node (ingress point for incoming data)
                virtual_node = TempNode(
                    satellite=sat.norad_id,
                    time_slot=k,
                    node_type=vtype,
                    route=f'virtual_{sat.norad_id}',
                    coordinate=(0.0, 0.0, sat.altitude)
                )
                self.temp_network.add_node(virtual_node)
                # State node (actual satellite state with storage)
                state_node = TempNode(
                    satellite=sat.norad_id,
                    time_slot=k,
                    node_type='state',
                    route=f'state_{sat.norad_id}',
                    coordinate=(0.0, 0.0, sat.altitude)
                )
                self.temp_network.add_node(state_node)

        # Add ground station nodes
        for gs_name, gs_data in self.gs_config.items():
            for k in range(self.num_slots):
                node = TempNode(
                    satellite=f'gs_{gs_name}',
                    time_slot=k,
                    node_type='ground',
                    route=f'gs_{gs_name}',
                    coordinate=(gs_data['lat'], gs_data['lon'], 0.0)
                )
                self.temp_network.add_node(node)

        # Add virtual ground station node (shared sink for all tasks)
        # This represents the ultimate data delivery point
        self.virtual_gs = TempNode(
            satellite='virtual_gs',
            time_slot=0,
            node_type='virtual_ground',
            route='virtual_gs',
            coordinate=(0.0, 0.0, 0.0)
        )
        self.temp_network.add_node(self.virtual_gs)

    def add_storage_layers(self):
        """Add storage arcs between consecutive time slots.

        Reference: README Section 5.3.2 storage_arc (waiting arc analogy)
        """
        for sat in self.tle_data:
            for k in range(self.num_slots - 1):
                start_node = TempNode(
                    satellite=sat.norad_id,
                    time_slot=k,
                    node_type='state',
                    route=f'state_{sat.norad_id}',
                    coordinate=(0.0, 0.0, sat.altitude)
                )
                end_node = TempNode(
                    satellite=sat.norad_id,
                    time_slot=k + 1,
                    node_type='state',
                    route=f'state_{sat.norad_id}',
                    coordinate=(0.0, 0.0, sat.altitude)
                )

                arc = TempArc(start_node, end_node, 'storage_arc')
                arc.cost = self.HOP_COST['storage_arc']  # Hop-based cost
                arc.set_energy('storage_arc', self.sat_params[sat.norad_id])
                arc.set_capacity('storage_arc', {
                    'storage_capacity': self.sat_params[sat.norad_id].storage_capacity
                })
                self.temp_network.add_arc(arc)

    def add_imaging_arcs(self):
        """Add imaging arcs from task entry nodes to RS satellites.

        Reference: README Section 5.3.2 imaging_arc

        Updated PTEG: Task -> imaging_arc -> Virtual_RS -> imaging_convergence_arc -> RS_State
        """
        # For each task, create imaging entry node at each feasible slot
        # Connect task node to Virtual_RS via imaging_arc
        # Then imaging_convergence_arc from Virtual_RS to RS_State

        task_idx = 0
        for lat_idx, (tlat, tlon) in enumerate([(l, lo) for l, lo in
                                                  zip(np.random.uniform(-60, 60, 20),
                                                      np.random.uniform(-180, 180, 20))]):
            for k in range(self.num_slots):
                # Check if any RS satellite can image this target at slot k
                for sat in self.tle_data:
                    if sat.satellite_type != 'RS':
                        continue

                    # Create task imaging node
                    task_node = TempNode(
                        satellite=f'task_{task_idx}',
                        time_slot=k,
                        node_type='imaging',
                        route=f'task_{task_idx}',
                        coordinate=(tlat, tlon, 0.0)
                    )
                    self.temp_network.add_node(task_node)

                    # Virtual_RS node (ingress for imaging data)
                    virtual_rs_node = TempNode(
                        satellite=sat.norad_id,
                        time_slot=k,
                        node_type='virtual_rs',
                        route=f'virtual_{sat.norad_id}',
                        coordinate=(0.0, 0.0, sat.altitude)
                    )

                    # imaging_arc: task -> Virtual_RS
                    arc = TempArc(task_node, virtual_rs_node, 'imaging_arc')
                    arc.cost = self.HOP_COST['imaging_arc']
                    arc.set_energy('imaging_arc', self.sat_params[sat.norad_id])
                    arc.set_capacity('imaging_arc', {})
                    self.temp_network.add_arc(arc)

                    # imaging_convergence_arc: Virtual_RS -> RS_State
                    rs_state_node = TempNode(
                        satellite=sat.norad_id,
                        time_slot=k,
                        node_type='state',
                        route=f'state_{sat.norad_id}',
                        coordinate=(0.0, 0.0, sat.altitude)
                    )
                    conv_arc = TempArc(virtual_rs_node, rs_state_node, 'imaging_convergence_arc')
                    conv_arc.cost = self.HOP_COST['imaging_convergence_arc']
                    conv_arc.capacity = 1
                    self.temp_network.add_arc(conv_arc)

            task_idx += 1

    def add_isl_arcs(self):
        """Add ISL arcs between CS satellites at same time slot.

        All visible ISL connections are created (no degree constraint).
        Edge capacity = (bandwidth * slot_time / data_per_task) = number of tasks
        that can be transmitted on this ISL per slot.

        Reference: README Section 5.3.2 isl_arc

        ISL capacity = (1000 Mbps * 200s / 8000 Mb) = 25 tasks per ISL per slot
        """
        from .visibility_calculator import VisibilityCalculator
        calc = VisibilityCalculator()

        rs_sats = [s for s in self.tle_data if s.satellite_type == 'RS']
        cs_sats = [s for s in self.tle_data if s.satellite_type == 'CS']

        # ISL capacity: bandwidth * slot / data = 1000 * 200 / 8000 = 25 tasks
        isl_bandwidth = 1000.0  # Mbps
        slot_duration_sec = 200.0  # seconds
        data_per_task = 8000.0  # Mb (1GB)
        isl_capacity = int(isl_bandwidth * slot_duration_sec / data_per_task)

        for k in range(self.num_slots):
            slot_time = self.time_window.start + k * self.slot_duration

            # Track ISL degree per CS per slot (max 4 links per CS)
            cs_isl_degree = {cs_sat.norad_id: 0 for cs_sat in cs_sats}
            MAX_ISL_LINKS = 4  # Each CS connects to at most 4 other CS

            # Collect all visible CS pairs with distances
            visible_pairs = []
            for cs1 in cs_sats:
                for cs2 in cs_sats:
                    if cs1.norad_id >= cs2.norad_id:
                        continue

                    # Check visibility
                    visible = self.inter_sat_visibility.get((cs1.norad_id, cs2.norad_id, k), False)
                    if not visible:
                        continue

                    # Compute distance for delay
                    pos1 = calc._satellite_position(cs1, slot_time)
                    pos2 = calc._satellite_position(cs2, slot_time)

                    if pos1 is None or pos2 is None:
                        continue

                    distance = math.sqrt(
                        (pos1[0] - pos2[0]) ** 2 +
                        (pos1[1] - pos2[1]) ** 2 +
                        (pos1[2] - pos2[2]) ** 2
                    )
                    visible_pairs.append((cs1, cs2, distance))

            # Sort by distance (prefer closer pairs)
            visible_pairs.sort(key=lambda x: x[2])

            # Create ISL arcs with degree constraint
            for cs1, cs2, distance in visible_pairs:
                # Check if both CSs have available degree
                if cs_isl_degree[cs1.norad_id] >= MAX_ISL_LINKS:
                    continue
                if cs_isl_degree[cs2.norad_id] >= MAX_ISL_LINKS:
                    continue

                # Increment degrees
                cs_isl_degree[cs1.norad_id] += 1
                cs_isl_degree[cs2.norad_id] += 1

                node1 = TempNode(
                    satellite=cs1.norad_id,
                    time_slot=k,
                    node_type='state',
                    route=f'state_{cs1.norad_id}',
                    coordinate=(0.0, 0.0, cs1.altitude)
                )
                node2 = TempNode(
                    satellite=cs2.norad_id,
                    time_slot=k,
                    node_type='state',
                    route=f'state_{cs2.norad_id}',
                    coordinate=(0.0, 0.0, cs2.altitude)
                )

                # CS1 -> CS2 with capacity = 25 tasks
                arc1 = TempArc(node1, node2, 'isl_arc')
                arc1.cost = self.HOP_COST['isl_arc']
                arc1.capacity = isl_capacity
                arc1.set_energy('isl_arc', self.sat_params[cs1.norad_id])
                self.temp_network.add_arc(arc1)

                # CS2 -> CS1
                arc2 = TempArc(node2, node1, 'isl_arc')
                arc2.cost = self.HOP_COST['isl_arc']
                arc2.capacity = isl_capacity
                arc2.set_energy('isl_arc', self.sat_params[cs2.norad_id])
                self.temp_network.add_arc(arc2)

    def add_downlink_arcs(self):
        """Add downlink arcs from satellites to ground stations.

        Only creates arcs when satellite is visible from ground station
        based on precomputed visibility.

        Reference: README Section 5.3.2 downlink_arc

        Downlink delay = propagation + transmission
        - Propagation: ~5ms for LEO to ground (altitude ~500km)
        - Transmission: data_size / bandwidth (depends on data volume)
        """
        c = 299792.458  # km/s (speed of light)
        Re = 6371.0  # km

        for k in range(self.num_slots):
            # Build lookup for existing state and ground nodes
            state_nodes = {}  # sat_norad_id -> node
            ground_nodes = {}  # gs_name -> node
            for node in self.temp_network.nodes:
                if node.time_slot == k:
                    if node.node_type == 'state':
                        state_nodes[node.satellite] = node
                    elif node.node_type == 'ground':
                        ground_nodes[node.satellite.replace('gs_', '')] = node

            for sat in self.tle_data:
                for gs_name, gs_data in self.gs_config.items():
                    # Only create arc if satellite is visible from ground station
                    if not self.ground_sat_visibility.get((sat.norad_id, gs_name, k), False):
                        continue

                    # Look up existing nodes instead of creating new ones
                    state_node = state_nodes.get(sat.norad_id)
                    ground_node = ground_nodes.get(gs_name)
                    if state_node is None or ground_node is None:
                        continue

                    # Propagation delay (simplified: 500km altitude)
                    prop_delay = sat.altitude / c / 60.0  # min

                    # Transmission delay: avg 2GB at 1000 Mbps
                    avg_data = 2.0  # GB
                    bandwidth = gs_data.get('bandwidth', 1000.0)  # Mbps
                    tx_delay = avg_data / bandwidth / 60.0  # min

                    arc = TempArc(state_node, ground_node, 'downlink_arc')
                    arc.cost = self.HOP_COST['downlink_arc']  # Hop-based cost
                    arc.set_energy('downlink_arc', self.sat_params[sat.norad_id])
                    arc.set_capacity('downlink_arc', {
                        'downlink_bandwidth': bandwidth
                    })
                    self.temp_network.add_arc(arc)

    def add_cross_domain_arcs(self):
        """Add cross-domain arcs from RS to CS via Virtual_CS nodes.

        PTEG structure (capacity=1 enforces one RS per CS per slot):
          RS_State -> (cross_domain_arc) -> Virtual_CS -> (cross_domain_convergence_arc) -> CS_State

        cross_domain_arc cost = RS-CS propagation delay (distance / speed of light).
        Non-zero costs prevent LP degeneracy by differentiating CS choices in A* pricing.
        """
        from .visibility_calculator import VisibilityCalculator

        rs_sats = [s for s in self.tle_data if s.satellite_type == 'RS']
        cs_sats = [s for s in self.tle_data if s.satellite_type == 'CS']

        calc = VisibilityCalculator()
        c_light = 299792.458  # km/s speed of light

        # Capacity constants for cross_domain arcs
        # ISL bandwidth = 1000 Mbps, slot = 200s, data = 1GB = 8000 Mb
        isl_bandwidth = 1000.0  # Mbps
        slot_duration_sec = 200.0  # seconds (fixed)
        data_per_task = 8000.0  # Mb (1GB)

        for k in range(self.num_slots):
            slot_time = self.time_window.start + k * self.slot_duration

            # Precompute RS/CS positions at this slot (ECI, km)
            rs_pos = {}
            for sat in rs_sats:
                p = calc._satellite_position(sat, slot_time)
                if p is not None:
                    rs_pos[sat.norad_id] = p
            cs_pos = {}
            for sat in cs_sats:
                p = calc._satellite_position(sat, slot_time)
                if p is not None:
                    cs_pos[sat.norad_id] = p

            # Each RS links to nearest CS (max 1), each CS linked by at most 1 RS
            # CS links to other CS via ISL (max 4 links per CS)
            rs_linked_cs = {}  # rs_norad_id -> cs_sat it links to
            cs_linked_by = {cs_sat.norad_id: None for cs_sat in cs_sats}  # cs_norad_id -> rs_norad_id

            # Calculate distances and sort
            rs_cs_distances = []
            for rs_sat in rs_sats:
                rs_p = rs_pos.get(rs_sat.norad_id)
                if rs_p is None:
                    continue
                for cs_sat in cs_sats:
                    cs_p = cs_pos.get(cs_sat.norad_id)
                    if cs_p is None:
                        continue
                    dist = ((rs_p[0]-cs_p[0])**2 + (rs_p[1]-cs_p[1])**2 + (rs_p[2]-cs_p[2])**2)**0.5
                    rs_cs_distances.append((rs_sat, cs_sat, dist))

            # Sort by distance and assign: each RS -> nearest CS, each CS <- at most 1 RS
            # Each RS links to at most 1 CS
            rs_cs_distances.sort(key=lambda x: x[2])
            for rs_sat, cs_sat, dist in rs_cs_distances:
                if cs_linked_by[cs_sat.norad_id] is not None:
                    continue  # CS already linked by another RS
                if rs_sat.norad_id in rs_linked_cs:
                    continue  # RS already linked to a CS
                rs_linked_cs[rs_sat.norad_id] = cs_sat
                cs_linked_by[cs_sat.norad_id] = rs_sat.norad_id
                if len(rs_linked_cs) >= len(rs_pos):
                    break

            # Find existing Virtual_CS and CS_State nodes for this slot
            # (created in build_state_nodes)
            virtual_cs_nodes = {}  # cs_norad_id -> virtual_cs_node
            cs_state_nodes = {}  # cs_norad_id -> cs_state_node
            for node in self.temp_network.nodes:
                if node.time_slot == k and node.node_type == 'virtual_cs':
                    virtual_cs_nodes[node.satellite] = node
                elif node.time_slot == k and node.node_type == 'state' and \
                     not node.route.startswith('state_gs'):
                    # It's a CS state node (not ground station)
                    cs_state_nodes[node.satellite] = node

            # Capacity: ISL_bw * slot / data = 1000 * 200 / 8000 = 25 tasks per slot
            cross_domain_cap = int(isl_bandwidth * slot_duration_sec / data_per_task)
            conv_cap = int(isl_bandwidth * slot_duration_sec / data_per_task)

            # Create RS_State nodes and cross_domain_arcs for RS with links
            for rs_sat in rs_sats:
                rs_state_node = TempNode(
                    satellite=rs_sat.norad_id,
                    time_slot=k,
                    node_type='state',
                    route=f'state_{rs_sat.norad_id}',
                    coordinate=(0.0, 0.0, rs_sat.altitude)
                )

                # Link RS to its nearest CS via cross_domain_arc
                if rs_sat.norad_id in rs_linked_cs:
                    linked_cs = rs_linked_cs[rs_sat.norad_id]
                    virtual_cs_node = virtual_cs_nodes.get(linked_cs.norad_id)
                    if virtual_cs_node:
                        arc = TempArc(rs_state_node, virtual_cs_node, 'cross_domain_arc')
                        arc.cost = self.HOP_COST['cross_domain_arc']
                        arc.capacity = cross_domain_cap
                        self.temp_network.add_arc(arc)

            # Create cross_domain_convergence_arcs: Virtual_CS -> CS_State (capacity=25)
            # Only create for CS that are actually linked by an RS
            for cs_norad_id in virtual_cs_nodes:
                if cs_linked_by[cs_norad_id] is None:
                    continue  # This CS is not linked by any RS
                virtual_cs_node = virtual_cs_nodes[cs_norad_id]
                cs_state_node = cs_state_nodes[cs_norad_id]
                conv_arc = TempArc(virtual_cs_node, cs_state_node, 'cross_domain_convergence_arc')
                conv_arc.cost = self.HOP_COST['cross_domain_convergence_arc']
                conv_arc.capacity = conv_cap
                self.temp_network.add_arc(conv_arc)

    def generate_tasks(self, seed: int = 42):
        """Generate task instances with source/sink arcs.

        For satellite scheduling, the flow is:
        Source -> (source_arc) -> RS satellite at visible slot
               -> (storage/ISL/downlink) -> Ground station
               -> (sink_arc) -> Sink

        CRITICAL: Tasks are generated ONLY for targets that are visible
        to at least one RS satellite, ensuring feasibility.

        Reference: README Section 4.3 task generation model
        """
        rng = np.random.RandomState(seed)

        # Counter of accepted tasks, used to index self.task_data_volumes in
        # increasing task-ID order. Reset on every call so repeated builds with
        # the same seed stay deterministic.
        self._n_accepted_tasks = 0

        # Generate tasks based on subset
        num_tasks = int(200 * self.subset)  # Base: 200 tasks

        # Build ground station nodes by slot
        gs_nodes_by_slot = {}
        for node in self.temp_network.nodes:
            if node.node_type == 'ground':
                if node.time_slot not in gs_nodes_by_slot:
                    gs_nodes_by_slot[node.time_slot] = []
                gs_nodes_by_slot[node.time_slot].append(node)

        # Build RS satellite list
        rs_sats = [s for s in self.tle_data if s.satellite_type == 'RS']

        # Precompute visibility windows for task generation
        # Map: (target_lat, target_lon) -> list of (rs_sat_id, visible_slots)
        calc = VisibilityCalculator()
        visible_targets = {}  # (lat, lon) -> {'sats': {sat_id: [slot1, slot2, ...]}, 'all_slots': set()}

        # Generate candidate targets and compute their visibility
        # Try multiple times to get enough visible targets
        candidate_targets = []
        max_attempts = 5000  # Increased for large scale instances
        for attempt in range(max_attempts):
            if len(candidate_targets) >= num_tasks * 2:  # Get more than needed
                break
            tlat = rng.uniform(-60, 60)
            tlon = rng.uniform(-180, 180)
            key = (round(tlat, 2), round(tlon, 2))  # Round to avoid floating point issues
            if key in candidate_targets:
                continue

            # Check if this target is visible to any RS satellite
            windows = calc.compute_visibility_window(
                rs_sats[0], tlat, tlon,
                self.time_window.start,
                self.time_window.end,
                num_points=50
            )

            if len(windows) > 0:
                # Find all visible slots for this target
                visible_slots = set()
                sat_visibility = {}
                for sat in rs_sats:
                    sat_windows = calc.compute_visibility_window(
                        sat, tlat, tlon,
                        self.time_window.start,
                        self.time_window.end,
                        num_points=50
                    )
                    if len(sat_windows) > 0:
                        slots_for_sat = set()
                        for w in sat_windows:
                            # Convert window times to slot indices
                            start_slot = max(0, int((w.start_time - self.time_window.start) / self.slot_duration))
                            end_slot = min(self.num_slots - 1, int((w.end_time - self.time_window.start) / self.slot_duration))
                            for s in range(start_slot, end_slot + 1):
                                slots_for_sat.add(s)
                                visible_slots.add(s)
                        if slots_for_sat:
                            sat_visibility[sat.norad_id] = sorted(slots_for_sat)

                if visible_slots:
                    visible_targets[key] = {
                        'sats': sat_visibility,
                        'all_slots': sorted(visible_slots)
                    }
                    candidate_targets.append(key)

        # Shuffle candidate targets
        rng.shuffle(candidate_targets)

        task_idx = 0
        for i in range(num_tasks):
            if i >= len(candidate_targets):
                break  # Not enough visible targets

            key = candidate_targets[i]
            tlat, tlon = key

            # Get visibility info for this target
            vis_info = visible_targets[key]
            all_slots = vis_info['all_slots']
            sat_visibility = vis_info['sats']

            if not all_slots:
                continue

            # Set time window based on actual visibility
            # Randomly select a contiguous subset of visible slots as the imaging window
            min_slot = min(all_slots)
            max_slot = max(all_slots)

            # Imaging window: select a contiguous subset
            if len(all_slots) < 3:
                release_slot = min_slot
                deadline_slot = max_slot
            else:
                max_window = min(10, len(all_slots))
                window_size = rng.randint(3, max_window + 1)  # +1 because randint is exclusive on high
                if len(all_slots) <= window_size:
                    release_slot = min_slot
                    deadline_slot = max_slot
                else:
                    # Pick a random contiguous window
                    start_idx = rng.randint(0, len(all_slots) - window_size)
                    release_slot = all_slots[start_idx]
                    deadline_slot = all_slots[start_idx + window_size - 1]

            task = Task(
                task_id=f'task_{task_idx}',
                target_lat=tlat,
                target_lon=tlon,
                data_volume=(self.task_data_volumes[self._n_accepted_tasks]
                             if self.task_data_volumes is not None
                             else 1.0),  # GB; default unit volume unless overridden
                priority=rng.uniform(1, 10),
                release_slot=release_slot,
                deadline_slot=deadline_slot
            )

            # Ensure valid window
            if task.deadline_slot >= task.release_slot and task.deadline_slot < self.num_slots:
                self.tasks.append(task)
                self._n_accepted_tasks += 1

                # Store visibility info for this task (which RS satellites can see it at which slots)
                self.task_visibility[task.task_id] = {
                    'target': (tlat, tlon),
                    'sats': sat_visibility,
                    'all_slots': all_slots
                }

                task_idx += 1

                # Create source and sink nodes
                source = TempNode(
                    satellite=f'source_{i}',
                    time_slot=task.release_slot,
                    node_type='source',
                    route=f'source_{i}',
                    coordinate=(task.target_lat, task.target_lon, 0.0)
                )
                sink = TempNode(
                    satellite=f'sink_{i}',
                    time_slot=task.deadline_slot,
                    node_type='sink',
                    route=f'sink_{i}',
                    coordinate=(task.target_lat, task.target_lon, 0.0)
                )

                self.sources[task.task_id] = source
                self.sinks[task.task_id] = sink

                self.temp_network.add_node(source)
                self.temp_network.add_node(sink)

                # Get RS satellites for this task's visibility window
                rs_sats = [s for s in self.tle_data if s.satellite_type == 'RS']

                # Create task waiting nodes and connect them
                prev_wait_node = None
                for k in range(task.release_slot, task.deadline_slot + 1):
                    wait_node = TempNode(
                        satellite=f'task_{i}',
                        time_slot=k,
                        node_type='waiting',
                        route=f'task_{i}',
                        coordinate=(task.target_lat, task.target_lon, 0.0)
                    )
                    self.temp_network.add_node(wait_node)

                    if prev_wait_node is None:
                        # source -> first waiting node
                        source_arc = TempArc(source, wait_node, 'source_arc')
                        source_arc.cost = 0.0
                        source_arc.capacity = 9999
                        self.temp_network.add_arc(source_arc)
                    else:
                        # wait -> wait (hop-based cost)
                        wait_arc = TempArc(prev_wait_node, wait_node, 'wait_arc')
                        wait_arc.cost = self.HOP_COST['wait_arc']
                        wait_arc.capacity = 9999
                        self.temp_network.add_arc(wait_arc)

                    # Create imaging arcs ONLY to RS satellites that can see this target at slot k
                    # This ensures feasibility: only tasks with visible RS satellites get valid paths
                    for sat in rs_sats:
                        sat_slots = sat_visibility.get(sat.norad_id, [])
                        if k not in sat_slots:
                            continue  # Skip if this RS can't see the target at slot k

                        # imaging_arc: waiting_node -> Virtual_RS
                        virtual_rs_node = TempNode(
                            satellite=sat.norad_id,
                            time_slot=k,
                            node_type='virtual_rs',
                            route=f'virtual_{sat.norad_id}',
                            coordinate=(0.0, 0.0, sat.altitude)
                        )
                        img_arc = TempArc(wait_node, virtual_rs_node, 'imaging_arc')
                        img_arc.cost = self.HOP_COST['imaging_arc']
                        img_arc.set_energy('imaging_arc', self.sat_params[sat.norad_id])
                        img_arc.set_capacity('imaging_arc', {})
                        self.temp_network.add_arc(img_arc)

                        # imaging_convergence_arc: Virtual_RS -> RS_State
                        rs_state_node = TempNode(
                            satellite=sat.norad_id,
                            time_slot=k,
                            node_type='state',
                            route=f'state_{sat.norad_id}',
                            coordinate=(0.0, 0.0, sat.altitude)
                        )
                        conv_arc = TempArc(virtual_rs_node, rs_state_node,
                                           'imaging_convergence_arc')
                        conv_arc.cost = self.HOP_COST['imaging_convergence_arc']
                        conv_arc.capacity = 1
                        self.temp_network.add_arc(conv_arc)

                    prev_wait_node = wait_node

                # Create sink arcs: ground at deadline_slot -> sink at deadline_slot
                # Ground station at the deadline slot can deliver data to sink
                for gs_node in gs_nodes_by_slot.get(task.deadline_slot, []):
                    sink_arc = TempArc(gs_node, sink, 'sink_arc')
                    sink_arc.cost = 0.0
                    sink_arc.capacity = 1
                    self.temp_network.add_arc(sink_arc)

                # Also add arcs from ground to virtual ground station (data delivery endpoint)
                # Capacity: bandwidth * slot / data = 1000 * 200 / 8000 = 25 tasks per slot per GS antenna
                for k in range(self.num_slots):
                    for gs_node in gs_nodes_by_slot.get(k, []):
                        vgs_arc = TempArc(gs_node, self.virtual_gs, 'vgs_arc')
                        vgs_arc.cost = 0.0
                        vgs_arc.capacity = 25  # 下行带宽容量：1000Mbps * 200s / 8000Mb = 25任务/时隙
                        self.temp_network.add_arc(vgs_arc)

            # virtual_gs -> sink (capacity=9999: unlimited aggregate from virtual_gs to each sink)
            vgs_to_sink = TempArc(self.virtual_gs, sink, 'sink_arc')
            vgs_to_sink.cost = 0.0
            vgs_to_sink.capacity = 9999
            self.temp_network.add_arc(vgs_to_sink)

    def set_arc_capacities(self):
        """Set capacities on all arcs based on satellite parameters.

        Note: cross_domain_arc and cross_domain_convergence_arc capacities are
        pre-calculated in add_cross_domain_arcs() and should not be overwritten.
        """
        for arc in self.temp_network.arcs:
            # Skip cross_domain arcs - they have pre-calculated capacities
            if arc.arc_type in ['cross_domain_arc', 'cross_domain_convergence_arc']:
                continue
            if arc.start.satellite in self.sat_params:
                params = self.sat_params[arc.start.satellite]
                # ISL capacity: bandwidth * slot / data = 1000 * 200 / 8000 = 25 tasks per slot
                isl_capacity = int(1000.0 * 200 / 8000)
                arc.set_capacity(arc.arc_type, {
                    'storage_capacity': params.storage_capacity,
                    'isl_capacity': isl_capacity,
                    'downlink_bandwidth': params.downlink_bandwidth,
                })

    def calculate_incidence_matrix(self):
        """Calculate incidence matrix for temporal network.

        IM(a, arc) = 1, IM(b, arc) = -1 for arc = (a, b)
        """
        self.incidence_matrix = {}
        for node in self.temp_network.nodes:
            self.incidence_matrix[node] = {}

        for arc in self.temp_network.arcs:
            self.incidence_matrix[arc.start][arc] = 1
            self.incidence_matrix[arc.end][arc] = -1

    def build_heuristic_dist(self):
        """Build heuristic distance for A* search.

        Reference: README Section 3.5 acceleration mechanism 1 (A*启发式函数)

        Uses full node keys to match the actual graph structure.
        """
        self.heuristic_dist = {}

        def get_node_key(node):
            return f"{node.satellite}_{node.time_slot}_{node.node_type}_{node.route}"

        for arc in self.temp_network.arcs:
            # Use full node keys to match graph representation
            start_key = get_node_key(arc.start)
            end_key = get_node_key(arc.end)
            key = (start_key, end_key)
            if key not in self.heuristic_dist or arc.cost < self.heuristic_dist[key]:
                self.heuristic_dist[key] = arc.cost

    def get_heuristic_fast(self):
        """All tasks end at VGS; cache one exact reverse base-cost table."""
        import networkx as nx
        from .satellite_generator import build_graph
        self.sinks = {task.task_id: self.virtual_gs for task in self.tasks}
        graph = build_graph(self)
        sink = f'{self.virtual_gs.satellite}_{self.virtual_gs.time_slot}_{self.virtual_gs.node_type}_{self.virtual_gs.route}'
        distances = nx.single_source_dijkstra_path_length(graph.reverse(copy=False), sink, weight='weight')
        self.heuristic = {node: {sink: dist} for node, dist in distances.items()}

    def _compute_heuristic_per_sink(self, G):
        """Fallback: compute heuristic by running Dijkstra from each sink."""
        import heapq

        sink_nodes = [n for n in G.nodes if n.startswith('sink_') or 'virtual_gs' in n]

        self.heuristic = {}
        for node in G.nodes:
            if 'sink_' not in node and 'source_' not in node:
                self.heuristic[node] = {}

        for sink in sink_nodes:
            # Dijkstra from sink on reversed graph
            dist = {n: float('inf') for n in G.nodes}
            dist[sink] = 0.0
            pq = [(0.0, sink)]
            visited = set()

            while pq:
                d, u = heapq.heappop(pq)
                if u in visited:
                    continue
                visited.add(u)

                for v, data in G.reverse()[u].items():
                    if v in visited:
                        continue
                    new_dist = d + data.get('weight', 1.0)
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        heapq.heappush(pq, (new_dist, v))

            for node in G.nodes:
                if 'sink_' not in node and 'source_' not in node:
                    if node not in self.heuristic:
                        self.heuristic[node] = {}
                    min_val = self.heuristic[node].get(sink, float('inf'))
                    self.heuristic[node][sink] = min(min_val, dist.get(node, float('inf')))


# Ground station configuration - 5 stations
DEFAULT_GS_CONFIG = {
    # China National Network (国网) Ground Stations
    'Beijing': {'lat': 39.9, 'lon': 116.4, 'bandwidth': 20000},
    'Shanghai': {'lat': 31.2, 'lon': 121.5, 'bandwidth': 20000},
    'Guangzhou': {'lat': 23.1, 'lon': 113.3, 'bandwidth': 20000},
    'Chengdu': {'lat': 30.6, 'lon': 104.0, 'bandwidth': 20000},
    'Wuhan': {'lat': 30.6, 'lon': 114.3, 'bandwidth': 20000},
    'Xian': {'lat': 34.3, 'lon': 108.9, 'bandwidth': 20000},
    'Urumqi': {'lat': 43.8, 'lon': 87.6, 'bandwidth': 20000},
    'Kashgar': {'lat': 39.5, 'lon': 76.0, 'bandwidth': 20000},
    'Haikou': {'lat': 20.0, 'lon': 110.3, 'bandwidth': 20000},
    'Kunming': {'lat': 25.0, 'lon': 102.7, 'bandwidth': 20000},
    'Lhasa': {'lat': 29.6, 'lon': 91.1, 'bandwidth': 20000},
    'Shenzhen': {'lat': 22.5, 'lon': 114.1, 'bandwidth': 20000},
    'Nanjing': {'lat': 32.0, 'lon': 118.8, 'bandwidth': 20000},
    'Hangzhou': {'lat': 30.3, 'lon': 120.2, 'bandwidth': 20000},
    'Chongqing': {'lat': 29.5, 'lon': 106.5, 'bandwidth': 20000},
    'Taiyuan': {'lat': 37.9, 'lon': 112.5, 'bandwidth': 20000},
    'Hohhot': {'lat': 40.8, 'lon': 111.7, 'bandwidth': 20000},
    'Harbin': {'lat': 45.8, 'lon': 126.5, 'bandwidth': 20000},
    'Changchun': {'lat': 43.9, 'lon': 125.3, 'bandwidth': 20000},
    'Shenyang': {'lat': 41.8, 'lon': 123.4, 'bandwidth': 20000},
    'Jinan': {'lat': 36.6, 'lon': 117.0, 'bandwidth': 20000},
    'Zhengzhou': {'lat': 34.7, 'lon': 113.6, 'bandwidth': 20000},
    'Changsha': {'lat': 28.2, 'lon': 112.9, 'bandwidth': 20000},
    'Nanchang': {'lat': 28.7, 'lon': 115.9, 'bandwidth': 20000},
    'Fuzhou': {'lat': 26.0, 'lon': 119.3, 'bandwidth': 20000},
    'Guiyang': {'lat': 26.6, 'lon': 106.7, 'bandwidth': 20000},
    'Nanning': {'lat': 22.8, 'lon': 108.3, 'bandwidth': 20000},
    'Yanqi': {'lat': 42.0, 'lon': 86.5, 'bandwidth': 20000},
    'Yongxing': {'lat': 40.5, 'lon': 93.7, 'bandwidth': 20000},
    'Sanya': {'lat': 18.3, 'lon': 109.5, 'bandwidth': 20000},
    # International/Ocean ground stations for global coverage
    'Kiruna': {'lat': 67.8, 'lon': 20.4, 'bandwidth': 20000},  # Sweden (Arctic)
    'McMurdo': {'lat': -77.8, 'lon': 166.7, 'bandwidth': 20000},  # Antarctica
    'Singapore': {'lat': 1.3, 'lon': 103.8, 'bandwidth': 20000},
    'Sydney': {'lat': -33.9, 'lon': 151.2, 'bandwidth': 20000},
    'Johannesburg': {'lat': -26.2, 'lon': 28.0, 'bandwidth': 20000},
    'Svalbard': {'lat': 78.2, 'lon': 15.6, 'bandwidth': 20000},  # Norway (Arctic)
    'Inuvik': {'lat': 68.3, 'lon': -133.7, 'bandwidth': 20000},  # Canada (Arctic)
    'Punta_Arenas': {'lat': -53.2, 'lon': -70.9, 'bandwidth': 20000},  # Chile
    'Perth': {'lat': -32.0, 'lon': 115.9, 'bandwidth': 20000},
}


if __name__ == '__main__':
    # Test instance generator
    print("Testing Instance generator...")

    parser = TLEParser()
    sats = parser._get_sample_satellites()

    # Take subset of satellites
    test_sats = sats * 10  # Simulate more satellites

    time_window = TimeWindow(start=0.0, end=60.0)  # 60 minute window

    instance = Instance(
        time_window=time_window,
        rs_count=20,
        cs_count=80,
        gs_config=DEFAULT_GS_CONFIG,
        tle_data=test_sats[:20],  # 20 satellites
        subset=0.1  # 10% of tasks
    )

    print(f"Building instance with {len(test_sats[:20])} satellites, {instance.num_slots} slots...")
    instance.build_instance(seed=42)

    print(f"  Temp nodes: {len(instance.temp_network.nodes)}")
    print(f"  Temp arcs: {len(instance.temp_network.arcs)}")
    print(f"  Tasks: {len(instance.tasks)}")
