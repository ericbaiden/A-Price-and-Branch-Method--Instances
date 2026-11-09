"""
Generator module for LEO Constellation Cross-Domain Collaborative Scheduling.
Based on intermodalTransportationNetworksCG framework, adapted for satellite scheduling.
"""

from .tle_parser import TLEParser, TLESatellite
from .satellite_generator import build_graph
from .instance_generator import (
    Instance, TimeWindow, Task,
    FlatNetwork, TempNetwork, FlatNode, TempNode, TempArc,
    SatelliteParams, DEFAULT_GS_CONFIG
)
from .visibility_calculator import compute_visibility_window, VisibilityCalculator

__all__ = [
    'TLEParser', 'TLESatellite',
    'build_graph',
    'Instance', 'TimeWindow', 'Task',
    'FlatNetwork', 'TempNetwork', 'FlatNode', 'TempNode', 'TempArc',
    'SatelliteParams', 'DEFAULT_GS_CONFIG',
    'compute_visibility_window', 'VisibilityCalculator',
]