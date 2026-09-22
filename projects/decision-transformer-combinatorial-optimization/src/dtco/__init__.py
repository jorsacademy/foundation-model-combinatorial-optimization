"""Offline Decision Transformer benchmark for Euclidean TSP."""

from .problem import TourAudit, audit_tour, generate_euclidean_instance, tour_length

__all__ = ["TourAudit", "audit_tour", "generate_euclidean_instance", "tour_length"]
__version__ = "0.1.0"
