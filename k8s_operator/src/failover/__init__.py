"""
Multi-Region Failover Services

Handles automatic failover, recovery, and DNS management for multi-region deployments.
"""

from .failover_controller import FailoverController, FailoverEvent
from .dns_updater import DNSUpdater
from .recovery import RecoveryManager

__all__ = ['FailoverController', 'FailoverEvent', 'DNSUpdater', 'RecoveryManager']
