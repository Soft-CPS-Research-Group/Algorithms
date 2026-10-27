"""Compatibility module for the shared simulator/agent controller."""
import sys
from algorithms.runtime import controller
sys.modules[__name__] = controller
