"""
HiFiGAN Vocoder
"""
from .inference import HiFiGANAPI, load_vocoder
from .model import HifiganGenerator

__all__ = ["HiFiGANAPI", "load_vocoder", "HifiganGenerator"]
