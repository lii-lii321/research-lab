"""路由层共享依赖。"""
from services.tracking import TrackingStore


def get_tracking_store() -> TrackingStore:
    return TrackingStore()
