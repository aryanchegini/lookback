import weakref
from collections.abc import Callable


class EventBus:
    """Minimal Observer hub that holds WEAK references to subscribers.

    Because references are weak, a subscriber that is garbage-collected is
    dropped automatically on the next publish — no manual unsubscribe, no
    "lapsed listener" leak where the bus keeps dead objects alive forever.
    """

    def __init__(self):
        # event name -> list of weak references to callbacks
        self._subscribers: dict[str, list] = {}

    def subscribe(self, event: str, callback: Callable) -> None:
        # WeakMethod for bound methods (obj.handler), plain ref otherwise.
        if hasattr(callback, "__self__"):
            ref = weakref.WeakMethod(callback)
        else:
            ref = weakref.ref(callback)
        self._subscribers.setdefault(event, []).append(ref)

    def publish(self, event: str, *args, **kwargs) -> int:
        """Call every live subscriber; prune dead ones. Returns live count."""
        live = []
        for ref in self._subscribers.get(event, []):
            callback = ref()
            if callback is None:
                continue  # subscriber was GC'd -> drop it
            callback(*args, **kwargs)
            live.append(ref)
        self._subscribers[event] = live
        return len(live)

    def subscriber_count(self, event: str) -> int:
        return sum(ref() is not None for ref in self._subscribers.get(event, []))
