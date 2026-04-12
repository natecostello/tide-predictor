import datetime
from zoneinfo import ZoneInfo

from timezonefinder import TimezoneFinder

from tides.models import Coordinate

_tf: TimezoneFinder | None = None


def _get_finder() -> TimezoneFinder:
    global _tf
    if _tf is None:
        _tf = TimezoneFinder()
    return _tf


def get_timezone_name(coord: Coordinate) -> str | None:
    return _get_finder().timezone_at(lat=coord.lat, lng=coord.lon)


def to_local_time(utc_time: datetime.datetime, coord: Coordinate) -> datetime.datetime:
    tz_name = get_timezone_name(coord)
    if tz_name is None:
        return utc_time
    return utc_time.astimezone(ZoneInfo(tz_name))
