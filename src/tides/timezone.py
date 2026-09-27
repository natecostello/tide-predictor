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


def get_zoneinfo(coord: Coordinate) -> datetime.tzinfo:
    """The coordinate's local timezone, or UTC when none is known.

    Open-ocean points resolve to timezonefinder's nautical Etc/GMT+-N zones,
    which are used as-is (consistently for grouping, default date and
    display); UTC is only the fallback when no zone is returned at all.
    """
    tz_name = get_timezone_name(coord)
    return ZoneInfo(tz_name) if tz_name else datetime.timezone.utc
