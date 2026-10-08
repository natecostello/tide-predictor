import datetime
from dataclasses import dataclass
from enum import Enum

METERS_TO_FEET = 3.28084


class Source(Enum):
    AUTO = "auto"
    NOAA = "noaa"
    STATION = "station"
    MODEL = "model"


@dataclass(frozen=True)
class Coordinate:
    lat: float
    lon: float

    def __post_init__(self):
        if not -90 <= self.lat <= 90:
            raise ValueError(f"Latitude must be between -90 and 90, got {self.lat}")
        if not -180 <= self.lon <= 180:
            raise ValueError(f"Longitude must be between -180 and 180, got {self.lon}")


@dataclass
class TideEvent:
    time: datetime.datetime
    height: float  # meters
    kind: str  # "high" or "low"

    @property
    def height_ft(self) -> float:
        return self.height * METERS_TO_FEET


@dataclass
class TidePoint(TideEvent):
    """A height on the curve at one moment, for `level` and `when`.

    `kind` is "high"/"low" at (or within tolerance of) a turn, otherwise
    "rising"/"falling". `rate` is m/h (0.0 for high/low). `near` is the
    near-turn window (from, to) for `when` rows reported as a turn.
    """

    rate: float = 0.0  # m/h
    near: tuple[datetime.datetime, datetime.datetime] | None = None

    @property
    def rate_ft(self) -> float:
        return self.rate * METERS_TO_FEET


@dataclass
class TideDay:
    date: datetime.date
    events: list[TideEvent]


@dataclass
class TideResult:
    coordinate: Coordinate
    source_type: Source
    station_id: str | None
    station_name: str | None
    station_distance_km: float | None
    model_name: str | None
    days: list[TideDay]
    datum: str = "msl"
