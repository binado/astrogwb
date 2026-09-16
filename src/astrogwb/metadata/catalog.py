"""Combined provenance carried by every catalog."""

from pydantic import BaseModel, ConfigDict

from astrogwb.metadata.population import PopulationMetadata
from astrogwb.metadata.waveform import WaveformMetadata


class CatalogMetadata(BaseModel):
    """Waveform and population provenance as one immutable record."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    waveform: WaveformMetadata
    population: PopulationMetadata
