# GeoDataQueryAgent-MVP

## Purpose
This directory is a placeholder for the **GeoDataQueryAgent-MVP**.

The `GeoDataQueryAgent-MVP` is responsible for retrieving predefined geospatial data elements from the Minimum Viable Product's (MVP) simplified Planetary Data Nexus (PDN-MVP). For the MVP, this involves providing the file paths to specific KML or GeoJSON files representing parts of the "Mini-Storybook."

## MVP Functionality
-   Receives a simulated task from the TOE-MVP specifying a geospatial element name.
-   Constructs the known file path to the corresponding data file within the PDN-MVP.
-   Returns this file path as its result.

## Detailed Specification
For detailed specifications, including its exact simulated A2A interactions and expected inputs/outputs for the MVP, please refer to **Section 3.2 (GeoDataQueryAgent-MVP)** in the main `a2a_world_mvp_specifications.md` document located in the `specifications/` directory of this repository.

## Future Development
In future iterations beyond the MVP, this agent would:
-   Interact with a fully implemented Planetary Data Nexus.
-   Perform more complex queries for geospatial data based on various criteria (location, type, resolution, time).
-   Handle actual A2A protocol communication.
