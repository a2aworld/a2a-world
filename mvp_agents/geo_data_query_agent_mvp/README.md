# GeoDataQueryAgent-MVP

## Purpose
This directory contains the **GeoDataQueryAgent-MVP**.

The `GeoDataQueryAgent-MVP` is responsible for "retrieving" predefined geospatial data elements by providing direct file paths to specific KML files that represent these elements. This simulates interaction with a Foundational Match Service (FMS) for the MVP.

## MVP Functionality
- The agent's core logic is encapsulated in the `handle_geo_data_query_task(task_id: str, element_id: str)` function.
- It receives a simulated task from the TOE-MVP specifying which geospatial element is needed, using an `element_id` (e.g., "mvp:SunStonePeak") as the key.
- Based on the `element_id`, it looks up a predefined KML file path from a hardcoded map (`MVP_GEOSPATIAL_ELEMENT_TO_KML_PATH`).
- Returns its result as a dictionary conforming to the A2A `SubmitTaskResult` structure, containing the `elementId` and its `filePath`, or an error message if not found.

## Detailed Specification
For detailed specifications, including its exact simulated A2A interactions and expected inputs/outputs for the MVP, please refer to **Section 3.2 (GeoDataQueryAgent-MVP)** in the main `a2a_world_mvp_specifications.md` document.

## Running the MVP Agent (Standalone Test)
The Python script `geo_data_query_agent_mvp.py` includes a standalone test block (`if __name__ == "__main__":`). You can run it directly to see example outputs:
```bash
python geo_data_query_agent_mvp.py
```
This test block now calls the `handle_geo_data_query_task` function with various sample `task_id`s and `element_id`s (e.g., "mvp:SunStonePeak", "mvp:HamsaHandGeoFeature"). It prints the A2A `SubmitTaskResult` formatted dictionaries for each test case, demonstrating both successful path retrievals and error handling for unknown element IDs.
