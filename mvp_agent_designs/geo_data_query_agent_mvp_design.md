## Python Structure Design for `GeoDataQueryAgent-MVP`

1.  **Approach:** A standalone Python function will be used for simplicity, suitable for the MVP's single-purpose nature and ease of use in various environments.

2.  **Function Name:** `get_geospatial_data_path(element_name: str) -> dict`

3.  **Input Parameter:**
    *   `element_name` (string): The specific name of the geospatial element being requested. For the MVP, this will correspond to the names used in the KML files and CKG-MVP (e.g., "Hamsa Hand Formation", "Prometheus's Eagle Formation", "Prometheus Stealing Fire Site", "Zeus Taking Fire Site", "Prometheus's Offering Site").

4.  **Output Structure (Dictionary):**
    *   **On Success:** The function will return a dictionary containing the status of the operation, the requested element name, and the path to its corresponding KML file.
        ```python
        {
            "status": "success",
            "elementName": "Hamsa Hand Formation",  # Echoes the input element_name
            "filePath": "PDN_MVP/geospatial/hamsa_hand_formation.kml"
        }
        ```
        or for a Prometheus site:
        ```python
        {
            "status": "success",
            "elementName": "Prometheus's Eagle Formation", # Echoes the input element_name
            "filePath": "PDN_MVP/geospatial/prometheus_sites.kml"
        }
        ```
    *   **On Failure (Element Not Found):** If the requested `element_name` is not found in the internal mapping, the function will return a dictionary indicating an error.
        ```python
        {
            "status": "error",
            "elementName": "NonExistent Site",  # Echoes the input element_name
            "message": "Geospatial element 'NonExistent Site' not found in known MVP data."
        }
        ```

5.  **Internal Data Storage:**
    *   A predefined Python dictionary, likely defined globally within the script or directly inside the function scope, will serve as the mapping between known `element_name` strings and their corresponding file paths. This dictionary will be hardcoded for the MVP.
    *   **Example Internal Dictionary:**
        ```python
        MVP_GEOSPATIAL_DATA_PATHS = {
            "Hamsa Hand Formation": "PDN_MVP/geospatial/hamsa_hand_formation.kml",
            "Prometheus's Eagle Formation": "PDN_MVP/geospatial/prometheus_sites.kml",
            "Prometheus Stealing Fire Site": "PDN_MVP/geospatial/prometheus_sites.kml",
            "Zeus Taking Fire Site": "PDN_MVP/geospatial/prometheus_sites.kml",
            "Prometheus's Offering Site": "PDN_MVP/geospatial/prometheus_sites.kml"
            # Note: All Prometheus sites point to the same KML file as per previous data generation,
            # as the KML contains multiple Placemarks. The specific Placemark is identified by its name.
        }
        ```
    *   The function will perform a lookup on this dictionary using the input `element_name`. If the key exists, it constructs a success response; otherwise, a failure response.I have created the design document `mvp_agent_designs/geo_data_query_agent_mvp_design.md` with the specified Python structure for the `GeoDataQueryAgent-MVP`.

The document details:
1.  **Approach:** Standalone Python function.
2.  **Function Name:** `get_geospatial_data_path(element_name: str) -> dict`.
3.  **Input Parameter:** `element_name` (string), with examples.
4.  **Output Structure (Dictionary):** Examples for success (including Hamsa and Prometheus sites) and failure cases, with specific error messaging.
5.  **Internal Data Storage:** Description of a predefined Python dictionary (`MVP_GEOSPATIAL_DATA_PATHS`) mapping element names to file paths, including all specified Hamsa and Prometheus elements and a note about multiple Prometheus Placemarks being in a single KML file.
