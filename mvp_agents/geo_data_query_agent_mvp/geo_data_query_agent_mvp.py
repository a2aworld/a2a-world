import json # Not strictly needed for this agent's logic, but good practice if it evolves

# Updated data map with element_ids as keys
MVP_GEOSPATIAL_ELEMENT_TO_KML_PATH = {
    # Primary MVP elements from a2a_world_mvp_specifications.md (Sections 1.2, 3.2)
    # These are simulated to be known by the FMS-MVP / GeoDataQueryAgent-MVP
    "mvp:SunStonePeak": "PDN_MVP/geospatial/sun_stone_peak.kml",
    "mvp:RiverOfReflection": "PDN_MVP/geospatial/river_of_reflection.kml",

    # Additional elements derived from the provided ckg_mvp.jsonld for broader MVP testing
    "mvp:HamsaHandGeoFeature": "PDN_MVP/geospatial/hamsa_hand_formation.kml",
    "mvp:PrometheusEagleGeoFeature": "PDN_MVP/geospatial/prometheus_sites.kml",
    "mvp:PrometheusFireTheftGeoFeature": "PDN_MVP/geospatial/prometheus_sites.kml", # Corresponds to old "Prometheus Stealing Fire Site"
    "mvp:ZeusFireTheftGeoFeature": "PDN_MVP/geospatial/prometheus_sites.kml",       # Corresponds to old "Zeus Taking Fire Site"
    "mvp:PrometheusOfferingGeoFeature": "PDN_MVP/geospatial/prometheus_sites.kml", # Corresponds to old "Prometheus's Offering Site"
}

def handle_geo_data_query_task(task_id: str, element_id: str) -> dict:
    """
    Retrieves the KML file path for a known geospatial element ID.
    This agent simulates querying a Foundational Match Service (FMS) for MVP purposes.

    Args:
        task_id: An identifier for the task.
        element_id: The unique ID of the geospatial element (e.g., "mvp:SunStonePeak").

    Returns:
        A dictionary structured according to A2A MVP specifications for SubmitTaskResult.
    """
    if element_id in MVP_GEOSPATIAL_ELEMENT_TO_KML_PATH:
        file_path = MVP_GEOSPATIAL_ELEMENT_TO_KML_PATH[element_id]
        return {
            "task_id": task_id,
            "status": "COMPLETED",
            "artifact": {
                "parts": [
                    {"data": {"elementId": element_id, "filePath": file_path}}
                ]
            }
        }
    else:
        return {
            "task_id": task_id,
            "status": "FAILED",
            "artifact": {
                "parts": [
                    {"text": f"Error: Element ID '{element_id}' not found in geospatial data map."}
                ]
            }
        }

if __name__ == '__main__':
    print("--- Testing GeoDataQueryAgent-MVP (Refactored) ---")

    # Test case 1: Known MVP element - SunStonePeak
    task1_id = "geo_task_001"
    element1_id = "mvp:SunStonePeak"
    result1 = handle_geo_data_query_task(task1_id, element1_id)
    print(f"\nRequesting Task ID: {task1_id}, Element ID: {element1_id}")
    print(f"Result: {json.dumps(result1, indent=2)}")
    assert result1["task_id"] == task1_id
    assert result1["status"] == "COMPLETED"
    assert result1["artifact"]["parts"][0]["data"]["elementId"] == element1_id
    assert result1["artifact"]["parts"][0]["data"]["filePath"] == "PDN_MVP/geospatial/sun_stone_peak.kml"

    # Test case 2: Known MVP element - RiverOfReflection
    task2_id = "geo_task_002"
    element2_id = "mvp:RiverOfReflection"
    result2 = handle_geo_data_query_task(task2_id, element2_id)
    print(f"\nRequesting Task ID: {task2_id}, Element ID: {element2_id}")
    print(f"Result: {json.dumps(result2, indent=2)}")
    assert result2["task_id"] == task2_id
    assert result2["status"] == "COMPLETED"
    assert result2["artifact"]["parts"][0]["data"]["filePath"] == "PDN_MVP/geospatial/river_of_reflection.kml"

    # Test case 3: Known element from CKG - HamsaHandGeoFeature
    task3_id = "geo_task_003"
    element3_id = "mvp:HamsaHandGeoFeature"
    result3 = handle_geo_data_query_task(task3_id, element3_id)
    print(f"\nRequesting Task ID: {task3_id}, Element ID: {element3_id}")
    print(f"Result: {json.dumps(result3, indent=2)}")
    assert result3["status"] == "COMPLETED"
    assert result3["artifact"]["parts"][0]["data"]["filePath"] == "PDN_MVP/geospatial/hamsa_hand_formation.kml"

    # Test case 4: Known element from CKG - PrometheusEagleGeoFeature
    task4_id = "geo_task_004"
    element4_id = "mvp:PrometheusEagleGeoFeature"
    result4 = handle_geo_data_query_task(task4_id, element4_id)
    print(f"\nRequesting Task ID: {task4_id}, Element ID: {element4_id}")
    print(f"Result: {json.dumps(result4, indent=2)}")
    assert result4["status"] == "COMPLETED"
    assert result4["artifact"]["parts"][0]["data"]["filePath"] == "PDN_MVP/geospatial/prometheus_sites.kml"

    # Test case 5: Unknown element
    task5_id = "geo_task_005"
    element5_id = "mvp:AtlantisMainPlaza" # This ID is not in our map
    result5 = handle_geo_data_query_task(task5_id, element5_id)
    print(f"\nRequesting Task ID: {task5_id}, Element ID: {element5_id}")
    print(f"Result: {json.dumps(result5, indent=2)}")
    assert result5["task_id"] == task5_id
    assert result5["status"] == "FAILED"
    assert "Error: Element ID 'mvp:AtlantisMainPlaza' not found" in result5["artifact"]["parts"][0]["text"]
    
    # Test case 6: Case sensitivity test for element ID (should fail if IDs are exact)
    task6_id = "geo_task_006"
    element6_id = "mvp:sunstonepeak" # Lowercase, assuming map keys are case-sensitive
    result6 = handle_geo_data_query_task(task6_id, element6_id)
    print(f"\nRequesting Task ID: {task6_id}, Element ID: {element6_id}")
    print(f"Result: {json.dumps(result6, indent=2)}")
    assert result6["status"] == "FAILED" # Should fail as map keys are case-sensitive

    print("\n--- All refactored test cases passed (if assertions are met) ---")

```
