GEOSPATIAL_DATA_MAP = {
    "Hamsa Hand Formation": "PDN_MVP/geospatial/hamsa_hand_formation.kml",
    "Prometheus's Eagle Formation": "PDN_MVP/geospatial/prometheus_sites.kml",
    "Prometheus Stealing Fire Site": "PDN_MVP/geospatial/prometheus_sites.kml",
    "Zeus Taking Fire Site": "PDN_MVP/geospatial/prometheus_sites.kml",
    "Prometheus's Offering Site": "PDN_MVP/geospatial/prometheus_sites.kml"
}

def get_geospatial_data_path(element_name: str) -> dict:
    """
    Retrieves the file path for a known geospatial element.

    Args:
        element_name: The name of the geospatial element.

    Returns:
        A dictionary containing the status, element name, and either
        the file path if found, or an error message if not found.
    """
    if element_name in GEOSPATIAL_DATA_MAP:
        return {
            "status": "success",
            "elementName": element_name,
            "filePath": GEOSPATIAL_DATA_MAP[element_name]
        }
    else:
        return {
            "status": "error",
            "elementName": element_name,
            "message": f"Geospatial element '{element_name}' not found in known data."
        }

if __name__ == '__main__':
    # Example Usage and Test Cases
    print("--- Testing GeoDataQueryAgent-MVP ---")

    # Test case 1: Known Hamsa element
    element1 = "Hamsa Hand Formation"
    result1 = get_geospatial_data_path(element1)
    print(f"\nRequesting: {element1}")
    print(f"Result: {result1}")
    assert result1["status"] == "success"
    assert result1["filePath"] == "PDN_MVP/geospatial/hamsa_hand_formation.kml"

    # Test case 2: Known Prometheus element
    element2 = "Prometheus's Eagle Formation"
    result2 = get_geospatial_data_path(element2)
    print(f"\nRequesting: {element2}")
    print(f"Result: {result2}")
    assert result2["status"] == "success"
    assert result2["filePath"] == "PDN_MVP/geospatial/prometheus_sites.kml"

    # Test case 3: Another known Prometheus element
    element3 = "Prometheus's Offering Site"
    result3 = get_geospatial_data_path(element3)
    print(f"\nRequesting: {element3}")
    print(f"Result: {result3}")
    assert result3["status"] == "success"
    assert result3["filePath"] == "PDN_MVP/geospatial/prometheus_sites.kml"

    # Test case 4: Unknown element
    element4 = "Atlantis Main Plaza"
    result4 = get_geospatial_data_path(element4)
    print(f"\nRequesting: {element4}")
    print(f"Result: {result4}")
    assert result4["status"] == "error"
    assert "not found in known data" in result4["message"]
    
    # Test case 5: Case sensitivity (assuming names are exact)
    element5 = "hamsa hand formation" # Lowercase
    result5 = get_geospatial_data_path(element5)
    print(f"\nRequesting: {element5}")
    print(f"Result: {result5}")
    assert result5["status"] == "error" # Should fail if map keys are case-sensitive

    print("\n--- All test cases passed (if assertions are met) ---")
```
