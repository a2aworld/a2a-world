import json
import os

# MVP_BASE_URI = "http://a2a.world/ontology/mvp#" # Not used as element_name is expected to be full @id

def handle_culture_data_query_task(task_id, request_type, element_name, ckg_filepath):
    """
    Handles cultural data queries based on the MVP specifications.

    Args:
        task_id (str): An identifier for the task, used in the response.
        request_type (str): The type of query ("get_cultural_links" or "get_associated_symbol").
        element_name (str): The @id of the element to query against (e.g., "mvp:SunStonePeak").
        ckg_filepath (str): Path to the ckg_mvp.jsonld file.

    Returns:
        dict: A dictionary structured according to A2A MVP specifications.
    """
    try:
        with open(ckg_filepath, 'r') as f:
            ckg_data = json.load(f)
    except FileNotFoundError:
        return {
            "task_id": task_id,
            "status": "FAILED",
            "error_message": f"CKG file not found at: {ckg_filepath}",
            "artifact": {"parts": [{"data": {"elementName": element_name, "reason": "CKG file not found"}}]}
        }
    except json.JSONDecodeError:
        return {
            "task_id": task_id,
            "status": "FAILED",
            "error_message": f"Failed to decode JSON from CKG file: {ckg_filepath}",
            "artifact": {"parts": [{"data": {"elementName": element_name, "reason": "CKG JSON decode error"}}]}
        }

    graph = ckg_data.get("@graph", [])
    if not graph:
        return {
            "task_id": task_id,
            "status": "FAILED",
            "error_message": "CKG data does not contain a '@graph' array or it is empty.",
            "artifact": {"parts": [{"data": {"elementName": element_name, "reason": "CKG @graph missing or empty"}}]}
        }

    if request_type == "get_cultural_links":
        narrative_links = []
        events = []
        
        for item in graph:
            # Ensure item is a dictionary
            if not isinstance(item, dict):
                continue

            if item.get("@type") == "mvp:NarrativeElement":
                mentions = item.get("mentionsFeature", [])
                # Ensure mentions is always a list for consistent processing
                if not isinstance(mentions, list):
                    mentions = [mentions]
                
                if element_name in mentions:
                    narrative_id = item.get("@id")
                    if narrative_id: # Only add if narrative has an ID
                        narrative_links.append(narrative_id)
                    
                    described_events_raw = item.get("describesEvent")
                    if described_events_raw:
                        if isinstance(described_events_raw, list):
                            events.extend(event_id for event_id in described_events_raw if event_id) # Add valid event IDs
                        elif isinstance(described_events_raw, str): # If it's a single string ID
                            events.append(described_events_raw)
        
        # Remove duplicates
        narrative_links = sorted(list(set(narrative_links)))
        events = sorted(list(set(events)))

        # As per spec, "COMPLETED" even if links are empty.
        # Failure would be due to file issues or fundamentally malformed CKG.
        return {
            "task_id": task_id,
            "status": "COMPLETED",
            "artifact": {
                "parts": [
                    {"data": {"elementName": element_name, "narrativeLinks": narrative_links, "events": events}}
                ]
            }
        }

    elif request_type == "get_associated_symbol":
        visual_element_id_found = None
        symbol_id_found = None
        
        geospatial_feature = None
        for item in graph:
            if not isinstance(item, dict): continue
            if item.get("@id") == element_name and item.get("@type") == "mvp:GeospatialFeature":
                geospatial_feature = item
                break
        
        if not geospatial_feature:
            return {
                "task_id": task_id,
                "status": "FAILED",
                "error_message": f"Geospatial feature '{element_name}' of type 'mvp:GeospatialFeature' not found in CKG.",
                "artifact": {"parts": [{"data": {"elementName": element_name, "reason": "Geospatial feature not found"}}]}
            }

        associated_visual_raw = geospatial_feature.get("associatedWithVisual")
        if not associated_visual_raw:
            return {
                "task_id": task_id,
                "status": "FAILED",
                "error_message": f"Geospatial feature '{element_name}' has no 'associatedWithVisual' property.",
                "artifact": {"parts": [{"data": {"elementName": element_name, "reason": "associatedWithVisual property missing"}}]}
            }
        
        # Per CKG, associatedWithVisual is a single ID string, not a list.
        # If it were a list in some CKG variant: current_visual_id_to_find = associated_visual_raw[0] if isinstance(associated_visual_raw, list) else associated_visual_raw
        current_visual_id_to_find = associated_visual_raw if isinstance(associated_visual_raw, str) else None
        if not current_visual_id_to_find:
             return {
                "task_id": task_id,
                "status": "FAILED",
                "error_message": f"Property 'associatedWithVisual' for '{element_name}' is not a valid ID string.",
                "artifact": {"parts": [{"data": {"elementName": element_name, "reason": "'associatedWithVisual' property malformed"}}]}
            }


        visual_element = None
        for item in graph:
            if not isinstance(item, dict): continue
            if item.get("@id") == current_visual_id_to_find and item.get("@type") == "mvp:VisualElement":
                visual_element = item
                break
        
        if not visual_element:
            return {
                "task_id": task_id,
                "status": "FAILED",
                "error_message": f"Visual element '{current_visual_id_to_find}' (linked by '{element_name}') not found or not of type 'mvp:VisualElement'.",
                "artifact": {"parts": [{"data": {"elementName": element_name, "reason": "Visual element not found"}}]}
            }
        
        visual_element_id_found = visual_element.get("@id") 
        depicted_symbol_raw = visual_element.get("depictsSymbol")
        if not depicted_symbol_raw:
            # This is a valid case per CKG for mvp:SolsticeSunriseImage. It should return "FAILED" as a symbol is expected by this request type.
            return {
                "task_id": task_id,
                "status": "FAILED",
                "error_message": f"Visual element '{visual_element_id_found}' has no 'depictsSymbol' property.",
                "artifact": {"parts": [{"data": {"elementName": element_name, "visualElement": visual_element_id_found, "reason": "'depictsSymbol' property missing"}}]}
            }

        # Per CKG, depictsSymbol is a single ID string.
        symbol_id_found = depicted_symbol_raw if isinstance(depicted_symbol_raw, str) else None
        if not symbol_id_found:
             return {
                "task_id": task_id,
                "status": "FAILED",
                "error_message": f"Property 'depictsSymbol' for visual element '{visual_element_id_found}' is not a valid ID string.",
                "artifact": {"parts": [{"data": {"elementName": element_name, "visualElement": visual_element_id_found, "reason": "'depictsSymbol' property malformed"}}]}
            }

        return {
            "task_id": task_id,
            "status": "COMPLETED",
            "artifact": {
                "parts": [
                    {"data": {"elementName": element_name, "associatedSymbol": symbol_id_found, "visualElement": visual_element_id_found}}
                ]
            }
        }

    else:
        return {
            "task_id": task_id,
            "status": "FAILED",
            "error_message": f"Unknown request_type: {request_type}",
            "artifact": {"parts": [{"data": {"elementName": element_name, "reason": "Unknown request type"}}]}
        }

if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    # Adjust if your ckg_mvp.jsonld is elsewhere relative to the script.
    # Assuming ckg_mvp.jsonld is in the root of the repository, and this script is in mvp_agents/culture_data_query_agent_mvp/
    ckg_path_example = os.path.join(script_dir, "..", "..", "ckg_mvp.jsonld")
    ckg_path_example = os.path.normpath(ckg_path_example)

    if not os.path.exists(ckg_path_example):
        print(f"CRITICAL: ckg_mvp.jsonld not found at expected path: {ckg_path_example}")
        print("Please ensure 'ckg_mvp.jsonld' is in the repository root.")
        print("Example tests will likely fail.")
    else:
        print(f"CKG file found at: {ckg_path_example}")

    print(f"\n--- Running Test Cases for CultureDataQueryAgent-MVP ---")

    # Test case 1: Get cultural links for SunStonePeak (expected: SkySerpentGiftNarrative, SolsticeGiftEvent)
    task1_id = "test001"
    request1_type = "get_cultural_links"
    element1_name = "mvp:SunStonePeak"
    result1 = handle_culture_data_query_task(task1_id, request1_type, element1_name, ckg_path_example)
    print(f"\nTask: {task1_id}, Request: {request1_type}, Element: {element1_name}")
    print(json.dumps(result1, indent=2))

    # Test case 2: Get associated symbol for RiverOfReflection (expected: SkySerpentSymbol, RiverSerpentGlyphImage)
    task2_id = "test002"
    request2_type = "get_associated_symbol"
    element2_name = "mvp:RiverOfReflection"
    result2 = handle_culture_data_query_task(task2_id, request2_type, element2_name, ckg_path_example)
    print(f"\nTask: {task2_id}, Request: {request2_type}, Element: {element2_name}")
    print(json.dumps(result2, indent=2))

    # Test case 3: Element not found for symbol association
    task3_id = "test003"
    request3_type = "get_associated_symbol"
    element3_name = "mvp:NonExistentPeak"
    result3 = handle_culture_data_query_task(task3_id, request3_type, element3_name, ckg_path_example)
    print(f"\nTask: {task3_id}, Request: {request3_type}, Element: {element3_name} (expected FAILED - GeoFeature not found)")
    print(json.dumps(result3, indent=2))

    # Test case 4: Cultural links for an element with no narrative mentions (e.g., a symbol)
    task4_id = "test004"
    request4_type = "get_cultural_links"
    element4_name = "mvp:SkySerpentSymbol" # Symbols are not directly mentioned via 'mentionsFeature' in narratives
    result4 = handle_culture_data_query_task(task4_id, request4_type, element4_name, ckg_path_example)
    print(f"\nTask: {task4_id}, Request: {request4_type}, Element: {element4_name} (expected COMPLETED, empty links/events)")
    print(json.dumps(result4, indent=2))
    
    # Test case 5: Unknown request type
    task5_id = "test005"
    request5_type = "get_unknown_data_type"
    element5_name = "mvp:SunStonePeak"
    result5 = handle_culture_data_query_task(task5_id, request5_type, element5_name, ckg_path_example)
    print(f"\nTask: {task5_id}, Request: {request5_type}, Element: {element5_name} (expected FAILED - Unknown request type)")
    print(json.dumps(result5, indent=2))

    # Test case 6: CKG file not found
    task6_id = "test006"
    request6_type = "get_cultural_links"
    element6_name = "mvp:SunStonePeak"
    result6 = handle_culture_data_query_task(task6_id, request6_type, element6_name, "non_existent_ckg.jsonld")
    print(f"\nTask: {task6_id}, Request: {request6_type}, Element: {element6_name} (expected FAILED - CKG not found)")
    print(json.dumps(result6, indent=2))

    # Test case 7: Cultural links for RiverOfReflection (expected: SkySerpentGiftNarrative, SolsticeGiftEvent)
    task7_id = "test007"
    request7_type = "get_cultural_links"
    element7_name = "mvp:RiverOfReflection"
    result7 = handle_culture_data_query_task(task7_id, request7_type, element7_name, ckg_path_example)
    print(f"\nTask: {task7_id}, Request: {request7_type}, Element: {element7_name}")
    print(json.dumps(result7, indent=2))
    
    # Test case 8: Get associated symbol for SunStonePeak. 
    # SunStonePeak -> SolsticeSunriseImage. SolsticeSunriseImage has NO depictsSymbol. Expected FAILED.
    task8_id = "test008"
    request8_type = "get_associated_symbol"
    element8_name = "mvp:SunStonePeak"
    result8 = handle_culture_data_query_task(task8_id, request8_type, element8_name, ckg_path_example)
    print(f"\nTask: {task8_id}, Request: {request8_type}, Element: {element8_name} (expected FAILED - depictsSymbol missing from SolsticeSunriseImage)")
    print(json.dumps(result8, indent=2))

    # Test case 9: Get associated symbol for an element that is not a GeospatialFeature
    task9_id = "test009"
    request9_type = "get_associated_symbol"
    element9_name = "mvp:SkySerpentGiftNarrative" # This is a NarrativeElement
    result9 = handle_culture_data_query_task(task9_id, request9_type, element9_name, ckg_path_example)
    print(f"\nTask: {task9_id}, Request: {request9_type}, Element: {element9_name} (expected FAILED - Not a GeoFeature)")
    print(json.dumps(result9, indent=2))

    # Test case 10: GeoFeature exists but is missing 'associatedWithVisual' (requires modified CKG or specific example)
    # For this, we'd need an element in CKG that IS a GeoFeature but LACKS the property.
    # If ckg_mvp.jsonld were modified for 'mvp:SunStonePeak' to remove 'associatedWithVisual', this path would be tested.
    # Current CKG has this property for both GeoFeatures, so this path isn't hit by standard data.
    # The logic for this is: `if not associated_visual_raw: return FAILED...`
    print(f"\nTest Case 10: (Conceptual) GeoFeature missing 'associatedWithVisual'. Covered by code, needs specific CKG data to trigger.")

    # Test case 11: VisualElement exists but 'depictsSymbol' is malformed (e.g., not a string)
    # This also requires a modified CKG. Example: if 'depictsSymbol' was `{"@id": "mvp:SkySerpentSymbol"}` instead of just the string.
    # Current code: `symbol_id_found = depicted_symbol_raw if isinstance(depicted_symbol_raw, str) else None`
    # Followed by `if not symbol_id_found: return FAILED...`
    print(f"Test Case 11: (Conceptual) VisualElement 'depictsSymbol' malformed. Covered by code, needs specific CKG data to trigger.")
    
    print(f"\n--- Test Cases Finished ---")

```
