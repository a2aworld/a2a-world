import json
import os

def handle_cross_referencing_task(
    task_id,
    geo_element1_filepath,
    cultural_links_element1,
    symbol_association_element2,
    narrative_filepath,
    element1_name_in_narrative, # e.g., "Sun Stone Peak"
    element2_name_in_narrative  # e.g., "River of Reflection"
    ):
    """
    Handles the Basic Cross-Referencing task as per MVP specifications.

    Args:
        task_id (str): Identifier for the task.
        geo_element1_filepath (str): Filepath to the KML file for the first geospatial element.
        cultural_links_element1 (dict): Cultural links data for the first element.
                                        Example: {"elementName": "mvp:SunStonePeak", 
                                                  "narrativeLinks": ["mvp:SkySerpentGiftNarrative"], 
                                                  "events": ["mvp:SolsticeGiftEvent"]}
        symbol_association_element2 (dict): Symbol association data for the second element.
                                           Example: {"elementName": "mvp:RiverOfReflection", 
                                                     "associatedSymbol": "mvp:SkySerpentSymbol", 
                                                     "visualElement": "mvp:RiverSerpentGlyphImage"}
        narrative_filepath (str): Filepath to the narrative text file.
        element1_name_in_narrative (str): Keyword for the first element in the narrative.
        element2_name_in_narrative (str): Keyword for the second element in the narrative.

    Returns:
        dict: A dictionary structured according to A2A MVP specifications.
    """
    narrative_content = ""
    try:
        with open(narrative_filepath, 'r') as f:
            narrative_content = f.read()
    except FileNotFoundError:
        return {
            "task_id": task_id,
            "status": "FAILED",
            "artifact": {
                "parts": [
                    {"text": f"Error: Narrative file not found at {narrative_filepath}."}
                ]
            }
        }

    # Condition 1: Verify cultural_links_element1 has a narrative link
    narrative_linked = False
    primary_narrative_id = None
    if cultural_links_element1 and \
       isinstance(cultural_links_element1.get("narrativeLinks"), list) and \
       len(cultural_links_element1["narrativeLinks"]) > 0:
        # Use the first narrative ID as the primary one for this MVP check
        temp_narrative_id = cultural_links_element1["narrativeLinks"][0]
        if isinstance(temp_narrative_id, str) and temp_narrative_id.strip():
            primary_narrative_id = temp_narrative_id
            narrative_linked = True

    # Condition 2: Verify narrative text contains both element names (case-insensitive)
    text_contains_element1 = False
    text_contains_element2 = False
    if narrative_content: # Ensure narrative_content was loaded
        if element1_name_in_narrative and isinstance(element1_name_in_narrative, str) and \
           element1_name_in_narrative.lower() in narrative_content.lower():
            text_contains_element1 = True
        if element2_name_in_narrative and isinstance(element2_name_in_narrative, str) and \
           element2_name_in_narrative.lower() in narrative_content.lower():
            text_contains_element2 = True
    
    narrative_text_verified = text_contains_element1 and text_contains_element2

    # Condition 3: Verify symbol_association_element2 links to a symbol and visual element
    symbol_association_verified = False
    associated_symbol_id = None
    visual_element_id = None
    if symbol_association_element2 and \
       isinstance(symbol_association_element2.get("associatedSymbol"), str) and \
       symbol_association_element2["associatedSymbol"].strip() and \
       isinstance(symbol_association_element2.get("visualElement"), str) and \
       symbol_association_element2["visualElement"].strip():
        associated_symbol_id = symbol_association_element2["associatedSymbol"]
        visual_element_id = symbol_association_element2["visualElement"]
        symbol_association_verified = True

    # Check all conditions
    if narrative_linked and narrative_text_verified and symbol_association_verified:
        narrative_filename = os.path.basename(narrative_filepath)
        geo_element1_filename = os.path.basename(geo_element1_filepath)
        
        # Construct success message using available IDs and names
        # primary_narrative_id is like "mvp:SkySerpentGiftNarrative"
        # elementX_name_in_narrative are like "Sun Stone Peak", "River of Reflection"
        # associated_symbol_id is like "mvp:SkySerpentSymbol"
        # visual_element_id is like "mvp:RiverSerpentGlyphImage"
        message_string = (
            f"MVP Success: Correlation Found - The narrative '{primary_narrative_id}' ({narrative_filename}) "
            f"links '{element1_name_in_narrative}' ({geo_element1_filename}) with '{element2_name_in_narrative}'. "
            f"The '{element2_name_in_narrative}' is associated with '{associated_symbol_id}' "
            f"via visual '{visual_element_id}'."
        )
    else:
        message_string = "MVP Correlation Not Found: Conditions for predefined link not fully met based on provided inputs."
        # Optional: Add more detailed failure reasons for debugging if needed by the TOE or user.
        # details = []
        # if not narrative_linked: details.append("Narrative link missing/invalid from cultural_links_element1")
        # if not narrative_text_verified: details.append(f"Narrative text verification failed (E1: {text_contains_element1}, E2: {text_contains_element2})")
        # if not symbol_association_verified: details.append("Symbol association missing/invalid from symbol_association_element2")
        # if details: message_string += " Reasons: " + "; ".join(details) + "."
        
    return {
        "task_id": task_id,
        "status": "COMPLETED", # Per spec, status is COMPLETED for logic success/failure.
        "artifact": {
            "parts": [
                {"text": message_string}
            ]
        }
    }

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Get absolute path of script dir
    # PDN_MVP is expected to be two levels up from the script's directory
    pdn_mvp_root = os.path.join(script_dir, "..", "..", "PDN_MVP")
    pdn_mvp_root = os.path.normpath(pdn_mvp_root) # Normalize path (e.g. ..\.. becomes actual path)

    # Create dummy files and directories for testing if they don't exist
    # These paths must be absolute or correctly relative to the CWD when the script is run.
    # For consistency, using absolute paths derived from script location.
    geospatial_dir = os.path.join(pdn_mvp_root, "geospatial")
    narrative_dir = os.path.join(pdn_mvp_root, "narrative")
    os.makedirs(geospatial_dir, exist_ok=True)
    os.makedirs(narrative_dir, exist_ok=True)
    
    mock_task_id = "xref_task_mvp_001"
    geo_sunstone_peak_filepath = os.path.join(geospatial_dir, "sun_stone_peak.kml")
    narrative_file = os.path.join(narrative_dir, "sky_serpent_gift.txt")

    # Create a dummy geo file (content doesn't matter for this agent's current logic)
    with open(geo_sunstone_peak_filepath, 'w') as f:
        f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n<Document>\n</Document>\n</kml>")

    # Dummy narrative text from spec for successful test
    dummy_narrative_text_success = (
        "In the age of the ancestors, a pact was made with the Sky. "
        "At the dawn of the longest day, light from the Sun Stone Peak "
        "touches the River of Reflection, awakening the great Sky Serpent "
        "whose shimmering coils trace the water's path through the valley. "
        "The Serpent, nourished by the sun's first kiss upon the waters, "
        "then bestows the gift of wisdom and cyclical renewal upon the land and its people."
    )
    with open(narrative_file, 'w') as f:
        f.write(dummy_narrative_text_success)

    # Inputs for successful correlation
    culture_links_success = {
        "elementName": "mvp:SunStonePeak", 
        "narrativeLinks": ["mvp:SkySerpentGiftNarrative"], 
        "events": ["mvp:SolsticeGiftEvent"]
    }
    symbol_association_success = {
        "elementName": "mvp:RiverOfReflection", 
        "associatedSymbol": "mvp:SkySerpentSymbol", 
        "visualElement": "mvp:RiverSerpentGlyphImage"
    }
    element1_narrative_name_success = "Sun Stone Peak" 
    element2_narrative_name_success = "River of Reflection"

    print(f"--- Running Test Case: Successful Correlation ---")
    print(f"Using PDN_MVP root: {pdn_mvp_root}")
    print(f"Geo Element 1 File: {geo_sunstone_peak_filepath}")
    print(f"Narrative File: {narrative_file}")

    if not os.path.exists(geo_sunstone_peak_filepath): print(f"Warning: Geo file missing at {geo_sunstone_peak_filepath}")
    if not os.path.exists(narrative_file): print(f"Warning: Narrative file missing at {narrative_file}")

    result_success = handle_cross_referencing_task(
        mock_task_id,
        geo_sunstone_peak_filepath,
        culture_links_success,
        symbol_association_success,
        narrative_file,
        element1_narrative_name_success,
        element2_narrative_name_success
    )
    print("\nAgent Result (Success Scenario):")
    print(json.dumps(result_success, indent=2))

    # --- Test Case: Correlation Not Found (Narrative text mismatch) ---
    print(f"\n--- Running Test Case: Correlation Not Found (Narrative text mismatch) ---")
    dummy_narrative_text_fail_content = "Light from the Sun Stone Peak is seen, but the stream is not mentioned here."
    with open(narrative_file, 'w') as f:
        f.write(dummy_narrative_text_fail_content)
    
    result_fail_text = handle_cross_referencing_task(
        "xref_task_mvp_002",
        geo_sunstone_peak_filepath,
        culture_links_success,
        symbol_association_success,
        narrative_file,
        element1_narrative_name_success,
        element2_narrative_name_success # "River of Reflection" will not be found
    )
    print("\nAgent Result (Narrative Text Mismatch):")
    print(json.dumps(result_fail_text, indent=2))

    # --- Test Case: Correlation Not Found (Symbol association missing: empty symbol string) ---
    print(f"\n--- Running Test Case: Correlation Not Found (Symbol association missing) ---")
    symbol_association_fail_symbol = {
        "elementName": "mvp:RiverOfReflection",
        "associatedSymbol": "", # Empty string for symbol
        "visualElement": "mvp:RiverSerpentGlyphImage"
    }
    with open(narrative_file, 'w') as f: # Restore correct narrative
        f.write(dummy_narrative_text_success)
    result_fail_symbol = handle_cross_referencing_task(
        "xref_task_mvp_003",
        geo_sunstone_peak_filepath,
        culture_links_success,
        symbol_association_fail_symbol,
        narrative_file,
        element1_narrative_name_success,
        element2_narrative_name_success
    )
    print("\nAgent Result (Symbol Association Missing - Empty Symbol ID):")
    print(json.dumps(result_fail_symbol, indent=2))

    # --- Test Case: Correlation Not Found (Narrative link missing: empty list) ---
    print(f"\n--- Running Test Case: Correlation Not Found (Narrative link missing) ---")
    culture_links_fail_narrative = {
        "elementName": "mvp:SunStonePeak",
        "narrativeLinks": [], # Empty list for narrative links
        "events": ["mvp:SolsticeGiftEvent"]
    }
    result_fail_narrative_link = handle_cross_referencing_task(
        "xref_task_mvp_004",
        geo_sunstone_peak_filepath,
        culture_links_fail_narrative,
        symbol_association_success,
        narrative_file,
        element1_narrative_name_success,
        element2_narrative_name_success
    )
    print("\nAgent Result (Narrative Link Missing - Empty List):")
    print(json.dumps(result_fail_narrative_link, indent=2))

    # --- Test Case: Narrative File Not Found ---
    print(f"\n--- Running Test Case: Narrative File Not Found ---")
    result_file_not_found = handle_cross_referencing_task(
        "xref_task_mvp_005",
        geo_sunstone_peak_filepath,
        culture_links_success,
        symbol_association_success,
        os.path.join(narrative_dir, "non_existent_narrative.txt"), # Non-existent file
        element1_narrative_name_success,
        element2_narrative_name_success
    )
    print("\nAgent Result (Narrative File Not Found):")
    print(json.dumps(result_file_not_found, indent=2))

    print("\nNote: Dummy files/dirs were created for testing in PDN_MVP relative to script location.")
    print("These are not automatically cleaned up by this test script.")
    print(f"PDN_MVP location: {pdn_mvp_root}")

```
