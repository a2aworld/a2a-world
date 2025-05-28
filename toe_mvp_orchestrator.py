import os
import sys # For exit
import json # For pretty printing errors, if needed

# Agent imports
from mvp_agents.geo_data_query_agent_mvp.geo_data_query_agent_mvp import handle_geo_data_query_task
from mvp_agents.culture_data_query_agent_mvp.culture_data_query_agent_mvp import handle_culture_data_query_task
from mvp_agents.basic_cross_referencing_agent_mvp.basic_cross_referencing_agent_mvp import handle_cross_referencing_task

def main():
    """
    Main orchestration logic for the TOE-MVP.
    """
    print("--- TOE-MVP Orchestration Started ---")

    # Define file paths (relative to repository root, where this script is located)
    ckg_filepath = "ckg_mvp.jsonld"
    # As per spec 2.1, narrative file is sky_serpent_gift.txt
    narrative_filepath = "PDN_MVP/narrative/sky_serpent_gift.txt" 

    # --- Step 1: Get Geospatial Data for SunStonePeak ---
    print("\nStep 1: Requesting SunStonePeak KML path from GeoDataQueryAgent-MVP...")
    geo_task_001_id = "geo_task_001"
    element_sunstone_peak_id = "mvp:SunStonePeak"
    
    result_geo_ssp = handle_geo_data_query_task(task_id=geo_task_001_id, element_id=element_sunstone_peak_id)

    if result_geo_ssp.get("status") == "FAILED":
        error_message = result_geo_ssp.get("artifact", {}).get("parts", [{}])[0].get("text", "Unknown error from GeoDataQueryAgent.")
        print(f"MVP Error: GeoDataQueryAgent-MVP failed for {element_sunstone_peak_id}. Reason: {error_message}")
        print("Aborting MVP run.")
        sys.exit(1)
    
    try:
        sunstone_peak_kml_path = result_geo_ssp["artifact"]["parts"][0]["data"]["filePath"]
        print(f"SunStonePeak KML path retrieved: {sunstone_peak_kml_path}")
    except (KeyError, IndexError, TypeError) as e:
        print(f"MVP Error: Failed to extract 'filePath' from GeoDataQueryAgent-MVP result for {element_sunstone_peak_id}.")
        print(f"Result was: {json.dumps(result_geo_ssp, indent=2)}")
        print(f"Error details: {e}")
        print("Aborting MVP run.")
        sys.exit(1)

    # --- Step 2: Get Cultural Links for SunStonePeak ---
    print("\nStep 2: Requesting cultural links for SunStonePeak from CultureDataQueryAgent-MVP...")
    culture_task_001_id = "culture_task_001"
    
    sunstone_peak_cultural_links_result = handle_culture_data_query_task(
        task_id=culture_task_001_id,
        request_type="get_cultural_links",
        element_name=element_sunstone_peak_id, # Using the @id
        ckg_filepath=ckg_filepath
    )

    if sunstone_peak_cultural_links_result.get("status") == "FAILED":
        error_message = sunstone_peak_cultural_links_result.get("error_message", "Unknown error from CultureDataQueryAgent.")
        # The Culture agent's FAILED artifact structure is slightly different from Geo for error text.
        # It puts error_message at top level, and artifact.parts[0].data.reason for some.
        # Let's prioritize error_message if available.
        if "artifact" in sunstone_peak_cultural_links_result and "parts" in sunstone_peak_cultural_links_result["artifact"]:
            try:
                reason = sunstone_peak_cultural_links_result["artifact"]["parts"][0]["data"]["reason"]
                error_message = f"{error_message} (Reason: {reason})"
            except (KeyError, IndexError, TypeError):
                pass # Stick with the top-level error_message
        print(f"MVP Error: CultureDataQueryAgent-MVP failed for {element_sunstone_peak_id} links. Reason: {error_message}")
        print("Aborting MVP run.")
        sys.exit(1)
    
    try:
        # Validate structure for critical data, though we pass the whole data part
        _ = sunstone_peak_cultural_links_result["artifact"]["parts"][0]["data"]["narrativeLinks"]
        print(f"Cultural links for SunStonePeak retrieved: {json.dumps(sunstone_peak_cultural_links_result['artifact']['parts'][0]['data'], indent=2)}")
    except (KeyError, IndexError, TypeError) as e:
        print(f"MVP Error: Failed to extract data from CultureDataQueryAgent-MVP result for {element_sunstone_peak_id} links.")
        print(f"Result was: {json.dumps(sunstone_peak_cultural_links_result, indent=2)}")
        print(f"Error details: {e}")
        print("Aborting MVP run.")
        sys.exit(1)


    # --- Step 3: Get Associated Symbol for RiverOfReflection ---
    print("\nStep 3: Requesting associated symbol for RiverOfReflection from CultureDataQueryAgent-MVP...")
    culture_task_002_id = "culture_task_002"
    element_river_reflection_id = "mvp:RiverOfReflection"

    river_reflection_symbol_association_result = handle_culture_data_query_task(
        task_id=culture_task_002_id,
        request_type="get_associated_symbol",
        element_name=element_river_reflection_id, # Using the @id
        ckg_filepath=ckg_filepath
    )

    if river_reflection_symbol_association_result.get("status") == "FAILED":
        error_message = river_reflection_symbol_association_result.get("error_message", "Unknown error from CultureDataQueryAgent.")
        if "artifact" in river_reflection_symbol_association_result and "parts" in river_reflection_symbol_association_result["artifact"]:
             try:
                reason = river_reflection_symbol_association_result["artifact"]["parts"][0]["data"]["reason"]
                error_message = f"{error_message} (Reason: {reason})"
             except (KeyError, IndexError, TypeError):
                pass
        print(f"MVP Error: CultureDataQueryAgent-MVP failed for {element_river_reflection_id} symbol. Reason: {error_message}")
        print("Aborting MVP run.")
        sys.exit(1)

    try:
        # Validate structure for critical data
        _ = river_reflection_symbol_association_result["artifact"]["parts"][0]["data"]["associatedSymbol"]
        print(f"Symbol association for RiverOfReflection retrieved: {json.dumps(river_reflection_symbol_association_result['artifact']['parts'][0]['data'], indent=2)}")
    except (KeyError, IndexError, TypeError) as e:
        print(f"MVP Error: Failed to extract data from CultureDataQueryAgent-MVP result for {element_river_reflection_id} symbol.")
        print(f"Result was: {json.dumps(river_reflection_symbol_association_result, indent=2)}")
        print(f"Error details: {e}")
        print("Aborting MVP run.")
        sys.exit(1)

    # --- Step 4: Perform Cross-Referencing ---
    print("\nStep 4: Requesting cross-referencing from BasicCrossReferencingAgent-MVP...")
    xref_task_001_id = "xref_task_001"
    # Element names for narrative search, as per spec 3.4
    element1_name_in_narrative = "Sun Stone Peak" 
    element2_name_in_narrative = "River of Reflection"

    # Prepare inputs for the cross-referencing agent
    # The agent expects the inner 'data' dictionary from the cultural agent results
    cultural_links_data_ssp = sunstone_peak_cultural_links_result['artifact']['parts'][0]['data']
    symbol_association_data_ror = river_reflection_symbol_association_result['artifact']['parts'][0]['data']

    result_xref = handle_cross_referencing_task(
        task_id=xref_task_001_id,
        geo_element1_filepath=sunstone_peak_kml_path,
        cultural_links_element1=cultural_links_data_ssp,
        symbol_association_element2=symbol_association_data_ror,
        narrative_filepath=narrative_filepath,
        element1_name_in_narrative=element1_name_in_narrative,
        element2_name_in_narrative=element2_name_in_narrative
    )

    if result_xref.get("status") == "FAILED": # FAILED here usually means file not found for narrative
        error_message = result_xref.get("artifact", {}).get("parts", [{}])[0].get("text", "Unknown error from BasicCrossReferencingAgent.")
        print(f"MVP Error: BasicCrossReferencingAgent-MVP failed. Reason: {error_message}")
        print("Aborting MVP run.")
        sys.exit(1)
    
    # --- Step 5: Output Final Result ---
    print("\nStep 5: Final Result from BasicCrossReferencingAgent-MVP:")
    try:
        final_text_output = result_xref["artifact"]["parts"][0]["text"]
        print(final_text_output) # This is the main success/failure message from XRef agent
    except (KeyError, IndexError, TypeError) as e:
        print(f"MVP Error: Failed to extract final text output from BasicCrossReferencingAgent-MVP result.")
        print(f"Result was: {json.dumps(result_xref, indent=2)}")
        print(f"Error details: {e}")
        print("Aborting MVP run.")
        sys.exit(1)

    print("\n--- TOE-MVP Orchestration Finished ---")

if __name__ == "__main__":
    # Check if required files exist before running
    required_files = [
        "ckg_mvp.jsonld",
        "PDN_MVP/narrative/sky_serpent_gift.txt",
        "PDN_MVP/geospatial/sun_stone_peak.kml", # Checked by Geo Agent, but good for TOE to be aware
        "PDN_MVP/geospatial/river_of_reflection.kml" # Checked by Geo Agent
    ]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("TOE-MVP Error: Required data files for the MVP run are missing.")
        for f in missing_files:
            print(f"  - Missing: {f}")
        print("Please ensure all PDN_MVP files and ckg_mvp.jsonld are correctly placed in the repository root.")
        # Specific check for agent scripts' existence
        agent_scripts = [
            "mvp_agents/geo_data_query_agent_mvp/geo_data_query_agent_mvp.py",
            "mvp_agents/culture_data_query_agent_mvp/culture_data_query_agent_mvp.py",
            "mvp_agents/basic_cross_referencing_agent_mvp/basic_cross_referencing_agent_mvp.py"
        ]
        missing_agents = [s for s in agent_scripts if not os.path.exists(s)]
        if missing_agents:
            print("\nTOE-MVP Error: One or more agent scripts are missing:")
            for s in missing_agents:
                print(f"  - Missing: {s}")
        sys.exit(1)
        
    main()

```

