# A2A World: Pilot Agent Specifications

This document outlines the specifications for the initial set of pilot agents designed to operate within the A2A World environment. These agents will interact with simulated data sources (Planetary Data Nexus, Cultural Knowledge Graph) and collaborate by exchanging messages according to the `a2a_world_protocol_extensions.md`.

## 1. GeoDataQueryAgent

*   **Purpose:** To retrieve geospatial data from the Planetary Data Nexus (which will be a mock/simulated service at this stage).
*   **A2A Interactions:**
    *   **Receives (Listens for):**
        *   Direct `GeospatialDataRequest` message (as defined in `a2a_world_protocol_extensions.md`) from a human operator or a coordinating agent.
        *   (Future) `TaskAssignment` message from an orchestrator agent.
    *   **Sends:**
        *   `GeospatialDataRequest` messages to the Planetary Data Nexus.
    *   **Receives:**
        *   `GeospatialDataResponse` messages from the Planetary Data Nexus.
    *   **Sends:**
        *   `FindingBroadcast` messages to the A2A Collaboration Hub, announcing the availability and key characteristics of newly acquired geospatial data.
*   **Core Logic:**
    *   Validate incoming `GeospatialDataRequest` trigger messages or formulate new requests based on internal configuration/triggers.
    *   Construct well-formed `GeospatialDataRequest` messages compliant with `a2a_world_protocol_extensions.md` and `a2a_world_data_schemas.md` (`GeospatialRequestParams`).
    *   Interface with the (simulated) Planetary Data Nexus to submit requests.
    *   Process `GeospatialDataResponse` messages, handling success, failure, or partial data scenarios.
    *   (Simulated) Store or log the metadata of retrieved data (e.g., data type, resolution, region covered, access link/path if applicable).
    *   Generate and broadcast `FindingBroadcast` messages to inform other agents about the acquired data.
*   **Input Parameters (for triggering/configuration via direct `GeospatialDataRequest` or internal setup):**
    *   `target_area_definition`: An object matching the structure of `target_area` within the `GeospatialRequestParams` schema (e.g., `{ "type": "bounding_box", "coordinates": { "min_latitude": ..., "max_longitude": ... } }`).
    *   `data_layers_to_request`: A list of objects, each matching the structure of an item in the `data_layers_requested` list within the `GeospatialRequestParams` schema (e.g., `[{ "layer_type": "lidar_terrain", "desired_resolution": "1m" }]`).
*   **Output (example `FindingBroadcast` payload):**
    ```json
    {
      "header": {
        "message_id": "uuid_finding_geo_001",
        "agent_id": "GeoDataQueryAgent_01",
        "timestamp": "iso_datetime_string",
        "protocol_version": "a2a_world_ext_v1.0"
      },
      "message_type": "FindingBroadcast",
      "payload": {
        "finding_id": "uuid_finding_geo_001", // Can be same as message_id for simplicity here
        "source_agent_id": "GeoDataQueryAgent_01",
        "finding_type": "geospatial_data_acquired",
        "data_description": "Acquired LiDAR terrain data for Nazca Sector 3, 1m resolution.",
        "location_context": {
          "type": "named_feature_id", // From a2a_world_data_schemas.md (GeospatialRequestParams.target_area)
          "named_feature_id": "Nazca_Sector_3"
        },
        "temporal_context": { // Optional, if data is time-specific
            "start_time": "iso_datetime_string_of_data_acquisition",
            "end_time": "iso_datetime_string_of_data_acquisition"
        },
        "confidence_score": 1.0,
        "supporting_evidence_links": ["nexus_data_reference:/geospatial/Nazca_Sector_3/lidar_1m_20231026.tif"], // Internal reference or manifest
        "tags": ["geospatial", "lidar", "nazca", "terrain_data"]
      }
    }
    ```

## 2. CultureDataQueryAgent

*   **Purpose:** To retrieve cultural and symbolic information from the Cultural Knowledge Graph (CKG) and Symbolic Lexicon (mock/simulated services at this stage).
*   **A2A Interactions:**
    *   **Receives (Listens for):**
        *   Direct `CulturalDataQuery` message (as defined in `a2a_world_protocol_extensions.md`) from a human operator or another agent.
        *   `FindingBroadcast` messages (e.g., from `GeoDataQueryAgent`) which might prompt it to search for cultural information related to a newly announced location or feature.
    *   **Sends:**
        *   `CulturalDataQuery` messages to the CKG/Symbolic Lexicon.
    *   **Receives:**
        *   `CulturalDataResponse` messages from the CKG/Symbolic Lexicon.
    *   **Sends:**
        *   `FindingBroadcast` messages announcing relevant cultural findings, including keywords, symbols, and related locations or concepts.
*   **Core Logic:**
    *   Validate incoming `CulturalDataQuery` trigger messages or formulate queries based on other triggers (like a `FindingBroadcast`).
    *   Construct well-formed `CulturalDataQuery` messages compliant with `a2a_world_protocol_extensions.md`.
    *   Interface with the (simulated) CKG/Symbolic Lexicon.
    *   Process `CulturalDataResponse` messages, extracting key information.
    *   Analyze extracted information to identify significant cultural elements (myths, symbols, keywords, locations mentioned, temporal references).
    *   Generate and broadcast `FindingBroadcast` messages detailing these cultural findings.
*   **Input Parameters (for triggering/configuration via direct `CulturalDataQuery` or internal logic):**
    *   `query_parameters`: An object matching the structure of `query_parameters` within the `CulturalDataQuery` message payload (e.g., `{ "query_type": "keyword_search", "keywords": ["sky god", "mountain"], "search_scope": ["myth", "ritual"], "region_filter": { "region_identifier": {"type": "named_feature_id", "named_feature_id": "Andean_Region_General"} } }`).
    *   `target_context` (Optional): Information that triggered the query, such as the `finding_id` of a `FindingBroadcast` (e.g., a geospatial pattern discovery) to provide context for the cultural query.
*   **Output (example `FindingBroadcast` payload):**
    ```json
    {
      "header": {
        "message_id": "uuid_finding_culture_001",
        "agent_id": "CultureDataQueryAgent_01",
        "timestamp": "iso_datetime_string",
        "protocol_version": "a2a_world_ext_v1.0"
      },
      "message_type": "FindingBroadcast",
      "payload": {
        "finding_id": "uuid_finding_culture_001",
        "source_agent_id": "CultureDataQueryAgent_01",
        "finding_type": "cultural_element_retrieved",
        "data_description": "Retrieved myth 'The Serpent Mountain and the Sky God' mentioning a celestial alignment, associated with the Andean region.",
        "location_context": { // Derived from the cultural entry, if available
          "type": "region_id", // Example, could be more specific like a GeoJSON polygon if in the CKG
          "region_id": "Andean_Highlands_General"
        },
        "temporal_context": { // Derived from the cultural entry, if available
            "start_time": "approx_800_CE", // Example of less precise dating
            "end_time": "approx_1200_CE"
        },
        "confidence_score": 0.8, // Based on source reliability from CKG (e.g. CulturalEntry.confidence_score_of_data)
        "supporting_evidence_links": ["ckg_entry_uri:/cultural_entries/myth_serpent_mountain_012"],
        "tags": ["mythology", "andean", "sky_god", "serpent", "celestial_alignment"]
      }
    }
    ```

## 3. BasicCrossReferencingAgent

*   **Purpose:** To identify simple, predefined types of correlations between geospatial findings and cultural findings by listening to `FindingBroadcast` messages.
*   **A2A Interactions:**
    *   **Receives (Listens for):**
        *   `FindingBroadcast` messages from any agent, but primarily from `GeoDataQueryAgent` and `CultureDataQueryAgent`.
    *   **Sends:**
        *   `HypothesisProposal` messages to the A2A Collaboration Hub when a potential correlation is identified based on its configured rules.
*   **Core Logic:**
    *   Maintain a short-term memory of recent `FindingBroadcast` messages (or specific types of findings).
    *   For each incoming `FindingBroadcast`, compare its content against stored findings based on a set of `correlation_rules`.
    *   **Examples of simple correlation logic:**
        *   **Location Keyword/ID Match:**
            *   If a `geospatial_data_acquired` finding has `location_context.named_feature_id` = "X".
            *   And a `cultural_element_retrieved` finding has `location_context.region_id` = "X" (or a synonym/related ID defined in rules).
            *   And/OR if keywords in `data_description` of both findings match (e.g., "Giza Plateau").
        *   **Symbol-Shape Keyword Match (Conceptual for Basic Agent):**
            *   If a (future) `geospatial_pattern_detected` finding includes `pattern_description`: "spiral shape".
            *   And a `cultural_element_retrieved` finding (specifically a symbol) includes `data_description`: "spiral symbol" or `keywords` contains "spiral".
            *   And both findings share a broadly similar `location_context`.
        *   **Temporal Overlap:**
            *   If a `geospatial_data_acquired` finding has a `temporal_context` (e.g., archaeological dating of a structure).
            *   And a `cultural_element_retrieved` finding has a `temporal_context` for the same general region.
            *   And these time periods overlap significantly according to defined rules.
    *   If a correlation rule is met with sufficient confidence, formulate and send a `HypothesisProposal`.
*   **Input Parameters (for configuration):**
    *   `correlation_rules`: A structured list or set of rules defining what constitutes a notable cross-reference. Each rule would specify:
        *   Types of `FindingBroadcast` messages to compare (e.g., `geospatial_data_acquired` vs. `cultural_element_retrieved`).
        *   Fields to compare within the payloads (e.g., `location_context`, `data_description`, `tags`, `temporal_context`).
        *   Matching logic (e.g., exact match, keyword overlap, semantic similarity if advanced, date range overlap).
        *   Confidence score to assign to the hypothesis if the rule is triggered.
        *   Example Rule:
            ```json
            {
              "rule_name": "GeoCultural_LocationKeywordMatch",
              "finding_type_A": "geospatial_data_acquired",
              "finding_type_B": "cultural_element_retrieved",
              "match_logic": {
                "type": "AND",
                "conditions": [
                  { "field_A": "location_context.named_feature_id", "field_B": "location_context.region_id", "comparison": "equals_or_related" },
                  { "field_A": "data_description.keywords", "field_B": "data_description.keywords", "comparison": "common_elements_min_1" }
                ]
              },
              "hypothesis_confidence": 0.6
            }
            ```
*   **Output (example `HypothesisProposal` payload):**
    ```json
    {
      "header": {
        "message_id": "uuid_hypothesis_001",
        "agent_id": "BasicCrossReferencingAgent_01",
        "timestamp": "iso_datetime_string",
        "protocol_version": "a2a_world_ext_v1.0"
      },
      "message_type": "HypothesisProposal",
      "payload": {
        "hypothesis_id": "uuid_hypothesis_001",
        "proposing_agent_id": "BasicCrossReferencingAgent_01",
        "hypothesis_statement": "Potential link: Geospatial feature 'Nazca_Sector_3' (LiDAR data acquired) and Cultural entry 'Myth_Condor_Lines' both reference the 'Condor' symbol/concept in the Nazca region.",
        "linked_findings": ["uuid_finding_geo_001", "uuid_finding_culture_002"], // IDs of the broadcasts that triggered this
        "supporting_arguments": "Rule 'GeoCultural_LocationKeywordMatch' triggered: Shared location context 'Nazca_Sector_3' and keyword 'Condor' present in cultural data description related to the region.",
        "initial_confidence_score": 0.6, // Based on the rule's defined confidence
        "query_for_evidence": { // Optional suggestion for further investigation
            "query_type": "CulturalDataQuery",
            "query_parameters": {
                "query_type": "symbol_lookup",
                "symbol_description": "Condor",
                "region_filter": { "region_identifier": {"type": "named_feature_id", "named_feature_id": "Nazca_Sector_3"} }
            }
        }
      }
    }
    ```
