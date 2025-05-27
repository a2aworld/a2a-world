# A2A Protocol Extensions for A2A World

## 1. Introduction

These protocol extensions are designed to facilitate communication and collaboration between autonomous agents within the A2A World. A2A World is a simulated environment where agents interact with shared data resources (the Data Nexus) and each other to discover, interpret, and understand complex geospatial and cultural phenomena. These extensions define standardized message formats for requesting data, sharing findings, proposing hypotheses, and coordinating tasks, enabling sophisticated emergent behaviors and collective intelligence.

## 2. General Message Structure

All messages exchanged within the A2A World adhere to a common wrapper structure to ensure consistency and provide essential metadata for message routing, tracking, and debugging.

```json
{
  "header": {
    "message_id": "uuid", // Unique identifier for this specific message
    "agent_id": "string", // Unique identifier of the sending agent
    "timestamp": "iso_datetime_string", // Time of message creation (UTC)
    "task_id": "uuid", // Optional: Identifier of the task this message relates to
    "protocol_version": "string" // e.g., "a2a_world_ext_v1.0"
  },
  "message_type": "string", // Specific type of the message (e.g., "GeospatialDataRequest")
  "payload": {
    // Message-specific fields
  }
}
```

**Key Header Fields:**

*   `message_id`: A universally unique identifier (UUID) for each message instance.
*   `agent_id`: The unique identifier of the agent originating the message.
*   `timestamp`: An ISO 8601 formatted datetime string indicating when the message was created.
*   `task_id`: An optional UUID linking the message to a specific task being orchestrated or worked on.
*   `protocol_version`: The version of the A2A World protocol extension being used.

## 3. Message Types for Data Nexus Interaction

These messages facilitate agent interaction with the A2A World's Data Nexus, which comprises geospatial information and a Cultural Knowledge Graph.

### 3.1. GeospatialDataRequest

*   **Purpose:** To request geospatial data from the Data Nexus.
*   **`message_type`:** `"GeospatialDataRequest"`
*   **Payload Fields:**
    *   `region_identifier`: `object` or `string` - Specifies the area of interest. Can be coordinates (e.g., GeoJSON polygon), a named administrative area, or a custom region ID.
    *   `data_type_requested`: `string` - Type of geospatial data needed (e.g., 'satellite_imagery', 'lidar_terrain', 'geological_map', 'elevation_model', 'vector_features').
    *   `resolution_preference`: `string` or `number` - Desired data resolution (e.g., 'high', 'medium', 'low', or specific meter/pixel value).
    *   `time_period_filter`: `object` (optional) - Specifies a time range for the data.
        *   `start_time`: `iso_datetime_string`
        *   `end_time`: `iso_datetime_string`
    *   `format_preference`: `string` (optional) - Preferred data format for delivery if applicable (e.g., 'GeoTIFF', 'NetCDF', 'GeoJSON').

### 3.2. GeospatialDataResponse

*   **Purpose:** To deliver requested geospatial data or report an issue.
*   **`message_type`:** `"GeospatialDataResponse"`
*   **Payload Fields:**
    *   `request_id`: `uuid` - The `message_id` of the corresponding `GeospatialDataRequest`.
    *   `status`: `string` - Outcome of the request ('success', 'failure', 'partial', 'pending').
    *   `data_format`: `string` (if status is 'success' or 'partial') - Format of the delivered data (e.g., 'GeoJSON', 'link_to_file:GeoTIFF', 'link_to_file:NetCDF').
    *   `data_payload`: `object` or `string` - The actual data if small enough (e.g., GeoJSON features), or a URI/link to download larger datasets.
    *   `metadata`: `object` (optional) - Additional information about the data, such as projection, acquisition date, resolution.
    *   `error_message`: `string` (if status is 'failure' or 'partial') - Description of the error or reason for partial data.

### 3.3. CulturalDataQuery

*   **Purpose:** To query the Cultural Knowledge Graph (CKG) within the Data Nexus.
*   **`message_type`:** `"CulturalDataQuery"`
*   **Payload Fields:**
    *   `query_type`: `string` - The type of query to perform (e.g., 'keyword_search', 'symbol_lookup', 'entity_retrieval', 'relationship_query', 'region_filter', 'temporal_filter').
    *   `query_parameters`: `object` - Parameters specific to the `query_type`.
        *   Example for `keyword_search`: `{ "keywords": ["jaguar", "temple"], "search_scope": ["artefacts", "myths"] }`
        *   Example for `symbol_lookup`: `{ "symbol_description": "spiral motif", "similarity_threshold": 0.8 }`
        *   Example for `region_filter`: `{ "region_identifier": "object", "spatial_relationship": "within" }` // region_identifier same as in GeospatialDataRequest
    *   `max_results`: `integer` - Maximum number of results to return.
    *   `result_format_preference`: `string` (optional) - Preferred format for results (e.g., 'linked_data_json', 'graphml_snippet').

### 3.4. CulturalDataResponse

*   **Purpose:** To return results from a Cultural Knowledge Graph query.
*   **`message_type`:** `"CulturalDataResponse"`
*   **Payload Fields:**
    *   `query_id`: `uuid` - The `message_id` of the corresponding `CulturalDataQuery`.
    *   `status`: `string` - Outcome of the query ('success', 'failure', 'no_results').
    *   `results`: `array` (if status is 'success') - A list of structured cultural data objects or knowledge graph snippets matching the query. Each object should have a stable URI or ID.
    *   `metadata`: `object` (optional) - Information about the query execution, like number of hits, query time.
    *   `error_message`: `string` (if status is 'failure') - Description of the error.

## 4. Message Types for Collaboration and Interpretation

These messages enable agents to share observations, propose interpretations, and collaboratively build understanding.

### 4.1. FindingBroadcast

*   **Purpose:** For an agent to share a discovery, observation, or a piece of processed information with other relevant agents.
*   **`message_type`:** `"FindingBroadcast"`
*   **Payload Fields:**
    *   `finding_id`: `uuid` - A newly generated unique identifier for this finding.
    *   `source_agent_id`: `string` - The `agent_id` of the agent reporting the finding (redundant with header but useful in payload for direct use).
    *   `finding_type`: `string` - Nature of the finding (e.g., 'potential_geospatial_pattern', 'cultural_element_reference', 'symbol_identification', 'anomaly_detection', 'data_correlation').
    *   `data_description`: `string` or `object` - A textual or structured description of the finding.
    *   `location_context`: `object` (optional) - Geospatial context of the finding (e.g., coordinates, reference to a `GeospatialDataResponse` `data_payload` region).
    *   `temporal_context`: `object` (optional) - Time period relevant to the finding.
        *   `start_time`: `iso_datetime_string`
        *   `end_time`: `iso_datetime_string`
    *   `confidence_score`: `float` (0.0-1.0) - The agent's confidence in the validity or significance of the finding.
    *   `supporting_evidence_links`: `array` of `string` (optional) - URIs or `message_id`s of data or messages that support this finding (e.g., link to specific satellite image, `GeospatialDataResponse` ID, CKG entry URI).
    *   `tags`: `array` of `string` (optional) - Keywords or tags to help categorize and route the finding.

### 4.2. HypothesisProposal

*   **Purpose:** For an agent to propose a specific interpretation, connection, or explanation based on one or more findings.
*   **`message_type`:** `"HypothesisProposal"`
*   **Payload Fields:**
    *   `hypothesis_id`: `uuid` - A newly generated unique identifier for this hypothesis.
    *   `proposing_agent_id`: `string` - The `agent_id` of the agent proposing the hypothesis.
    *   `hypothesis_statement`: `string` - A clear, textual description of the proposed hypothesis.
    *   `linked_findings`: `array` of `uuid` - A list of `finding_id`s that form the basis for this hypothesis.
    *   `supporting_arguments`: `string` or `object` - Textual or structured arguments explaining how the linked findings support the hypothesis.
    *   `initial_confidence_score`: `float` (0.0-1.0) - The agent's initial confidence in this hypothesis.
    *   `query_for_evidence`: `object` (optional) - A suggested query (e.g., for CKG or geospatial data) that could yield further evidence.

### 4.3. EvidenceSubmission

*   **Purpose:** For agents to submit evidence that supports or contradicts an existing hypothesis.
*   **`message_type`:** `"EvidenceSubmission"`
*   **Payload Fields:**
    *   `target_hypothesis_id`: `uuid` - The `hypothesis_id` of the hypothesis this evidence pertains to.
    *   `submitting_agent_id`: `string` - The `agent_id` of the agent submitting the evidence.
    *   `evidence_type`: `string` - Type of evidence ('supporting', 'contradicting').
    *   `evidence_description`: `string` or `object` - Description of the evidence and its relevance.
    *   `confidence_adjustment_factor`: `float` - A factor suggesting how this evidence might adjust the confidence in the hypothesis (e.g., +0.1 for supporting, -0.2 for contradicting). The actual update mechanism for hypothesis confidence is managed by a designated agent or consensus mechanism.
    *   `new_data_links`: `array` of `string` (optional) - Links to new data or `finding_id`s that constitute this evidence.

### 4.4. PareidoliaSuggestionRequest

*   **Purpose:** Specifically for Pareidolia Simulation Agents, to request an analysis of geospatial data for meaningful patterns based on cultural prompts.
*   **`message_type`:** `"PareidoliaSuggestionRequest"`
*   **Payload Fields:**
    *   `target_geospatial_data_id`: `string` or `uuid` - Identifier for the geospatial data to be analyzed (e.g., a `message_id` from a `GeospatialDataResponse` or a Data Nexus URI).
    *   `cultural_keywords_prompt`: `array` of `string` - Keywords to guide the pattern recognition (e.g., ["jaguar", "face", "serpent"]).
    *   `symbol_lexicon_references`: `array` of `string` (optional) - References to specific symbols or motifs from the CKG to look for.
    *   `sensitivity_level`: `float` (0.0-1.0, optional) - Indication of how aggressively the agent should look for patterns (higher means more, potentially less accurate, suggestions).

### 4.5. PareidoliaSuggestionResponse

*   **Purpose:** To provide potential pareidolic interpretations found in geospatial data.
*   **`message_type`:** `"PareidoliaSuggestionResponse"`
*   **Payload Fields:**
    *   `request_id`: `uuid` - The `message_id` of the corresponding `PareidoliaSuggestionRequest`.
    *   `status`: `string` - ('success', 'failure', 'no_patterns_found').
    *   `suggested_patterns`: `array` of `object` (if status is 'success') - List of potential patterns.
        *   `pattern_description`: `string` - Textual description of the perceived pattern.
        *   `confidence_score`: `float` (0.0-1.0) - The pareidolia agent's confidence in this specific suggestion.
        *   `location_in_data`: `object` - Coordinates or bounding box defining the pattern's location within the target geospatial data (e.g., GeoJSON).
        *   `matched_prompt_elements`: `array` of `string` (optional) - Which keywords or symbols from the request this pattern relates to.
    *   `error_message`: `string` (if status is 'failure').

## 5. Message Types for Task Orchestration (Conceptual)

These messages are for higher-level coordination of tasks among agents. The exact mechanisms for task allocation and management might be complex and handled by specialized orchestrator agents.

### 5.1. NewTaskAnnouncement

*   **Purpose:** To announce a new task that requires collaboration or capabilities from one or more agents.
*   **`message_type`:** `"NewTaskAnnouncement"`
*   **Payload Fields:**
    *   `task_id`: `uuid` - A newly generated unique identifier for this task.
    *   `task_description`: `string` - A detailed description of the task goals, objectives, and expected outcomes.
    *   `issuing_agent_id`: `string` - The agent (or system component) announcing the task.
    *   `required_capabilities`: `array` of `string` - List of agent skills or roles needed (e.g., 'geospatial_analysis', 'ckg_reasoning', 'pareidolia_simulation').
    *   `input_data_references`: `array` of `string` - Links or IDs to initial data required for the task.
    *   `deadline`: `iso_datetime_string` (optional) - Suggested completion deadline.
    *   `reward_criteria`: `string` (optional) - How successful task completion will be measured or rewarded.

### 5.2. TaskClaimOrBid

*   **Purpose:** For an agent to claim responsibility for an announced task or bid for it if a selection process is involved.
*   **`message_type`:** `"TaskClaimOrBid"`
*   **Payload Fields:**
    *   `task_id`: `uuid` - The identifier of the task being claimed or bid upon.
    *   `claiming_agent_id`: `string` - The `agent_id` of the agent making the claim/bid.
    *   `type`: `string` - 'claim' or 'bid'.
    *   `agent_capabilities_match_score`: `float` (0.0-1.0, optional for 'claim', usually required for 'bid') - Self-assessed score of how well the agent's capabilities match the task requirements.
    *   `proposed_plan`: `string` (optional for 'bid') - Brief outline of how the agent intends to tackle the task.
    *   `bid_details`: `object` (optional for 'bid') - Any specific terms or conditions for the bid.

## 6. Data Format Considerations

*   **Structured Data (Payloads, CKG objects):** JSON is the preferred format for all message payloads and structured data objects returned from the Cultural Knowledge Graph, due to its widespread support and ease of parsing.
*   **Geospatial Data:**
    *   For vector data or small raster excerpts, inline GeoJSON is acceptable within `data_payload`.
    *   For larger raster or vector datasets, messages should contain links (URIs) to files. Preferred formats for these linked files include:
        *   **GeoTIFF:** For raster imagery and gridded data.
        *   **NetCDF:** For multi-dimensional scientific data, including time-series geospatial data.
        *   **Cloud Optimized GeoTIFF (COG):** For efficient web access to raster data.
        *   **GeoPackage:** For vector features and raster maps.
*   **Links/URIs:** All links to external data or resources should be stable and accessible to the relevant agents.
*   **Timestamps:** ISO 8601 format (e.g., `YYYY-MM-DDTHH:MM:SS.sssZ`) should be used for all timestamps.

This document provides a foundational set of protocol extensions. As A2A World evolves, new message types and refinements to existing ones are anticipated.
