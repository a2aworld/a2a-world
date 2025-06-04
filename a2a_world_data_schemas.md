# A2A World Data Nexus: Core Data Schemas

This document outlines the core data schemas used within the A2A World's Planetary Data Nexus. These schemas define the structure for cultural knowledge, symbolic representations, and parameters for accessing geospatial data.

## 1. Cultural Knowledge Graph Schema

The Cultural Knowledge Graph (CKG) stores interconnected information about human cultures, myths, beliefs, and historical events.

**Object:** `CulturalEntry`

| Field Name                    | Data Type                                                                                   | Description                                                                                                | Notes                                                                 |
| ----------------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `entry_id`                    | STRING                                                                                      | Unique identifier for the cultural entry.                                                                  | e.g., UUID                                                            |
| `entry_type`                  | STRING (Enum)                                                                               | Type of cultural entry.                                                                                    | Values: 'myth', 'folklore', 'ritual', 'belief', 'linguistic_data', 'historical_event', 'celestial_observation' |
| `name_primary`                | STRING                                                                                      | Primary name or title of the entry.                                                                        |                                                                       |
| `names_alternative`           | LIST<STRING>                                                                                | Other known names, titles, or transliterations.                                                            |                                                                       |
| `description_short`           | STRING                                                                                      | A brief summary or abstract of the entry.                                                                  |                                                                       |
| `content_full`                | TEXT                                                                                        | The full textual content (e.g., the myth itself, detailed description of a ritual) or a link to an external resource. | Can be markdown or plain text. For external, use URI.                 |
| `source_references`           | LIST<STRING>                                                                                | Bibliography, citations, or links to primary/secondary source materials.                                   |                                                                       |
| `region_of_origin`            | STRING                                                                                      | Primary geographic region or area associated with the entry.                                               | e.g., "Nile Valley", "Andes Mountains", "Mesoamerica"                 |
| `associated_cultures`         | LIST<STRING>                                                                                | Specific cultures, ethnic groups, or communities associated with this entry.                               | e.g., "Ancient Egyptian", "Inca", "Maya"                              |
| `time_period_general`         | STRING                                                                                      | A general historical or archaeological period.                                                             | e.g., 'Bronze Age', 'Pre-Columbian', 'Classical Antiquity', 'Contemporary' |
| `date_range_specific`         | OBJECT { `start_date`: STRING, `end_date`: STRING }                                         | Specific date range in ISO 8601 format, if known.                                                          | Optional. `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SSZ`                      |
| `keywords`                    | LIST<STRING>                                                                                | Relevant keywords for searching and categorization.                                                        |                                                                       |
| `associated_symbols_ids`      | LIST<STRING>                                                                                | List of `symbol_id`s from the Symbolic Lexicon that are relevant to this entry.                            | Links to `SymbolEntry.symbol_id`                                      |
| `geospatial_tags`             | LIST<OBJECT>                                                                                | Specific geographic locations relevant to the entry.                                                       |                                                                       |
| &nbsp;&nbsp;`tag_name`        | &nbsp;&nbsp;STRING                                                                           | &nbsp;&nbsp;Descriptive name for the location tag.                                                          | &nbsp;&nbsp;e.g., "Sacred Grove", "Temple Site"                       |
| &nbsp;&nbsp;`latitude`        | &nbsp;&nbsp;FLOAT                                                                            | &nbsp;&nbsp;Latitude of the tagged location.                                                               | &nbsp;&nbsp;WGS84 decimal degrees                                       |
| &nbsp;&nbsp;`longitude`       | &nbsp;&nbsp;FLOAT                                                                            | &nbsp;&nbsp;Longitude of the tagged location.                                                              | &nbsp;&nbsp;WGS84 decimal degrees                                       |
| &nbsp;&nbsp;`radius_km`       | &nbsp;&nbsp;FLOAT                                                                            | &nbsp;&nbsp;Optional radius to define an area around the point.                                             | &nbsp;&nbsp;Optional                                                  |
| &nbsp;&nbsp;`description`     | &nbsp;&nbsp;STRING                                                                           | &nbsp;&nbsp;Brief description of the location's relevance to the entry.                                     | &nbsp;&nbsp;Optional                                                  |
| `temporal_tags`               | LIST<OBJECT>                                                                                | Specific points or periods in time relevant to the entry.                                                  |                                                                       |
| &nbsp;&nbsp;`tag_name`        | &nbsp;&nbsp;STRING                                                                           | &nbsp;&nbsp;Descriptive name for the temporal tag.                                                          | &nbsp;&nbsp;e.g., "Festival Date", "Eruption Event"                   |
| &nbsp;&nbsp;`event_date`      | &nbsp;&nbsp;STRING                                                                           | &nbsp;&nbsp;Specific date or date range (ISO 8601).                                                        |                                                                       |
| &nbsp;&nbsp;`description`     | &nbsp;&nbsp;STRING                                                                           | &nbsp;&nbsp;Brief description of the temporal tag's relevance.                                              | &nbsp;&nbsp;Optional                                                  |
| `semantic_links`              | LIST<OBJECT>                                                                                | Links to other `CulturalEntry` items, defining relationships between them.                                 |                                                                       |
| &nbsp;&nbsp;`linked_entry_id` | &nbsp;&nbsp;STRING                                                                           | &nbsp;&nbsp;The `entry_id` of the related cultural entry.                                                   |                                                                       |
| &nbsp;&nbsp;`relationship_type`| &nbsp;&nbsp;STRING                                                                           | &nbsp;&nbsp;Nature of the link.                                                                            | &nbsp;&nbsp;e.g., 'is_variant_of', 'influenced_by', 'describes_origin_of', 'critiques', 'references' |
| &nbsp;&nbsp;`description`     | &nbsp;&nbsp;STRING                                                                           | &nbsp;&nbsp;Optional description of the relationship.                                                       |                                                                       |
| `confidence_score_of_data`  | FLOAT                                                                                       | A score from 0.0 to 1.0 indicating the perceived accuracy, verifiability, or scholarly consensus of the entry's data. |                                                                       |
| `last_updated`                | TIMESTAMP                                                                                   | Timestamp of the last modification to this entry.                                                          | ISO 8601 DateTime string.                                             |

## 2. Symbolic Lexicon Schema

The Symbolic Lexicon stores information about individual symbols, motifs, and ideograms found across cultures and time periods.

**Object:** `SymbolEntry`

| Field Name                        | Data Type                                                              | Description                                                                                             | Notes                                                                     |
| --------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `symbol_id`                       | STRING                                                                 | Unique identifier for the symbol entry.                                                                 | e.g., UUID                                                                |
| `name_primary`                    | STRING                                                                 | Common or primary name of the symbol.                                                                   |                                                                           |
| `names_alternative`               | LIST<STRING>                                                           | Other known names or descriptive titles for the symbol.                                                 |                                                                           |
| `description_textual`             | TEXT                                                                   | Detailed description of the symbol's visual appearance, components, and established meanings or interpretations. | Can be markdown or plain text.                                            |
| `images`                          | LIST<OBJECT>                                                           | Visual representations of the symbol.                                                                   |                                                                           |
| &nbsp;&nbsp;`image_url_or_data`   | &nbsp;&nbsp;STRING                                                      | &nbsp;&nbsp;URL to an image file or base64 encoded image data.                                          |                                                                           |
| &nbsp;&nbsp;`caption`             | &nbsp;&nbsp;STRING                                                      | &nbsp;&nbsp;Brief description or context for the image.                                                 |                                                                           |
| &nbsp;&nbsp;`source`              | &nbsp;&nbsp;STRING                                                      | &nbsp;&nbsp;Origin or copyright information for the image.                                               | &nbsp;&nbsp;Optional                                                      |
| `cultural_origins`                | LIST<STRING>                                                           | Cultures or regions where this symbol is known to have originated or is prominently used.               | e.g., "Ancient Greek", "Celtic", "San People"                             |
| `associated_archetypes`           | LIST<STRING>                                                           | Archetypal concepts or figures the symbol is often associated with.                                     | e.g., 'trickster', 'great mother', 'world tree', 'sky god', 'hero'        |
| `related_concepts`                | LIST<STRING>                                                           | Abstract ideas, themes, or phenomena the symbol represents or is linked to.                             | e.g., 'fertility', 'creation', 'destruction', 'wisdom', 'power'           |
| `linked_cultural_entry_ids`       | LIST<STRING>                                                           | List of `entry_id`s from the Cultural Knowledge Graph where this symbol plays a role or is mentioned.   | Links to `CulturalEntry.entry_id`                                         |
| `cross_references_to_other_symbols`| LIST<OBJECT>                                                          | Relationships to other symbols within the lexicon.                                                      |                                                                           |
| &nbsp;&nbsp;`related_symbol_id`   | &nbsp;&nbsp;STRING                                                      | &nbsp;&nbsp;The `symbol_id` of the related symbol.                                                      |                                                                           |
| &nbsp;&nbsp;`relationship_type`   | &nbsp;&nbsp;STRING                                                      | &nbsp;&nbsp;Nature of the relationship.                                                                  | &nbsp;&nbsp;e.g., 'is_variant_of', 'is_component_of', 'is_opposite_to', 'evolved_from' |
| `first_known_appearance`          | STRING                                                                 | Context of the symbol's earliest known appearance.                                                      | e.g., 'Ancient Egypt, Old Kingdom', 'Lascaux Caves', 'Indus Valley Civilization' |
| `last_updated`                    | TIMESTAMP                                                              | Timestamp of the last modification to this entry.                                                       | ISO 8601 DateTime string.                                                 |

## 3. Geospatial Data Access Request Parameters Schema

This schema details the structure of parameters for requesting geospatial data from the Data Nexus. It complements the `GeospatialDataRequest` message defined in `a2a_world_protocol_extensions.md`.

**Object:** `GeospatialRequestParams`

| Field Name                  | Data Type                                                                  | Description                                                                                                     | Notes                                                                                                                               |
| --------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `request_id`                | STRING                                                                     | Unique identifier for this request, used to correlate with the response.                                        | Should match `message_id` in the A2A protocol.                                                                                      |
| `agent_id`                  | STRING                                                                     | Identifier of the agent making the request.                                                                     | Should match `agent_id` in the A2A protocol.                                                                                        |
| `target_area`               | OBJECT                                                                     | Defines the geographical area of interest for the data request.                                                 |                                                                                                                                     |
| &nbsp;&nbsp;`type`          | &nbsp;&nbsp;STRING (Enum)                                                   | &nbsp;&nbsp;The method used to define the target area.                                                          | &nbsp;&nbsp;Values: 'point_radius', 'bounding_box', 'polygon', 'named_feature_id'                                                   |
| &nbsp;&nbsp;`coordinates`   | &nbsp;&nbsp;ANY                                                             | &nbsp;&nbsp;The coordinate data, structure depends on `type`.                                                    | &nbsp;&nbsp;For 'point_radius': `{ "latitude": float, "longitude": float, "radius_km": float }` <br> &nbsp;&nbsp;For 'bounding_box': `{ "min_latitude": float, "min_longitude": float, "max_latitude": float, "max_longitude": float }` <br> &nbsp;&nbsp;For 'polygon': GeoJSON Polygon coordinates array. |
| &nbsp;&nbsp;`named_feature_id` | &nbsp;&nbsp;STRING                                                        | &nbsp;&nbsp;Identifier for a pre-defined named geographic feature within the Nexus.                               | &nbsp;&nbsp;e.g., 'NazcaLines_Sector1', 'AmazonBasin_ZoneX'. Used if `type` is 'named_feature_id'.                                |
| `data_layers_requested`     | LIST<OBJECT>                                                               | Specifies the types and characteristics of geospatial data layers being requested.                              |                                                                                                                                     |
| &nbsp;&nbsp;`layer_type`    | &nbsp;&nbsp;STRING (Enum)                                                   | &nbsp;&nbsp;The specific type of geospatial data layer.                                                         | &nbsp;&nbsp;Values: 'optical_satellite', 'radar_satellite', 'lidar_terrain', 'geological_map', 'archaeological_site_db', 'magnetic_field_data', 'ocean_floor_map', 'bathymetry', 'gravity_anomaly' |
| &nbsp;&nbsp;`desired_resolution` | &nbsp;&nbsp;STRING                                                      | &nbsp;&nbsp;Target resolution for the data.                                                                     | &nbsp;&nbsp;e.g., '0.5m', '10m', '1km', 'as_available'. The Nexus will attempt to match or provide the closest available.           |
| &nbsp;&nbsp;`spectral_bands`| &nbsp;&nbsp;LIST<STRING>                                                   | &nbsp;&nbsp;Specific spectral bands required, for multispectral or hyperspectral data.                           | &nbsp;&nbsp;Optional. e.g., ['visible_red', 'nir', 'swir_1']. Standard names preferred.                                             |
| &nbsp;&nbsp;`time_period_filter` | &nbsp;&nbsp;OBJECT                                                      | &nbsp;&nbsp;Filters data based on acquisition time.                                                             | &nbsp;&nbsp;Optional.                                                                                                               |
| &nbsp;&nbsp;&nbsp;&nbsp;`start_date` | &nbsp;&nbsp;&nbsp;&nbsp;STRING                                      | &nbsp;&nbsp;&nbsp;&nbsp;Start of the period (ISO 8601).                                                         |                                                                                                                                     |
| &nbsp;&nbsp;&nbsp;&nbsp;`end_date`   | &nbsp;&nbsp;&nbsp;&nbsp;STRING                                      | &nbsp;&nbsp;&nbsp;&nbsp;End of the period (ISO 8601).                                                           |                                                                                                                                     |
| &nbsp;&nbsp;&nbsp;&nbsp;`filter_type`| &nbsp;&nbsp;&nbsp;&nbsp;STRING (Enum)                               | &nbsp;&nbsp;&nbsp;&nbsp;How to apply the date filter.                                                           | &nbsp;&nbsp;&nbsp;&nbsp;Values: 'exact_match' (data must be within this window), 'overlap' (any data overlapping this window), 'latest_available' (within or before window, newest first). |
| `output_format_preference`  | STRING                                                                     | Preferred format for the delivered data.                                                                        | e.g., 'GeoJSON', 'GeoTIFF_link', 'NetCDF_link', 'COG_link'. The Nexus will try to accommodate.                                    |
| `priority`                  | INTEGER                                                                    | Optional priority level for the request (e.g., 1-5, higher is more important).                                  | Used by the Data Nexus to manage request queues if necessary.                                                                       |

## 4. Standard Task Artifact Schemas

This section defines standardized structures for common complex `Artifact` payloads, often used within a `DataPart` of an A2A `Artifact` object.

### 4.1. GrandChallengeSubmissionArtifact

* **Purpose:** Defines the structure for an agent's submission to a "Grand Challenge," such as the "A2A World Quantum Grand Challenge." This structure is designed to align with comprehensive submission requirements, like those of the XPRIZE Quantum Applications.
* **Root Type:** Typically used as the content of a `DataPart` within an `Artifact`.
* **Schema Definition:**
    ```json
    {
      "submission_title": "string",
      "submitting_agent_id": "string",
      "submission_timestamp": "iso_datetime_string",
      "grand_challenge_id": "string",
      "overall_summary_abstract": "string",

      "section_1_problem_statement_and_scope": {
        "general_problem_description": "string",
        "specific_cs_problem_formulation": "string",
        "relevance_to_societal_benefit": {
          "application_area": "string",
          "un_sdg_alignment": ["string"], // Optional list of UN SDG codes
          "detailed_argument": "string"
        },
        "supporting_documents_uri": ["string"] // Optional: Links to FileParts
      },

      "section_2_impact_on_problem_area": {
        "bottleneck_addressed": "string",
        "projected_real_world_impact": "string",
        "quantified_change_metrics": [ // Aligns with Phase II expectations
          {
            "metric_name": "string",
            "current_baseline": "string",
            "projected_quantum_impact": "string",
            "confidence_level": "float"
          }
        ],
        "expert_validation_references_uri": ["string"] // Optional: Links to FileParts
      },

      "section_3_quantum_advantage_demonstration": { // Aligns with XPRIZE Req. 3
        "quantum_algorithm_description_uri": "string", // Link to a FilePart
        "asymptotic_runtime_analysis": {
          "gate_complexity": "string",
          "space_complexity_qubits": "string"
        },
        "classical_algorithm_comparison": {
          "best_known_classical_approach": "string",
          "classical_runtime_analysis": "string",
          "overall_quantum_speedup_justification": "string"
        },
        "system_parameters_and_approximations": "string"
      },

      "section_4_classical_benchmarking": { // Aligns with XPRIZE Req. 4
        "classical_implementation_details_uri": "string", // Optional: Link to FilePart
        "comparative_performance_data_uri": "string", // Link to FilePart
        "argument_for_quantum_necessity_over_classical_approximation": "string"
      },

      "section_5_viability_analysis": { // Aligns with XPRIZE Req. 5
        "target_problem_sizes_for_impact": "string",
        "quantum_resource_estimation": {
          "architecture_assumptions": "string",
          "logical_qubits_required": "integer",
          "t_gate_count_or_equivalent": "integer", // Or other dominant gate
          "circuit_depth": "integer",
          "required_circuit_repetitions": "integer",
          "estimated_runtime_on_target_architecture": "string"
          // Include specific NISQ details if applicable
        },
        "projected_timeline_to_impact_assessment": "string" // Considers "time value of impact"
      },

      "section_6_novelty_of_contribution": { // Aligns with XPRIZE Req. 6
        "statement_of_novelty": "string",
        "comparison_to_established_methods": "string",
        "thought_delta_magnitude": "string" // Justification of conceptual advance
      },

      "appendices_and_supplementary_materials_uris": [ // Links to other FileParts
        "string" // e.g., "uri:/artifacts/quantnaza_gc001/raw_simulation_data.zip"
      ]
    }

## 5. Standard Learning Packet Schema

This section defines the structure for "Validated Learning Packets" used within the "A2A Collective Learning & Enhancement Loop"[cite: 11].

### 5.1. ValidatedLearningPacket

* **Purpose:** Encapsulates a validated piece of knowledge or skill enhancement for sharing and integration by agents. It can represent diverse types of learnings like model weights, procedural scripts, knowledge graph relationships, or effective methodologies.
* **Root Type:** Typically used as the content of a `DataPart` within an `Artifact` referenced by a `ValidatedLearningBroadcast` message.
* **Schema Definition:**
    ```json
    {
      "learning_packet_id": "string",
      "learning_type": "string", // Enum: "ModelWeights", "ProceduralScript", "KnowledgeGraphRelationship", "AnalyticalMethodology", "OptimizationTechnique", "IdentifiedPattern", "AntiPattern", "SimulationParameterSet", "EfficientWorkflow", "Other"
      "learning_format_version": "string",
      "title": "string",
      "description": "string",
      "content": {
        // Flexible content based on learning_type. Examples:
        // For "ModelWeights": { "model_architecture_id": "string", "weights_uri": "string", "hyperparameters": {} }
        // For "ProceduralScript": { "script_language": "string", "script_uri": "string", "dependencies": ["string"] }
        // For "KnowledgeGraphRelationship": { "subject_uri": "string", "predicate_uri": "string", "object_uri_or_literal": "any", "graph_context_uri": "string" }
      },
      "metadata": {
        "originating_agent_id": "string",
        "discovery_timestamp": "iso_datetime_string",
        "context_of_applicability": {
          "description": "string",
          "relevant_task_types": ["string"], // Optional
          "relevant_data_modalities": ["string"], // Optional
          "known_limitations_or_risks": "string" // Optional
        },
        "validation_details": { // From "A2A-Veritas"
          "validation_status": "string", // Enum
          "validator_agent_ids_or_service_id": ["string"], // Tying to "A2A-TrustFabric"
          "validation_report_uri": "string", // Link to ValidationReportMessage Artifact
          "validation_timestamp": "iso_datetime_string",
          "confidence_score_of_learning": { "score_value": "float", "confidence_scale_id": "string" }
        },
        "version_history_uri": "string", // Optional
        "keywords_and_tags": ["string"]
      }
    }

This document provides the foundational schemas for data within the A2A World Planetary Data Nexus. These schemas are expected to evolve as the system matures.
