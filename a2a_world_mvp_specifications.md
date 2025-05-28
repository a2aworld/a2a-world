# A2A World: Minimum Viable Product (MVP) Specifications

This document outlines the specifications for the Minimum Viable Product (MVP) of the A2A World system. The MVP is designed to demonstrate core functionalities using a limited but illustrative dataset, focusing on the interaction between pilot agents, foundational data services, and a basic visualization interface.

## 1. MVP Data Subset ("Mini-Storybook")

### 1.1. Overview

This MVP Data Subset, nicknamed the "Mini-Storybook," represents a small, tightly interconnected collection of hypothetical geospatial, visual, and narrative elements. It is designed to provide a concrete basis for testing the core functionalities of the A2A World MVP, including agent interactions, data queries, and basic hypothesis generation.

These elements are intended as placeholders. For the actual MVP implementation, they should be refined and replaced with specific, real-world data extracted from Om Kundalini's "Geospatial Storybook of Human Heritage" and the "Heaven on Earth As Above, So Below.kml" file, as provided and curated by the project visionary. The goal here is to illustrate the *types* of data and the *nature of connections* the MVP will handle.

### 1.2. Geospatial Elements (Derived from KML concepts)

These elements would be stored and served by the Foundational Match Service (FMS), derived from KML data.

*   **Element G1: "The Sun Stone Peak"**
    *   **Description:** A prominent mountain peak, significant rock outcrop, or a specific point on a structure that has a notable solar alignment, particularly at sunrise on a solstice or equinox.
    *   **Hypothetical KML Snippet Idea:**
        ```xml
        <Placemark>
          <name>Sun Stone Peak</name>
          <description>A key peak that aligns with the solstice sunrise, believed to be a focal point for ancient solar rituals.</description>
          <Point>
            <coordinates>-71.12345, -13.98765, 4500</coordinates> <!-- Example: Lon, Lat, Alt. To be supplied by visionary. -->
          </Point>
        </Placemark>
        ```
    *   **Significance for MVP:** Represents a primary, precisely located geographic point of interest with inherent symbolic (solar) significance. This would be an entry in the FMS.

*   **Element G2: "The River of Reflection"**
    *   **Description:** A specific stretch of a river, ancient canal, or natural waterway that is thematically and/or spatially linked to the "Sun Stone Peak." It might be where the light from the peak falls or aligns, or it could have symbolic importance in relation to the peak.
    *   **Hypothetical KML Snippet Idea:**
        ```xml
        <Placemark>
          <name>River of Reflection - Sacred Bend</name>
          <description>This specific bend in the river is said to capture the first light from Sun Stone Peak during the solstice. The pathway of the river is often likened to a serpent.</description>
          <LineString>
            <coordinates>
              -71.12300,-13.98700,4450
              -71.12250,-13.98650,4448
              -71.12200,-13.98600,4445
              -71.12150,-13.98550,4440 <!-- Example coordinates. To be supplied by visionary. -->
            </coordinates>
          </LineString>
        </Placemark>
        ```
    *   **Significance for MVP:** Represents a linear geographic feature connected to G1, also stored in the FMS.

### 1.3. Visual Elements

These elements would be referenced in relevant data structures (e.g., `CulturalEntry` or `SymbolEntry` in the CKG, or potentially linked from FMS entries if appropriate).

*   **Element V1: "Solstice Sunrise at Sun Stone Peak"**
    *   **Description:** An image (photograph, artist's impression, or diagram) depicting or symbolizing the sun aligning with or emerging from behind "Sun Stone Peak" (G1) during a solstice or other significant astronomical event.
    *   **Placeholder Filename:** `sunstone_sunrise_solstice.jpg`
    *   **Link to Geospatial:** Directly associated with Element G1 ("The Sun Stone Peak").
    *   **Significance for MVP:** Provides a visual anchor for the phenomenon at G1.

*   **Element V2: "River Serpent Glyph"**
    *   **Description:** An image of a petroglyph, geoglyph, or artistic motif found near or thematically representing the "River of Reflection" (G2). The motif might depict a serpent, a winding water symbol, or a constellation associated with the river.
    *   **Placeholder Filename:** `river_serpent_glyph_motif.png`
    *   **Link to Geospatial:** Associated with Element G2 ("The River of Reflection").
    *   **Significance for MVP:** Offers a symbolic visual representation linked to G2 and the narrative.

### 1.4. Narrative Element

This element would primarily reside within the Cultural Knowledge Graph (CKG) as a `CulturalEntry`.

*   **Element N1: "The Sky Serpent's Gift"**
    *   **Hypothetical Text Snippet (as `CulturalEntry.content_full`):**
        > "In the age of the ancestors, a pact was made with the Sky. At the dawn of the longest day, light from the Sun Stone Peak touches the River of Reflection, awakening the great Sky Serpent whose shimmering coils trace the water's path through the valley. The Serpent, nourished by the sun's first kiss upon the waters, then bestows the gift of wisdom and cyclical renewal upon the land and its people."
    *   **Significance for MVP:** Provides a brief, rich narrative that explicitly links G1 ("Sun Stone Peak") and G2 ("River of Reflection"). It also introduces the "Sky Serpent" concept, which connects to V2 ("River Serpent Glyph").

### 1.5. Documented Interconnections for MVP Analysis

These are the explicit links that the MVP system, particularly the `BasicCrossReferencingAgent-MVP`, should be able to identify and articulate based on the data subset.

*   **Connection 1 (Geo-Narrative):**
    *   **Statement:** "Sun Stone Peak" (G1) is mentioned in "The Sky Serpent's Gift" (N1) as the location of the solstice sunrise ("light from the Sun Stone Peak").
    *   **Data Points Involved:** `FoundationalMatchEntry` for G1 (from FMS), `CulturalEntry` for N1 (from CKG).

*   **Connection 2 (Geo-Visual-Narrative):**
    *   **Statement:** The "River of Reflection" (G2) is linked to the "Sky Serpent" in "The Sky Serpent's Gift" (N1) ("Sky Serpent whose shimmering coils trace the water's path"). The "River Serpent Glyph" (V2) is a visual representation of this serpent, associated with the river.
    *   **Data Points Involved:** `FoundationalMatchEntry` for G2 (from FMS), `CulturalEntry` for N1 (from CKG), `SymbolEntry` or `CulturalEntry` (linking to V2 image and G2).

*   **MVP Analytical Goal (Example for `BasicCrossReferencingAgent-MVP`):**
    *   The agent, upon processing information related to G1, G2, N1, and V2 (e.g., via `FindingBroadcast` messages or direct queries), should aim to identify and report a hypothesis similar to:
        > "Hypothesis: The narrative 'The Sky Serpent's Gift' (N1) describes a solar event at 'Sun Stone Peak' (G1) that illuminates the 'River of Reflection' (G2). This river is associated with a 'Sky Serpent' concept, which is visually represented by the 'River Serpent Glyph' (V2) found in proximity to G2. This suggests a cohesive symbolic link between the peak, the river, the narrative, and the glyph, centered around a solar event and serpent symbolism."

This "Mini-Storybook" provides a focused, manageable dataset to test the end-to-end flow of information and basic analytical capabilities of the A2A World MVP.

## 2. PDN-MVP and CKG-MVP/Symbolic Lexicon-MVP Specifications

This section defines the simple storage and representation for the MVP's "Mini-Storybook" data.

### 2.1. Planetary Data Nexus - MVP (PDN-MVP)

*   **Purpose:** To provide a basic storage location for the raw data files of the "Mini-Storybook." For the MVP, this will be a simplified stand-in for the more complex Planetary Data Nexus.
*   **Storage Mechanism:** A simple local directory structure <b>accessible</b> to the MVP agents.
*   **Directory Structure Example:**
    ```
    PDN_MVP/
    ├── geospatial/
    │   ├── sun_stone_peak.kml  // Hypothetical KML for Element G1 (actual KML content TBD by visionary)
    │   └── river_of_reflection.kml // Hypothetical KML for Element G2 (actual KML content TBD by visionary)
    ├── visual/
    │   ├── sunstone_sunrise.jpg    // Element V1 (placeholder or actual image)
    │   └── river_serpent_glyph.png // Element V2 (placeholder or actual image)
    └── narrative/
        └── sky_serpent_gift.txt    // Element N1 (text file containing the narrative)
    ```
*   **Data Access for MVP Agents:**
    *   For the MVP, the `GeoDataQueryAgent-MVP` will not directly interact with these KML files in the PDN-MVP. Instead, it will receive `FoundationalMatchEntry` objects from the `FMS-MVP` (which is presumed to have processed these KMLs).
    *   Visual and narrative file paths stored in the `CKG-MVP` (see below) will point to files within this PDN-MVP structure (e.g., `PDN_MVP/visual/sunstone_sunrise.jpg`). Agents needing to "access" these (like the Visualization Interface, or a future agent that might perform image analysis) would use these paths.

### 2.2. Cultural Knowledge Graph - MVP (CKG-MVP) & Symbolic Lexicon - MVP

*   **Purpose:** To provide a minimal, structured representation of the entities, symbols, and relationships within the "Mini-Storybook" data subset for MVP agents (primarily `CultureDataQueryAgent-MVP`) to query. This also serves as the Symbolic Lexicon for the MVP.
*   **Format:** A single JSON-LD (JSON for Linking Data) file. This allows for expressing linked data concepts without requiring a full triple store database for the MVP.
*   **Filename:** `ckg_mvp.jsonld`
*   **Mini-Ontology (Implicit in JSON-LD context):**
    *   **Base URI:** `http://a2a.world/ontology/mvp#` (Hypothetical)
    *   **Classes (Examples - can be implicit or explicit via `rdf:type`):** `mvp:GeospatialFeature`, `mvp:VisualElement`, `mvp:NarrativeElement`, `mvp:Symbol`, `mvp:Event`
    *   **Properties (Examples - defined in `@context`):**
        *   `rdfs:label` (used as `hasName`)
        *   `rdfs:comment` (used as `hasDescription`)
        *   `mvp:locatedAtKML` (points to KML file path in PDN-MVP, type `@id`)
        *   `mvp:imageFile` (points to image file path in PDN-MVP, type `@id`)
        *   `mvp:textFile` (points to text file path in PDN-MVP, type `@id`)
        *   `mvp:associatedWithVisual` (links a feature to a visual element, type `@id`)
        *   `mvp:depictsSymbol` (links a visual element to a symbol, type `@id`)
        *   `mvp:mentionsFeature` (links a narrative to a geospatial feature, type `@id`, can be a list)
        *   `mvp:describesEvent` (links a narrative to an event, type `@id`)
        *   `mvp:mentionsSymbol` (links a narrative to a symbol, type `@id`)
*   **Instance Data (Illustrative JSON-LD Content for `ckg_mvp.jsonld`):**
    ```json
    {
      "@context": {
        "mvp": "http://a2a.world/ontology/mvp#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "hasName": "rdfs:label",
        "hasDescription": "rdfs:comment",
        "locatedAtKML": {"@id": "mvp:locatedAtKML", "@type": "@id"},
        "imageFile": {"@id": "mvp:imageFile", "@type": "@id"},
        "textFile": {"@id": "mvp:textFile", "@type": "@id"},
        "associatedWithVisual": {"@id": "mvp:associatedWithVisual", "@type": "@id"},
        "depictsSymbol": {"@id": "mvp:depictsSymbol", "@type": "@id"},
        "mentionsFeature": {"@id": "mvp:mentionsFeature", "@type": "@id"},
        "describesEvent": {"@id": "mvp:describesEvent", "@type": "@id"},
        "mentionsSymbol": {"@id": "mvp:mentionsSymbol", "@type": "@id"}
      },
      "@graph": [
        {
          "@id": "mvp:SunStonePeak",
          "@type": "mvp:GeospatialFeature",
          "hasName": "The Sun Stone Peak",
          "hasDescription": "A prominent mountain peak significant at solstice sunrise. (Corresponds to FMS Entry G1)",
          "locatedAtKML": "PDN_MVP/geospatial/sun_stone_peak.kml",
          "associatedWithVisual": "mvp:SolsticeSunriseImage"
        },
        {
          "@id": "mvp:RiverOfReflection",
          "@type": "mvp:GeospatialFeature",
          "hasName": "The River of Reflection",
          "hasDescription": "A stretch of river thematically linked to Sun Stone Peak. (Corresponds to FMS Entry G2)",
          "locatedAtKML": "PDN_MVP/geospatial/river_of_reflection.kml",
          "associatedWithVisual": "mvp:RiverSerpentGlyphImage"
        },
        {
          "@id": "mvp:SolsticeSunriseImage",
          "@type": "mvp:VisualElement",
          "hasName": "Solstice Sunrise at Sun Stone Peak (V1)",
          "imageFile": "PDN_MVP/visual/sunstone_sunrise.jpg"
        },
        {
          "@id": "mvp:RiverSerpentGlyphImage",
          "@type": "mvp:VisualElement",
          "hasName": "River Serpent Glyph (V2)",
          "imageFile": "PDN_MVP/visual/river_serpent_glyph.png",
          "depictsSymbol": "mvp:SkySerpentSymbol"
        },
        {
          "@id": "mvp:SkySerpentSymbol",
          "@type": "mvp:Symbol",
          "hasName": "Sky Serpent"
        },
        {
          "@id": "mvp:SkySerpentGiftNarrative",
          "@type": "mvp:NarrativeElement",
          "hasName": "The Sky Serpent's Gift (N1)",
          "textFile": "PDN_MVP/narrative/sky_serpent_gift.txt",
          "mentionsFeature": ["mvp:SunStonePeak", "mvp:RiverOfReflection"],
          "describesEvent": "mvp:SolsticeGiftEvent",
          "mentionsSymbol": "mvp:SkySerpentSymbol"
        },
        {
          "@id": "mvp:SolsticeGiftEvent",
          "@type": "mvp:Event",
          "hasDescription": "The Sky Serpent bestowing wisdom, triggered by solstice sunrise at Sun Stone Peak and River of Reflection."
        }
      ]
    }
    ```
*   **Data Access for MVP Agents:**
    *   The `CultureDataQueryAgent-MVP` will be responsible for reading and parsing this `ckg_mvp.jsonld` file.
    *   For the MVP, "querying" will involve simple lookups within the parsed JSON-LD structure (e.g., finding an entity by its `@id`, iterating through the `@graph` to find entities with specific properties or `rdf:type`). No complex SPARQL or graph database queries are required for the MVP.
    *   The agent will then translate the relevant parts of this JSON-LD data into `FindingBroadcast` messages, using the A2A World protocol schemas.

This simplified CKG-MVP and PDN-MVP will provide the necessary backend data context for the pilot agents to perform their basic interactions and for the `BasicCrossReferencingAgent-MVP` to achieve the MVP analytical goal outlined in Section 1.5.

## 3. MVP Agent Designs

This section specifies the three agent types for the MVP: `GeoDataQueryAgent-MVP`, `CultureDataQueryAgent-MVP`, and `BasicCrossReferencingAgent-MVP`.

### 3.1. Common MVP Agent Assumptions

*   **A2A Communication:** For the MVP, agents will simulate A2A message exchange. Actual HTTP/A2A protocol implementation is deferred. The TOE-MVP script (Task Orchestration Engine - MVP) will simulate `AssignTask` by directly calling a function on the agent, and agents will return results directly (e.g., as structured Python dictionaries or objects).
*   **Capabilities:** Agent capabilities are predefined and understood by the TOE-MVP script. There will be no dynamic registration or discovery of agent capabilities beyond the TOE-MVP script potentially reading a simple configuration file (see Hub-MVP in Section 4) that lists available agents and their basic function signatures.
*   **Error Handling:** Error handling will be minimal for the MVP (e.g., file not found if `ckg_mvp.jsonld` is missing). Robust error handling is deferred.

### 3.2. GeoDataQueryAgent-MVP

*   **MVP Purpose:** To provide the file paths to the predefined geospatial data elements (G1: "SunStonePeak", G2: "RiverOfReflection") from the PDN-MVP. In a full system, this agent would query the FMS-MVP. For the MVP, it will directly return paths, simulating a successful FMS query for these specific items.
*   **Core Logic:**
    1.  Receives a simulated `AssignTask` request from TOE-MVP specifying which geospatial element is needed (e.g., a task like `"GetDataFor:SunStonePeak"` or `"GetDataFor:RiverOfReflection"`).
    2.  Based on the request, constructs the known file path within the `PDN_MVP/geospatial/` directory.
        *   If "SunStonePeak" is requested, path is `PDN_MVP/geospatial/sun_stone_peak.kml`.
        *   If "RiverOfReflection" is requested, path is `PDN_MVP/geospatial/river_of_reflection.kml`.
    3.  Returns a simulated `SubmitTaskResult` containing an "Artifact" with a `DataPart` that specifies the element name and its file path.
*   **Simulated A2A Interaction (Illustrative for internal TOE-MVP logic):**
    *   **Task from TOE-MVP:** `TaskID: geo_task_001`
        *   **Input to Agent Function (example):** `{"requestType": "get_geospatial_path", "elementName": "SunStonePeak"}`
    *   **Result from Agent Function (example):**
        ```python
        {
            "task_id": "geo_task_001",
            "status": "COMPLETED",
            "artifact": {
                "parts": [
                    {"data": {"elementName": "SunStonePeak", "filePath": "PDN_MVP/geospatial/sun_stone_peak.kml"}}
                ]
            }
        }
        ```

### 3.3. CultureDataQueryAgent-MVP

*   **MVP Purpose:** To query the `ckg_mvp.jsonld` file for information related to elements from the "Mini-Storybook" data subset.
*   **Core Logic:**
    1.  Receives a simulated `AssignTask` request from TOE-MVP (e.g., a task like `"GetDataFor:SunStonePeak_NarrativeLinks"` or `"GetDataFor:RiverOfReflection_Symbol"`).
    2.  Loads and parses the `ckg_mvp.jsonld` file into an internal data structure (e.g., a list of dictionaries).
    3.  Performs a simple search/filter operation on the parsed JSON-LD data based on the request. This involves:
        *   Finding the entry in the `@graph` array whose `@id` matches the target element (e.g., `mvp:SunStonePeak` or `mvp:RiverOfReflection`).
        *   Extracting specific, predefined properties relevant to the `MVP Analytical Goal` (Section 1.5).
            *   If "SunStonePeak_NarrativeLinks" requested: Extract `mentionsFeature` from `mvp:SkySerpentGiftNarrative` that point to `mvp:SunStonePeak`, and `describesEvent` from the same narrative.
            *   If "RiverOfReflection_Symbol" requested: Find `mvp:RiverOfReflection`, get its `associatedWithVisual` (e.g., `mvp:RiverSerpentGlyphImage`), then find `mvp:RiverSerpentGlyphImage` and get its `depictsSymbol` (e.g., `mvp:SkySerpentSymbol`).
    4.  Returns a simulated `SubmitTaskResult` containing an "Artifact" with a `DataPart` that includes the extracted information.
*   **Simulated A2A Interaction (Illustrative for internal TOE-MVP logic):**
    *   **Task from TOE-MVP:** `TaskID: culture_task_001`
        *   **Input to Agent Function (example):** `{"requestType": "get_cultural_links", "elementName": "SunStonePeak"}`
    *   **Result from Agent Function (example):**
        ```python
        {
            "task_id": "culture_task_001",
            "status": "COMPLETED",
            "artifact": {
                "parts": [
                    {"data": {"elementName": "SunStonePeak", "narrativeLinks": ["mvp:SkySerpentGiftNarrative"], "events": ["mvp:SolsticeGiftEvent"]}}
                ]
            }
        }
        ```
    *   ---
    *   **Task from TOE-MVP:** `TaskID: culture_task_002`
        *   **Input to Agent Function (example):** `{"requestType": "get_associated_symbol", "elementName": "RiverOfReflection"}`
    *   **Result from Agent Function (example):**
        ```python
        {
            "task_id": "culture_task_002",
            "status": "COMPLETED",
            "artifact": {
                "parts": [
                    {"data": {"elementName": "RiverOfReflection", "associatedSymbol": "mvp:SkySerpentSymbol", "visualElement": "mvp:RiverSerpentGlyphImage"}}
                ]
            }
        }
        ```

### 3.4. BasicCrossReferencingAgent-MVP

*   **MVP Purpose:** To perform one specific, predefined cross-referencing check based on the outputs received (simulated) from the `GeoDataQueryAgent-MVP` and `CultureDataQueryAgent-MVP`, aiming to achieve the `MVP Analytical Goal` defined in Section 1.5.
*   **Core Logic:**
    1.  Receives a simulated `AssignTask` request from TOE-MVP. This task input will contain the structured data representing the (simulated) outputs from the previous two agents:
        *   File path for "SunStonePeak" (e.g., `PDN_MVP/geospatial/sun_stone_peak.kml`).
        *   Narrative links for "SunStonePeak" (e.g., `{"narrativeLinks": ["mvp:SkySerpentGiftNarrative"], "events": ["mvp:SolsticeGiftEvent"]}`).
        *   Symbol links for "RiverOfReflection" (e.g., `{"associatedSymbol": "mvp:SkySerpentSymbol", "visualElement": "mvp:RiverSerpentGlyphImage"}`).
        *   The path to the narrative text file (`PDN_MVP/narrative/sky_serpent_gift.txt`) for content verification.
    2.  **Predefined Correlation Check (MVP Analytical Goal Logic):**
        *   Loads the content of the narrative text file (e.g., `sky_serpent_gift.txt`).
        *   Verify that the narrative data (from `culture_sunstone_peak_links`) includes `mvp:SkySerpentGiftNarrative`.
        *   Verify that the text content of `sky_serpent_gift.txt` mentions keywords related to "Sun Stone Peak" AND "River of Reflection" (simple string matching for MVP).
        *   Verify that the symbol data (from `culture_river_reflection_symbol`) links `mvp:RiverOfReflection` to `mvp:SkySerpentSymbol` via `mvp:RiverSerpentGlyphImage`.
    3.  If all conditions for the predefined `MVP Analytical Goal` are met, construct a success message string.
    4.  If not, construct a "correlation not found" message string.
    5.  Returns a simulated `SubmitTaskResult` containing an "Artifact" with a `TextPart` detailing the outcome.
*   **Simulated A2A Interaction (Illustrative for internal TOE-MVP logic):**
    *   **Task from TOE-MVP:** `TaskID: xref_task_001`
        *   **Input to Agent Function (example):**
            ```python
            {
                "geo_sunstone_peak": {"filePath": "PDN_MVP/geospatial/sun_stone_peak.kml"},
                "culture_sunstone_peak_links": {"narrativeLinks": ["mvp:SkySerpentGiftNarrative"], "events": ["mvp:SolsticeGiftEvent"]},
                "culture_river_reflection_symbol": {"associatedSymbol": "mvp:SkySerpentSymbol", "visualElement": "mvp:RiverSerpentGlyphImage"},
                "narrative_text_file": "PDN_MVP/narrative/sky_serpent_gift.txt"
            }
            ```
    *   **Result from Agent Function (Success Example):**
        ```python
        {
            "task_id": "xref_task_001",
            "status": "COMPLETED",
            "artifact": {
                "parts": [
                    {"text": "MVP Success: Correlation Found - The narrative 'The Sky Serpent's Gift' (sky_serpent_gift.txt) links 'SunStonePeak' (sun_stone_peak.kml) with 'RiverOfReflection'. The 'RiverOfReflection' is associated with 'SkySerpentSymbol' via visual 'RiverSerpentGlyphImage'."}
                ]
            }
        }
        ```
    *   **Result from Agent Function (Failure Example):**
        ```python
        {
            "task_id": "xref_task_001",
            "status": "COMPLETED",
            "artifact": {
                "parts": [
                    {"text": "MVP Correlation Not Found: Conditions for predefined link not fully met based on provided inputs."}
                ]
            }
        }
        ```

## 4. A2A Collaboration Hub-MVP & TOE-MVP Specifications

This section defines the minimal "stage" (Hub) and "director" (TOE) for our MVP play.

### 4.1. A2A Collaboration Hub - MVP

*   **Overall Purpose:** For the MVP, the Hub's functionalities (Registry, Discovery) are heavily simplified and primarily simulated or handled by direct configuration.
*   **4.1.1. Agent Registry (MVP)**
    *   **Implementation:** A simple JSON file named `agent_registry_mvp.json`.
    *   **Content:** Lists the three MVP agent types, their simulated network addresses (e.g., placeholder URLs or local function identifiers if in a monolithic prototype), and a single, hardcoded capability string for each that corresponds to their MVP task.
    *   **Example `agent_registry_mvp.json`:**
        ```json
        {
          "agents": [
            {
              "agentId": "GeoDataQueryAgent-MVP-01",
              "agentName": "MVP Geo Data Agent",
              "networkAddress": "local_function:geo_agent_mvp_handler",
              "capabilities": ["fetch_mvp_geospatial_path"]
            },
            {
              "agentId": "CultureDataQueryAgent-MVP-01",
              "agentName": "MVP Culture Data Agent",
              "networkAddress": "local_function:culture_agent_mvp_handler",
              "capabilities": ["fetch_mvp_cultural_info"]
            },
            {
              "agentId": "BasicCrossReferencingAgent-MVP-01",
              "agentName": "MVP Cross Referencing Agent",
              "networkAddress": "local_function:xref_agent_mvp_handler",
              "capabilities": ["perform_mvp_correlation_check"]
            }
          ]
        }
        ```
    *   **Usage:**
        *   The TOE-MVP script might load this at startup to identify the "handlers" for each agent.
        *   For the MVP, the `BasicCrossReferencingAgent-MVP` will not read this directly; the TOE-MVP will explicitly pass it the data it needs. Future, more autonomous agents might use such a registry.

*   **4.1.2. Capability Discovery (MVP)**
    *   **Implementation:** Not a dynamic service. The TOE-MVP script "knows" which agent to call for which step of its predefined sequence, either through hardcoded logic or by using the specific capability strings from `agent_registry_mvp.json` to select the correct agent handler.
    *   There is no inter-agent dynamic discovery in the MVP.

*   **4.1.3. A2A Communication (MVP)**
    *   **Implementation:** As stated in Agent Assumptions (Section 3.1), actual A2A protocol over HTTP is deferred.
    *   Interactions will be simulated by direct function calls within the TOE-MVP script, where each agent's logic is encapsulated in a "handler" function.
    *   The data structures passed to and returned from these handler functions will conceptually mirror the `AssignTask` (input to handler) and `SubmitTaskResult` (output from handler) message structures, including the `Artifact` and `Parts` model, to maintain consistency with the overall A2A design philosophy.

### 4.2. Task Orchestration Engine - MVP (TOE-MVP)

*   **Overall Purpose:** To execute a predefined, linear sequence of tasks involving the three MVP agents to achieve the `MVP Analytical Goal` (defined in Section 1.5).
*   **Implementation:** A simple script (e.g., Python script named `toe_mvp_orchestrator.py`).
*   **Core Logic (Sequential Script Flow):**
    1.  **Initialization:**
        *   Load agent handlers (e.g., by importing Python functions for `GeoDataQueryAgent-MVP`, `CultureDataQueryAgent-MVP`, `BasicCrossReferencingAgent-MVP`).
        *   Define the file paths for the PDN-MVP data (e.g., `sun_stone_peak.kml`, `river_of_reflection.kml`, `sky_serpent_gift.txt`) and the `ckg_mvp.jsonld` file. These will be passed as inputs to the agent handlers as needed.
    2.  **Step 1: Get Geospatial Data for SunStonePeak.**
        *   Call the `GeoDataQueryAgent-MVP` handler function with input specifying "SunStonePeak" (e.g., `{"requestType": "get_geospatial_path", "elementName": "SunStonePeak"}`).
        *   Receive the result (e.g., `{"elementName": "SunStonePeak", "filePath": "PDN_MVP/geospatial/sun_stone_peak.kml"}`). Store this path.
    3.  **Step 2: Get Cultural Links for SunStonePeak.**
        *   Call the `CultureDataQueryAgent-MVP` handler function with input specifying "SunStonePeak" cultural links (e.g., `{"requestType": "get_cultural_links", "elementName": "SunStonePeak", "ckgPath": "path/to/ckg_mvp.jsonld"}`).
        *   Receive the result (e.g., `{"elementName": "SunStonePeak", "narrativeLinks": ["mvp:SkySerpentGiftNarrative"], "events": ["mvp:SolsticeGiftEvent"]}`). Store this.
    4.  **Step 3: Get Associated Symbol for RiverOfReflection.**
        *   Call the `CultureDataQueryAgent-MVP` handler function with input specifying "RiverOfReflection" associated symbol (e.g., `{"requestType": "get_associated_symbol", "elementName": "RiverOfReflection", "ckgPath": "path/to/ckg_mvp.jsonld"}`).
        *   Receive the result (e.g., `{"elementName": "RiverOfReflection", "associatedSymbol": "mvp:SkySerpentSymbol", "visualElement": "mvp:RiverSerpentGlyphImage"}`). Store this.
    5.  **Step 4: Perform Cross-Referencing.**
        *   Prepare the aggregated input data for the `BasicCrossReferencingAgent-MVP` handler by combining the results from Steps 1, 2, and 3, and including the direct path to the narrative text file (e.g., `PDN_MVP/narrative/sky_serpent_gift.txt`).
        *   Call the `BasicCrossReferencingAgent-MVP` handler function with this aggregated input.
        *   Receive the result (e.g., a text string describing the correlation success or failure).
    6.  **Step 5: Output Final Result.**
        *   Print the text result from Step 4 to the console (e.g., `MVP Success: Correlation Found...` or `MVP Correlation Not Found...`).
*   **Error Handling:**
    *   Basic print statements if an agent handler conceptually "fails" (e.g., if `ckg_mvp.jsonld` is not found by `CultureDataQueryAgent-MVP` handler, or if the narrative text file is missing for `BasicCrossReferencingAgent-MVP`).
    *   The script will likely halt on such errors for MVP simplicity. No complex recovery or retry logic.

This section outlines the minimal infrastructure needed to simulate the core collaborative workflow for the A2A World MVP. The TOE-MVP acts as the "script" and "director," guiding the MVP agents through their roles to achieve the specific analytical goal of the "Mini-Storybook."

## 5. MVP Demonstrable Output

This section specifies the expected console output from the `toe_mvp_orchestrator.py` script, demonstrating a successful run or basic error conditions.

### 5.1. Successful Output

*   **Format:** A single line printed to the console by the `toe_mvp_orchestrator.py` script.
*   **Content:** The text result returned by the `BasicCrossReferencingAgent-MVP` upon successfully identifying the predefined correlation outlined in the `MVP Analytical Goal` (Section 1.5).
*   **Example Console Output (Success):**
    ```
    MVP Success: Correlation Found - The narrative 'The Sky Serpent's Gift' (sky_serpent_gift.txt) links 'SunStonePeak' (sun_stone_peak.kml) with 'RiverOfReflection'. The 'RiverOfReflection' is associated with 'SkySerpentSymbol' via visual 'RiverSerpentGlyphImage'.
    ```

### 5.2. Error Message Examples (Illustrative)

*   **Purpose:** To show basic error handling by the TOE-MVP script if a step in its predefined sequence fails due to an agent handler's inability to complete its task (e.g., missing input file, required data not found in the MVP dataset).
*   **Format:** Messages printed to the console by the `toe_mvp_orchestrator.py` script. The script will typically abort after such an error in the MVP.
*   **Example Console Output (Data File Not Found by GeoDataQueryAgent-MVP):**
    ```
    MVP Error: GeoDataQueryAgent-MVP failed. Reason: File not found at PDN_MVP/geospatial/sun_stone_peak.kml. Aborting MVP run.
    ```
*   **Example Console Output (Required Link Not Found by CultureDataQueryAgent-MVP):**
    ```
    MVP Error: CultureDataQueryAgent-MVP failed. Reason: Could not find narrative link for 'SunStonePeak' in ckg_mvp.jsonld. Aborting MVP run.
    ```
*   **Example Console Output (Correlation Not Met by BasicCrossReferencingAgent-MVP):**
    *   This is technically not an "error" in the TOE-MVP script's execution flow but rather the expected "negative" outcome of the `BasicCrossReferencingAgent-MVP`'s logic. The TOE-MVP will still print the result from the agent.
    ```
    MVP Result: Correlation Not Found - Conditions for predefined link not fully met based on provided inputs.
    ```
*   **Note:** These are illustrative. The actual error messages and conditions will depend on the specific checks implemented within the MVP agent handler functions and the TOE-MVP script's logic. The key is that the TOE-MVP provides some indication of where and why a failure occurred in its sequence.
