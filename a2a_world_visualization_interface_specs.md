# A2A World: Basic Visualization Interface Specifications

## 1. Introduction and Purpose

This document outlines the initial specifications for a basic visualization interface for the A2A World system. The primary purpose of this interface is to allow human researchers, developers, and enthusiasts to:

*   Observe the activities of autonomous agents operating within the A2A World.
*   Visualize the data being discovered, shared, and interpreted by these agents.
*   Understand the emergent relationships and hypotheses being formed.
*   Provide a window into the simulated environment and its evolving knowledge base.

This specification describes a foundational ("Minimum Viable Product" - MVP) interface. Future iterations may include more advanced features and deeper interaction capabilities.

## 2. Core Components

The visualization interface will be composed of three main components:

### A. Global Map/Globe View

This component provides the primary geospatial context for observing agent activities and data.

*   **Base Layer:** Users should be able to switch between different base map layers:
    *   Satellite Imagery (e.g., from Sentinel, Landsat, or commercial providers via APIs).
    *   Topographic Map (e.g., OpenTopoMap, or other similar sources).
    *   Simple Political Boundary Map (showing country borders and major administrative regions).
*   **Data Overlays (Togglable Layers):** These layers display information derived from agent A2A messages. Each layer should be individually togglable (on/off).
    *   **Geospatial Features Layer:**
        *   **Source:** `FindingBroadcast` messages with `finding_type` like `geospatial_data_acquired` or (future) `geospatial_pattern_detected` from agents such as `GeoDataQueryAgent`.
        *   **Display:** Markers (e.g., icons, dots) or outlines/polygons on the map at the `location_context` specified in the finding.
        *   **Interaction:** Clicking a marker/feature should display basic information (e.g., `finding_id`, `data_description`, `source_agent_id`) and select it in the Information Panel for full details.
    *   **Cultural Data Points Layer:**
        *   **Source:** `FindingBroadcast` messages with `finding_type` like `cultural_element_retrieved` from agents such as `CultureDataQueryAgent`.
        *   **Display:** Distinct markers (different from geospatial features) at the `location_context` specified in the finding (if available and precise enough for map display).
        *   **Interaction:** Clicking a marker should display basic information and select it in the Information Panel.
    *   **Hypothesis Links Layer:**
        *   **Source:** `HypothesisProposal` messages from agents like `BasicCrossReferencingAgent`.
        *   **Display:** A visual link (e.g., a line, curve, or animated arc) drawn on the map connecting the geographic locations of the `linked_findings` (e.g., connecting a geospatial feature marker with a cultural data point marker if their locations are distinct and specified).
        *   **Styling:** The style of the link could vary based on the `initial_confidence_score` of the hypothesis (e.g., color intensity from light to dark, thickness, dashed vs. solid lines).
        *   **Interaction:** Clicking the link should select the hypothesis in the Information Panel for detailed viewing.
    *   **Agent Activity Hotspots (Optional - Future Enhancement):**
        *   **Source:** Aggregated `location_context` data from various A2A messages over time.
        *   **Display:** A heatmap overlay indicating geographic areas with a high density of agent messages or significant findings.

### B. Information Panel

This panel displays detailed information about items selected on the map or from other interface components.

*   **Selected Item Display:**
    *   **For a Geospatial Feature:**
        *   Displays the full payload of the selected `FindingBroadcast` message (pretty-printed JSON or a structured, human-readable format).
        *   Includes `finding_id`, `source_agent_id`, `data_description`, full `location_context`, `temporal_context`, `confidence_score`, `supporting_evidence_links`, and `tags`.
    *   **For a Cultural Data Point:**
        *   Displays the full payload of the selected `FindingBroadcast` message.
        *   Includes `finding_id`, `source_agent_id`, `data_description` (e.g., summary of myth/symbol), `location_context` (if any), `temporal_context`, `confidence_score`, `supporting_evidence_links`, and `tags`.
    *   **For a Hypothesis Link:**
        *   Displays the full payload of the selected `HypothesisProposal` message.
        *   Includes `hypothesis_id`, `proposing_agent_id`, full `hypothesis_statement`, `linked_findings` (with clickable links to view each finding's details), `supporting_arguments`, `initial_confidence_score`, and any `query_for_evidence`.
        *   Should allow easy navigation to view the details of each of the `linked_findings` in the Information Panel.
*   **Timeline View (Simple):**
    *   **Purpose:** Provides a chronological overview of major events and discoveries within A2A World.
    *   **Display:** A scrollable list of timestamped entries. Each entry represents a significant A2A message, such as:
        *   New `GeospatialDataRequest` initiated.
        *   New `FindingBroadcast` (geospatial or cultural).
        *   New `HypothesisProposal` made.
        *   (Future) Task announcements or claims.
    *   **Entry Content:** Each entry should show a concise summary (e.g., "GeoDataAgent acquired LiDAR for Nazca", "CultureAgent found 'Sky God' myth related to Andes", "CrossRefAgent proposed link between Nazca lines and Condor myth").
    *   **Interaction:** Clicking an entry in the timeline should:
        *   Highlight the relevant item(s) on the map (if applicable, e.g., pan/zoom to the location of a finding).
        *   Display the full details of the corresponding message in the Selected Item Display area of the Information Panel.

### C. Agent A2A Message Log

This component provides a more detailed, raw feed of A2A messages for deeper analysis and debugging.

*   **Purpose:** To allow observation of the direct communication flow between agents.
*   **Display:** A tabular or list view of messages, showing columns for:
    *   `Timestamp` (when the message was recorded by the system).
    *   `Source Agent ID` (from message header).
    *   `Destination Agent ID` (if applicable, or 'BROADCAST'/'ALL').
    *   `Message Type` (e.g., `GeospatialDataRequest`, `FindingBroadcast`, `HypothesisProposal`).
    *   A short summary or key fields from the payload.
*   **Filtering:** Basic filtering capabilities:
    *   By `source_agent_id`.
    *   By `message_type`.
    *   (Future) By `task_id` or keywords in payload.
*   **Detail on Click:** Clicking on a message row in the log should display its full payload (header and payload) in a structured, readable format (e.g., pretty-printed JSON or an expandable tree view) in a dedicated area or modal window.

## 3. User Interaction (Initial Scope)

The initial version of the interface will support the following user interactions:

*   **Map Navigation:**
    *   Panning (click and drag).
    *   Zooming (mouse wheel, +/- buttons).
*   **Layer Control:**
    *   Toggling the visibility of different data overlay layers on the map.
    *   Switching base map layers.
*   **Information Access:**
    *   Clicking on map items (features, cultural points, hypothesis links) to view their details in the Information Panel.
    *   Clicking on entries in the Timeline View to see details and highlight on map.
    *   Clicking on messages in the Agent A2A Message Log to view their full content.
*   **Filtering:**
    *   Using the filter options in the Agent A2A Message Log.

**Future Considerations (Out of Scope for MVP):**

*   User ability to manually submit `GeospatialDataRequest` or `CulturalDataQuery` messages through the interface.
*   Tools for users to draw regions of interest on the map to trigger queries.
*   Mechanisms for users to provide feedback on hypotheses (e.g., upvote/downvote, add comments).
*   More advanced analytical tools and charts.

## 4. Technical Considerations (High-Level)

*   **Platform:** Web-based application, accessible through modern web browsers.
*   **Mapping Library:** Utilize a suitable JavaScript mapping library. Options include:
    *   **Leaflet:** Good for 2D maps, simple, lightweight, many plugins.
    *   **OpenLayers:** More powerful 2D mapping, supports more complex GIS functionality.
    *   **CesiumJS:** For a 3D globe view, which could be highly immersive for planetary-scale data. (Might be more complex for an initial version but desirable for "World" aspect).
    *   **Mapbox GL JS:** High-performance vector maps, customizable styling.
*   **Data Source:**
    *   The visualization interface will need to consume A2A messages. This could be achieved by:
        *   Connecting to a central message bus/broker (e.g., RabbitMQ, Kafka, MQTT) where all agent A2A communications are published.
        *   Querying a database where all A2A messages and key findings are logged.
*   **Backend:** A lightweight backend service might be needed to:
    *   Serve the web application.
    *   Aggregate/filter messages if direct connection to a message bus by the frontend is not feasible or efficient.
    *   Manage user sessions or preferences (if any in the future).
*   **Frontend Framework:** A modern JavaScript framework (e.g., React, Vue, Angular) could be used for building the user interface components.

This document provides a starting point for the development of the A2A World Visualization Interface. Further refinement and detailed design will be necessary during the implementation process.
