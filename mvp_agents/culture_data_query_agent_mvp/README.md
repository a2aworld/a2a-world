# CultureDataQueryAgent-MVP

## Purpose
This directory is a placeholder for the **CultureDataQueryAgent-MVP**.

The `CultureDataQueryAgent-MVP` is responsible for querying a minimal, predefined Cultural Knowledge Graph (CKG-MVP) for information related to the "Mini-Storybook" data subset used in the Minimum Viable Product (MVP).

## MVP Functionality
-   Receives a simulated task from the TOE-MVP specifying what cultural information is needed (e.g., links for a specific element, associated symbols).
-   Loads and parses the `ckg_mvp.jsonld` file.
-   Performs simple search/filter operations on the JSON-LD data to find predefined information relevant to the MVP's analytical goal.
-   Returns the extracted information as its result.

## Detailed Specification
For detailed specifications, including its exact simulated A2A interactions and expected inputs/outputs for the MVP, please refer to **Section 3.3 (CultureDataQueryAgent-MVP)** in the main `a2a_world_mvp_specifications.md` document located in the `specifications/` directory of this repository.

## Future Development
In future iterations beyond the MVP, this agent would:
-   Interact with a comprehensive Cultural Knowledge Graph and Symbolic Lexicon.
-   Execute complex semantic queries (e.g., SPARQL if the CKG is RDF-based).
-   Handle actual A2A protocol communication.
-   Potentially perform NLP tasks to extract information from textual sources within the CKG.

## Running the MVP Agent (Standalone Test)
The Python script `culture_data_query_agent_mvp.py` includes a standalone test block. You can run it directly to see example outputs based on the `ckg_mvp.jsonld` file:
```bash
python culture_data_query_agent_mvp.py
```
This requires `ckg_mvp.jsonld` to be present at `../../ckg_mvp.jsonld` (i.e., in the repository root) relative to the script.
