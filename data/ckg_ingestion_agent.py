#!/usr/bin/env python3
"""
CKG Ingestion Agent for Phase Zero Bootstrap

This script downloads curated texts from Project Gutenberg and Internet Sacred Text Archive,
processes them using spaCy for Named Entity Recognition (NER) and NLTK for dependency parsing,
extracts entities and relationships, and ingests them into the Cultural Knowledge Graph (CKG).

Usage:
    python ckg_ingestion_agent.py

Requirements:
    - ArangoDB running on localhost:8529
    - Python packages: requests, spacy, nltk, arango
    - spaCy model: en_core_web_sm
"""

import logging
import requests
import spacy
import nltk
import sys
import os
from nltk.parse import CoreNLPDependencyParser
from collections import defaultdict

# Add project root to sys.path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.ckg import CulturalKnowledgeGraph
from data.ckg.operations import (
    insert_text_source,
    insert_mythological_entity,
    insert_geographic_feature,
    insert_cultural_concept,
    insert_edge,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Curated text sources (URLs to plain text)
TEXT_SOURCES = [
    # Project Gutenberg examples
    "https://www.gutenberg.org/files/11/11-0.txt",  # Alice's Adventures in Wonderland
    "https://www.gutenberg.org/files/74/74-0.txt",  # The Adventures of Tom Sawyer
    "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
    # Internet Sacred Text Archive examples (plain text URLs)
    "https://www.sacred-texts.com/hin/rigveda/rv01001.htm",  # Rig Veda (HTML, need to extract text)
    # Add more as needed
]


def download_text(url):
    """
    Download text content from a URL.

    Args:
        url (str): URL to download from

    Returns:
        str: Downloaded text content
    """
    try:
        logger.info(f"Downloading text from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        # For HTML pages, we might need to parse, but assuming plain text for now
        return response.text
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return None


def process_text_with_spacy(text):
    """
    Process text with spaCy for NER.

    Args:
        text (str): Input text

    Returns:
        spacy.Doc: Processed document
    """
    logger.info("Processing text with spaCy for NER")
    doc = nlp(text)
    return doc


def extract_entities(doc):
    """
    Extract named entities from spaCy doc.

    Args:
        doc (spacy.Doc): Processed document

    Returns:
        dict: Dictionary of entity types to list of entities
    """
    entities = defaultdict(list)
    for ent in doc.ents:
        entities[ent.label_].append(
            {"text": ent.text, "start": ent.start_char, "end": ent.end_char}
        )
    logger.info(f"Extracted entities: {dict(entities)}")
    return entities


def extract_relationships(doc):
    """
    Extract relationships using dependency parsing (simplified).

    Args:
        doc (spacy.Doc): Processed document

    Returns:
        list: List of relationships (subject, verb, object)
    """
    relationships = []
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            subject = token.text
            verb = token.head.text
            # Find object
            for child in token.head.children:
                if child.dep_ == "dobj":
                    obj = child.text
                    relationships.append((subject, verb, obj))
                    break
    logger.info(f"Extracted relationships: {relationships}")
    return relationships


def map_entity_to_ckg_type(entity_label):
    """
    Map spaCy entity label to CKG entity type.

    Args:
        entity_label (str): spaCy label (e.g., 'PERSON', 'GPE')

    Returns:
        str: CKG entity type
    """
    mapping = {
        "PERSON": "mythological",  # Assuming cultural/mythological context
        "ORG": "cultural",
        "GPE": "geographic",
        "LOC": "geographic",
        "MISC": "cultural",
    }
    return mapping.get(entity_label, "cultural")


def ingest_to_ckg(ckg, title, content, entities, relationships):
    """
    Ingest extracted data into CKG.

    Args:
        ckg (CulturalKnowledgeGraph): CKG instance
        title (str): Text title
        content (str): Text content
        entities (dict): Extracted entities
        relationships (list): Extracted relationships
    """
    try:
        # Insert text source
        text_result = ckg.insert_entity(
            "text", title=title, content=content, source_type="book"
        )
        text_id = f"TextSource/{text_result['_key']}"
        logger.info(f"Inserted text source: {text_id}")

        # Insert entities
        entity_ids = {}
        for label, ents in entities.items():
            ckg_type = map_entity_to_ckg_type(label)
            for ent in ents:
                name = ent["text"]
                if name not in entity_ids:
                    if ckg_type == "mythological":
                        result = ckg.insert_entity(
                            "mythological",
                            name=name,
                            description=f"Entity from {title}",
                        )
                    elif ckg_type == "geographic":
                        result = ckg.insert_entity(
                            "geographic", name=name, type="location", coordinates=[0, 0]
                        )  # Placeholder coords
                    elif ckg_type == "cultural":
                        result = ckg.insert_entity(
                            "cultural", name=name, description=f"Concept from {title}"
                        )
                    entity_ids[name] = f"{result['_id']}"
                    logger.info(f"Inserted entity: {name} as {ckg_type}")

        # Insert relationships
        for subj, verb, obj in relationships:
            if subj in entity_ids and obj in entity_ids:
                ckg.insert_relationship(
                    "RELATED_TO", entity_ids[subj], entity_ids[obj], relation=verb
                )
                logger.info(f"Inserted relationship: {subj} {verb} {obj}")

        # Link entities to text
        for name, ent_id in entity_ids.items():
            ckg.insert_relationship("MENTIONED_IN", ent_id, text_id)
            logger.info(f"Linked {name} to {title}")

    except Exception as e:
        logger.error(f"Failed to ingest data: {e}")


def main():
    """
    Main ingestion process.
    """
    logger.info("Starting CKG Ingestion Agent")

    # Initialize CKG
    ckg = CulturalKnowledgeGraph()

    for url in TEXT_SOURCES:
        text = download_text(url)
        if not text:
            continue

        # Extract title from URL or content
        title = url.split("/")[-1].split(".")[0]

        # Process text
        doc = process_text_with_spacy(text)
        entities = extract_entities(doc)
        relationships = extract_relationships(doc)

        # Ingest to CKG
        ingest_to_ckg(ckg, title, text, entities, relationships)

    logger.info("CKG Ingestion Agent completed")


if __name__ == "__main__":
    main()
