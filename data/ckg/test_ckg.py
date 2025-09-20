#!/usr/bin/env python3
"""
Test script for Cultural Knowledge Graph functionality.
This script tests the basic functionality of the CKG module.
"""

import sys
import os

# Add the parent directory to the path to import the ckg module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from data.ckg import CulturalKnowledgeGraph

    print("✓ Successfully imported CulturalKnowledgeGraph")
except ImportError as e:
    print(f"✗ Failed to import CulturalKnowledgeGraph: {e}")
    sys.exit(1)


def test_import():
    """Test that all modules can be imported."""
    try:
        from data.ckg.connection import get_db_connection
        from data.ckg.schema import create_collections
        from data.ckg.operations import insert_mythological_entity, get_all_entities

        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_ckg_class():
    """Test the CKG class instantiation."""
    try:
        ckg = CulturalKnowledgeGraph()
        print("✓ CulturalKnowledgeGraph class instantiated successfully")
        print(f"  - Host: {ckg.host}")
        print(f"  - Database: {ckg.database}")
        return True
    except Exception as e:
        print(f"✗ Failed to instantiate CKG class: {e}")
        return False


def test_connection_attempt():
    """Test database connection attempt (may fail if ArangoDB not running)."""
    try:
        ckg = CulturalKnowledgeGraph()
        # This will attempt to connect, but may fail if ArangoDB is not running
        db = ckg.connect()
        print("✓ Database connection successful")
        return True
    except Exception as e:
        print(f"⚠ Database connection failed (expected if ArangoDB not running): {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Cultural Knowledge Graph Implementation")
    print("=" * 50)

    tests = [
        ("Module Import", test_import),
        ("CKG Class", test_ckg_class),
        ("Database Connection", test_connection_attempt),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        if test_func():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All tests passed! The CKG implementation is ready.")
    else:
        print("⚠ Some tests failed. Check the output above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
