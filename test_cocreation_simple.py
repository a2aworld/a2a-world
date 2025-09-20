#!/usr/bin/env python3
"""
Simple test script for the Co-Creation Workflow integration.

This script tests the core integration without heavy dependencies.
"""

import sys
from pathlib import Path
import json

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_file_structure():
    """Test that all required files exist."""
    print("🧪 Testing file structure...")

    required_files = [
        "workflow/cocreation_workflow.py",
        "workflow/__init__.py",
        "backend/api/workflow.py",
        "interfaces/genesis_interface.html",
        "interfaces/web/index.html",
        "interfaces/web/script.js",
    ]

    missing_files = []
    for file_path in required_files:
        full_path = Path(__file__).parent / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")

    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False

    print("  ✅ All required files present")
    return True


def test_api_routes():
    """Test that API routes are properly defined."""
    print("🧪 Testing API route definitions...")

    try:
        # Read the workflow API file
        api_file = Path(__file__).parent / "backend/api/workflow.py"
        with open(api_file, "r") as f:
            content = f.read()

        # Check for key route definitions
        required_routes = [
            '@router.post("/start"',
            '@router.get("/status/{workflow_id}"',
            '@router.post("/feedback"',
            '@router.get("/history"',
            '@router.post("/trigger-autonomous"',
            '@router.get("/philosophy"',
        ]

        missing_routes = []
        for route in required_routes:
            if route not in content:
                missing_routes.append(route)
            else:
                print(f"  ✅ {route}")

        if missing_routes:
            print(f"  ❌ Missing routes: {missing_routes}")
            return False

        print("  ✅ All required API routes defined")
        return True

    except Exception as e:
        print(f"  ❌ API route test failed: {e}")
        return False


def test_workflow_stages():
    """Test that workflow stages are properly defined."""
    print("🧪 Testing workflow stage definitions...")

    try:
        # Read the workflow file
        workflow_file = Path(__file__).parent / "workflow/cocreation_workflow.py"
        with open(workflow_file, "r") as f:
            content = f.read()

        # Check for workflow stages
        stages = ["DOUBT", "DISCOVERY", "ART", "WISDOM", "KNOWLEDGE", "PUBLICATION"]
        missing_stages = []

        for stage in stages:
            if stage not in content:
                missing_stages.append(stage)
            else:
                print(f"  ✅ Stage: {stage}")

        if missing_stages:
            print(f"  ❌ Missing stages: {missing_stages}")
            return False

        print("  ✅ All workflow stages defined")
        return True

    except Exception as e:
        print(f"  ❌ Workflow stage test failed: {e}")
        return False


def test_interface_features():
    """Test that interface features are implemented."""
    print("🧪 Testing interface features...")

    try:
        # Test Genesis Interface
        genesis_file = Path(__file__).parent / "interfaces/genesis_interface.html"
        with open(genesis_file, "r", encoding="utf-8") as f:
            genesis_content = f.read()

        genesis_features = [
            "Co-Creation Workflow",
            "Posthuman Creativity",
            "workflow-form",
            "feedback-modal",
        ]

        missing_genesis = []
        for feature in genesis_features:
            if feature not in genesis_content:
                missing_genesis.append(feature)
            else:
                print(f"  ✅ Genesis: {feature}")

        # Test Web Interface enhancements
        web_script_file = Path(__file__).parent / "interfaces/web/script.js"
        with open(web_script_file, "r") as f:
            web_content = f.read()

        web_features = [
            "checkWorkflowStatus",
            "toggleCoCreationMode",
            "currentWorkflowId",
        ]

        missing_web = []
        for feature in web_features:
            if feature not in web_content:
                missing_web.append(feature)
            else:
                print(f"  ✅ Web: {feature}")

        if missing_genesis or missing_web:
            print(f"  ❌ Missing Genesis features: {missing_genesis}")
            print(f"  ❌ Missing Web features: {missing_web}")
            return False

        print("  ✅ All interface features implemented")
        return True

    except Exception as e:
        print(f"  ❌ Interface test failed: {e}")
        return False


def test_integration_points():
    """Test that integration points are properly connected."""
    print("🧪 Testing integration points...")

    try:
        # Check backend main.py for workflow integration
        main_file = Path(__file__).parent / "backend/main.py"
        with open(main_file, "r") as f:
            main_content = f.read()

        integration_points = [
            "workflow.router",
            "initialize_systems",
            "cocreation_workflow",
        ]

        missing_integration = []
        for point in integration_points:
            if point not in main_content:
                missing_integration.append(point)
            else:
                print(f"  ✅ Integration: {point}")

        if missing_integration:
            print(f"  ❌ Missing integration points: {missing_integration}")
            return False

        print("  ✅ All integration points connected")
        return True

    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False


def generate_test_report(results):
    """Generate a test report."""
    print("\n" + "=" * 60)
    print("📊 Co-Creation Workflow Integration Test Report")
    print("=" * 60)

    passed = sum(1 for result in results if result[1])
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All integration tests passed!")
        print("The Co-Creation Workflow is successfully integrated.")
        print("\nNext steps:")
        print("1. Install required dependencies (langchain, langgraph, etc.)")
        print("2. Start the backend server: python -m uvicorn backend.main:app")
        print("3. Open interfaces/genesis_interface.html in a browser")
        print("4. Test the complete workflow from doubt to knowledge")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
        return False


def main():
    """Run all integration tests."""
    print("🚀 Terra Constellata Co-Creation Workflow Integration Tests")
    print("Testing the unified human-AI co-creation system")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("File Structure", test_file_structure()))
    results.append(("API Routes", test_api_routes()))
    results.append(("Workflow Stages", test_workflow_stages()))
    results.append(("Interface Features", test_interface_features()))
    results.append(("Integration Points", test_integration_points()))

    # Generate report
    success = generate_test_report(results)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
