#!/usr/bin/env python3
"""E2E test for evaluator API.

This script tests the full evaluation pipeline:
1. API health check
2. List available models
3. Start an evaluation job
4. Poll for completion
5. Verify results structure

Usage:
    python test_evaluator_e2e.py
    python test_evaluator_e2e.py --api-url http://localhost:8003/api
"""
import argparse
import requests
import time
import sys


def test_evaluator_e2e(api_base: str):
    print("=" * 60)
    print("E2E Evaluator Test")
    print(f"API Base: {api_base}")
    print("=" * 60)

    # 1. Check API health
    print("\n1. Checking API health...")
    try:
        r = requests.get(f"{api_base}/health", timeout=10)
        assert r.status_code == 200, f"API not healthy: {r.status_code} - {r.text}"
        print("   OK - API is healthy")
    except requests.exceptions.ConnectionError:
        print(f"   FAILED - Cannot connect to API at {api_base}")
        print("   Make sure the API server is running:")
        print("   uvicorn bashgym.api.routes:app --host 0.0.0.0 --port 8003")
        sys.exit(1)

    # 2. List available models
    print("\n2. Listing available models...")
    r = requests.get(f"{api_base}/models")
    assert r.status_code == 200, f"Failed to list models: {r.text}"
    models = r.json()

    if len(models) == 0:
        print("   WARNING - No models available for evaluation")
        print("   You need to train a model first, or create a mock model directory.")
        print("")
        print("   To create a mock model for testing:")
        print("   mkdir -p data/models/test_model/merged")
        print("   echo '{}' > data/models/test_model/merged/config.json")
        print("")
        # Create mock model for testing
        from pathlib import Path
        mock_model_dir = Path.cwd() / "data" / "models" / "test_model" / "merged"
        mock_model_dir.mkdir(parents=True, exist_ok=True)
        (mock_model_dir / "config.json").write_text("{}")
        print("   Created mock model 'test_model' for testing")

        # Re-fetch models
        r = requests.get(f"{api_base}/models")
        models = r.json()

    if len(models) == 0:
        print("   FAILED - Still no models available after creating mock")
        sys.exit(1)

    model_id = models[0]["model_id"]
    print(f"   OK - Found {len(models)} models, using: {model_id}")

    # 3. Start evaluation
    print("\n3. Starting evaluation...")
    r = requests.post(f"{api_base}/evaluation/run", json={
        "model_id": model_id,
        "benchmarks": ["simple_test"],
        "num_samples": 2
    })
    assert r.status_code == 200, f"Failed to start evaluation: {r.text}"
    job = r.json()
    job_id = job["job_id"]
    print(f"   OK - Started job: {job_id}")

    # 4. Poll for completion
    print("\n4. Waiting for completion...")
    max_wait = 60  # seconds
    start = time.time()
    while time.time() - start < max_wait:
        r = requests.get(f"{api_base}/evaluation/{job_id}")
        assert r.status_code == 200, f"Failed to get status: {r.text}"
        job = r.json()
        status = job["status"]
        print(f"   Status: {status}")

        if status == "completed":
            print("\n5. Evaluation completed!")
            print(f"   Results: {job['results']}")
            break
        elif status == "failed":
            print(f"\n   FAILED: {job.get('error', 'Unknown error')}")
            sys.exit(1)

        time.sleep(2)
    else:
        print("\n   TIMEOUT - evaluation took too long")
        sys.exit(1)

    # 5. Verify results structure
    print("\n6. Verifying results structure...")
    results = job.get("results", {})
    assert "simple_test" in results, "Missing simple_test results"
    result = results["simple_test"]
    assert "score" in result, "Missing score"
    assert "passed" in result, "Missing passed count"
    assert "total" in result, "Missing total count"
    print(f"   OK - Score: {result['score']:.1%} ({result['passed']}/{result['total']})")

    # 6. Test listing evaluations
    print("\n7. Testing list evaluations...")
    r = requests.get(f"{api_base}/evaluation")
    assert r.status_code == 200, f"Failed to list evaluations: {r.text}"
    evaluations = r.json()
    assert len(evaluations) > 0, "No evaluations returned"
    assert any(e["job_id"] == job_id for e in evaluations), "Our job not in list"
    print(f"   OK - Found {len(evaluations)} evaluation(s)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(description="E2E test for evaluator API")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8003/api",
        help="API base URL (default: http://localhost:8003/api)"
    )
    args = parser.parse_args()

    sys.exit(test_evaluator_e2e(args.api_url))


if __name__ == "__main__":
    main()
