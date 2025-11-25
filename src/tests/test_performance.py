import time
import requests
import statistics
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "http://localhost:8000"

CACHE_TEST_QUERIES = [
    "Why do we need non-linear activation functions like ReLU?",
    "Describe the architecture of AlexNet and how it differs from LeNet.",
    "How do Gated Recurrent Units (GRUs) solve the vanishing gradient problem?",
    "Show the PyTorch implementation of the cross-entropy loss function.",
    "What does the d2l.synthetic_data function do?",
]

D2L_CONCEPTS = [
    "ReLU", "Sigmoid", "Tanh", "Softmax", "Dropout", "Batch Normalization",
    "LeNet", "AlexNet", "VGG", "NiN", "GoogLeNet", "ResNet", "DenseNet",
    "RNN", "GRU", "LSTM", "BiRNN", "Deep RNN",
    "Encoder-Decoder", "Seq2Seq", "Beam Search", "Attention Mechanism",
    "Bahdanau Attention", "Multi-Head Attention", "Self-Attention",
    "Transformer", "BERT", "GPT", "ViT",
    "Convolution", "Pooling", "Padding", "Stride", "Channels",
    "SGD", "Momentum", "Adagrad", "RMSProp", "Adam",
    "Cross-Entropy Loss", "MSE Loss", "L1 Regularization", "L2 Regularization",
    "Vanishing Gradients", "Exploding Gradients", "Weight Decay",
    "Fine-tuning", "Transfer Learning", "Data Augmentation", "Anchor Boxes", "IoU"
]

# Generate 50 unique, semantically valid queries
UNIQUE_QUERIES = [f"Explain the concept of {concept} in deep learning." for concept in D2L_CONCEPTS]


def test_single_search(query, bypass_cache=True):
    """Test search endpoint (Retrieval + Reranking)"""
    start = time.time()
    try:
        payload = {
            "query": query,
            "k": 10,
            "use_rerank": True,
        }
        if bypass_cache:
            payload["bypass_cache"] = True
            
        response = requests.post(
            f"{BASE_URL}/search",
            json=payload,
            timeout=20
        )
        latency = (time.time() - start) * 1000

        return {"success": True, "latency": latency, "status": response.status_code}

    except Exception as e:
        return {"success": False, "latency": 0, "error": str(e)}


def test_single_ask(query):
    """Test ask endpoint"""
    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            json={"query": query, "k": 5, "use_rerank": True},
            timeout=60
        )
        latency = (time.time() - start) * 1000

        return {"success": True, "latency": latency, "status": response.status_code}

    except Exception as e:
        return {"success": False, "latency": 0, "error": str(e)}


def test_single_ask_stream(query):
    """Test ask/stream endpoint"""
    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/ask/stream",
            json={"query": query, "k": 5, "use_rerank": True},
            timeout=60,
            stream=True
        )

        if response.status_code != 200:
            return {"success": False, "latency": 0, "error": f"Status {response.status_code}"}

        full_response = ""
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                full_response += chunk

        latency = (time.time() - start) * 1000
        has_metadata = "__METADATA__" in full_response

        return {
            "success": True,
            "latency": latency,
            "status": response.status_code,
            "has_metadata": has_metadata,
            "response_length": len(full_response)
        }

    except Exception as e:
        return {"success": False, "latency": 0, "error": str(e)}


def test_p95_latency(endpoint="search"):
    """Test P95 latency using UNIQUE queries to force Cache Misses"""
    print(f"\n> Testing P95 Latency ({endpoint})...")
    latencies = []

    if endpoint == "search":
        test_func = test_single_search
        queries_to_use = UNIQUE_QUERIES
    elif endpoint == "stream":
        test_func = test_single_ask_stream
        queries_to_use = CACHE_TEST_QUERIES
    else:
        test_func = test_single_search
        queries_to_use = UNIQUE_QUERIES

    num_requests = min(50, len(queries_to_use))

    for i in range(num_requests):
        query = queries_to_use[i]
        result = test_func(query)
        if result["success"]:
            latencies.append(result["latency"])
            print(f"   [{i + 1}/{num_requests}] {result['latency']:.0f}ms", end='\r')

    print()

    if latencies and len(latencies) >= 2:
        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]
        avg = statistics.mean(latencies)

        print(f"   Average: {avg:.0f}ms")
        print(f"   Median (P50): {p50:.0f}ms")
        print(f"   P95: {p95:.0f}ms")
        print(f"   P99: {p99:.0f}ms")

        return {"avg": avg, "p50": p50, "p95": p95}

    return None


def test_throughput(endpoint="search"):
    """Test throughput"""
    print(f"\n> Testing Throughput ({endpoint})...")

    if endpoint == "search":
        test_func = test_single_search
        queries = UNIQUE_QUERIES
    else:
        test_func = test_single_ask_stream
        queries = CACHE_TEST_QUERIES

    num_requests = 20
    max_workers = 5

    start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(test_func, queries[i % len(queries)])
                   for i in range(num_requests)]
        results = [f.result() for f in futures]

    duration = time.time() - start
    successful = sum(1 for r in results if r["success"])
    rps = successful / duration if duration > 0 else 0

    print(f"   Throughput: {rps:.2f} req/s")
    return {"rps": rps}


def test_cache_performance():
    """Test multi-layer cache performance"""
    print("\n> Testing Cache Performance...")

    test_queries = CACHE_TEST_QUERIES[:3]

    # Cache Misses
    print("   Testing Cache misses...")
    miss_times = []

    for query in test_queries:
        start = time.time()
        # Bypass cache to force real search
        requests.post(f"{BASE_URL}/search", json={
            "query": query, 
            "k": 5, 
            "use_rerank": True,
            "bypass_cache": True
        })
        latency = (time.time() - start) * 1000
        miss_times.append(latency)
        print(f"      Miss: {latency:.0f}ms")

    avg_miss_time = statistics.mean(miss_times)
    time.sleep(0.5)

    # Cache Hits
    print("   Testing Cache hits...")
    hit_times = []

    for query in test_queries:
        start = time.time()
        requests.post(f"{BASE_URL}/search", json={"query": query, "k": 5, "use_rerank": True})
        latency = (time.time() - start) * 1000
        hit_times.append(latency)
        print(f"      Hit:  {latency:.0f}ms")

    avg_hit_time = statistics.mean(hit_times)
    speedup = avg_miss_time / avg_hit_time if avg_hit_time > 0 else 0

    print(f"\n   Avg Miss (Inference): {avg_miss_time:.0f}ms")
    print(f"   Avg Hit (Cache):      {avg_hit_time:.0f}ms")
    print(f"   Speedup:              {speedup:.1f}x")

    return {
        "speedup": speedup, 
        "hit_rate": 100 if avg_hit_time < 50 else 0,
        "avg_hit_time": avg_hit_time,
        "avg_miss_time": avg_miss_time
    }


def test_uptime():
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
        return 100
    except:
        return 0


def main():
    print("---System Performance Test---")

    # Check Server
    try:
        requests.get(f"{BASE_URL}/health")
    except:
        print("Server not running.")
        return

    # Test 1: Uptime
    test_uptime()

    # Test 2: Cache Performance
    test_cache_performance()

    # Test 3: Search Latency
    test_p95_latency("search")

    # Test 4: Search Throughput
    test_throughput("search")

    # Test 5: Stream Latency
    test_p95_latency("stream")

    # Test 6: Stream Throughput
    test_throughput("stream")

    print()
    print("---End---")


if __name__ == "__main__":
    main()
