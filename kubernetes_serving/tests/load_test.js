/**
 * K6 Load Test Script for Model API
 * 
 * This script performs load testing on the deployed Kubernetes service
 * to verify performance under load and test auto-scaling behavior.
 * 
 * Install K6:
 *   Windows: winget install k6
 *   Mac: brew install k6
 *   Linux: https://k6.io/docs/getting-started/installation/
 * 
 * Run: k6 run load-test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const predictionLatency = new Trend('prediction_latency');
const successfulPredictions = new Counter('successful_predictions');

// Test configuration
export const options = {
  stages: [
    // Warm-up: 10 VUs for 30 seconds
    { duration: '30s', target: 10 },
    
    // Ramp-up: Increase to 50 VUs over 1 minute
    { duration: '1m', target: 50 },
    
    // Sustained load: 50 VUs for 2 minutes
    { duration: '2m', target: 50 },
    
    // Peak load: Spike to 100 VUs for 30 seconds (test auto-scaling)
    { duration: '30s', target: 100 },
    
    // Cool down: Decrease to 0 VUs over 30 seconds
    { duration: '30s', target: 0 },
  ],
  
  // Thresholds (pass/fail criteria)
  thresholds: {
    // 95% of requests must complete within 500ms
    'prediction_latency': ['p(95)<500'],
    
    // Error rate must be below 1%
    'errors': ['rate<0.01'],
    
    // Overall request duration
    'http_req_duration': ['p(99)<1000'],
  },
};

// API endpoint (change this based on your service URL)
const BASE_URL = __ENV.API_URL || 'http://localhost:8080';

// Generate a simple test image (1x1 red pixel as PNG)
// In a real test, you'd use an actual image file
const createTestImage = () => {
  // This is a minimal 1x1 red pixel PNG encoded as base64
  const base64Image = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==';
  return http.file(Uint8Array.from(atob(base64Image), c => c.charCodeAt(0)), 'test.png', 'image/png');
};

export function setup() {
  // Run once before the test starts
  console.log('Starting load test...');
  console.log(`Target URL: ${BASE_URL}`);
  
  // Health check
  const healthRes = http.get(`${BASE_URL}/health`);
  check(healthRes, {
    'health check is 200': (r) => r.status === 200,
  });
  
  if (healthRes.status !== 200) {
    console.error('Health check failed! Service may not be ready.');
  }
}

export default function () {
  // Test 1: Health endpoint (lightweight check)
  const healthRes = http.get(`${BASE_URL}/health`);
  check(healthRes, {
    'health check status is 200': (r) => r.status === 200,
    'health check has status field': (r) => JSON.parse(r.body).status !== undefined,
  });
  
  sleep(0.5);
  
  // Test 2: Prediction endpoint with form data
  const formData = {
    file: createTestImage(),
    top_k: '3',
  };
  
  const predictionRes = http.post(`${BASE_URL}/predict`, formData);
  
  const success = check(predictionRes, {
    'prediction status is 200': (r) => r.status === 200,
    'prediction has predictions': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.predictions && Array.isArray(body.predictions);
      } catch (e) {
        return false;
      }
    },
    'prediction latency is reasonable': (r) => r.timings.duration < 1000,
  });
  
  // Record metrics
  if (success) {
    successfulPredictions.add(1);
    predictionLatency.add(predictionRes.timings.duration);
  } else {
    errorRate.add(1);
  }
  
  sleep(1);
}

export function teardown(data) {
  // Run once after the test completes
  console.log('Load test completed!');
}

// Alternative: Simple smoke test (run with: k6 run --vus 1 --duration 30s load-test.js)
export function smokeTest() {
  const res = http.get(`${BASE_URL}/health`);
  check(res, { 'smoke test: status is 200': (r) => r.status === 200 });
  sleep(1);
}
