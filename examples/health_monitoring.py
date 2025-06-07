#!/usr/bin/env python3
"""
Example script demonstrating health monitoring capabilities.

This script shows how to:
1. Perform one-time health checks
2. Set up continuous monitoring
3. Handle health alerts
4. Export health metrics
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphiti_health import (
    HealthStatus,
    ComponentType,
    Neo4jHealthChecker,
    GraphitiHealthChecker,
    SystemHealthChecker,
    HealthMonitor,
    HealthService,
    check_system_health,
)
from graphiti_config import get_config


async def simple_health_check():
    """Perform a simple one-time health check."""
    print("=== Simple Health Check ===\n")
    
    # Check system health
    health_status = await check_system_health()
    
    print(f"Overall Status: {health_status['status'].upper()}")
    print(f"Timestamp: {health_status['timestamp']}")
    
    # Display component status
    print("\nComponent Status:")
    for component, details in health_status['components'].items():
        status = details.get('status', 'unknown')
        error = details.get('error', '')
        print(f"  - {component}: {status}")
        if error:
            print(f"    Error: {error}")


async def detailed_component_checks():
    """Perform detailed checks for each component."""
    print("\n=== Detailed Component Checks ===\n")
    
    # Neo4j Health Check
    print("Neo4j Database:")
    neo4j_checker = Neo4jHealthChecker()
    neo4j_result = await neo4j_checker.perform_check()
    
    print(f"  Status: {neo4j_result.status.value}")
    print(f"  Latency: {neo4j_result.latency_ms:.2f}ms")
    if neo4j_result.details:
        print(f"  Node Count: {neo4j_result.details.get('node_count', 'N/A')}")
        print(f"  Relationship Count: {neo4j_result.details.get('relationship_count', 'N/A')}")
        print(f"  Connection State: {neo4j_result.details.get('connection_state', 'N/A')}")
    if neo4j_result.error:
        print(f"  Error: {neo4j_result.error}")
    
    # Graphiti Health Check
    print("\nGraphiti Service:")
    graphiti_checker = GraphitiHealthChecker()
    graphiti_result = await graphiti_checker.perform_check()
    
    print(f"  Status: {graphiti_result.status.value}")
    print(f"  Latency: {graphiti_result.latency_ms:.2f}ms")
    if graphiti_result.details:
        print(f"  Endpoint: {graphiti_result.details.get('endpoint', 'N/A')}")
        print(f"  Connection State: {graphiti_result.details.get('connection_state', 'N/A')}")
    if graphiti_result.error:
        print(f"  Error: {graphiti_result.error}")
    
    # Display metrics
    print("\nHealth Metrics:")
    print(f"  Neo4j - Success Rate: {neo4j_checker.metrics.uptime_percentage:.1f}%")
    print(f"  Neo4j - Average Latency: {neo4j_checker.metrics.average_latency_ms:.2f}ms")
    print(f"  Graphiti - Success Rate: {graphiti_checker.metrics.uptime_percentage:.1f}%")
    print(f"  Graphiti - Average Latency: {graphiti_checker.metrics.average_latency_ms:.2f}ms")


async def continuous_monitoring_demo():
    """Demonstrate continuous monitoring with alerts."""
    print("\n=== Continuous Monitoring Demo ===\n")
    print("Starting health monitoring (10 second demo)...")
    print("Press Ctrl+C to stop\n")
    
    # Create system health checker
    system_checker = SystemHealthChecker()
    
    # Create monitor with 2-second interval
    monitor = HealthMonitor(
        system_checker,
        interval_seconds=2,
        failure_threshold=2
    )
    
    # Add alert callback
    def alert_handler(result, consecutive_failures):
        print(f"\n🚨 ALERT: {result.component.value} is {result.status.value}!")
        print(f"   Consecutive failures: {consecutive_failures}")
        if result.error:
            print(f"   Error: {result.error}")
        print()
    
    monitor.add_alert_callback(alert_handler)
    
    # Add status callback
    def status_handler(result):
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_icon = "✅" if result.status == HealthStatus.HEALTHY else "❌"
        print(f"[{timestamp}] {result.component.value}: {status_icon} {result.status.value} "
              f"(latency: {result.latency_ms:.0f}ms)")
    
    system_checker.add_callback(status_handler)
    
    # Start monitoring
    await monitor.start()
    
    try:
        # Run for 10 seconds
        await asyncio.sleep(10)
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    
    await monitor.stop()
    
    # Show final metrics
    print("\nFinal Metrics:")
    metrics = system_checker.get_metrics()
    print(f"  Total Checks: {metrics['total_checks']}")
    print(f"  Successful: {metrics['successful_checks']}")
    print(f"  Failed: {metrics['failed_checks']}")
    print(f"  Uptime: {metrics['uptime_percentage']:.1f}%")


async def export_health_report():
    """Export health report to JSON file."""
    print("\n=== Exporting Health Report ===\n")
    
    # Create health service
    service = HealthService()
    
    # Perform multiple checks to build history
    print("Performing health checks...")
    for i in range(3):
        await service.system_checker.perform_check()
        await asyncio.sleep(1)
    
    # Get comprehensive status
    status = await service.get_health_status()
    
    # Export to file
    report_file = "health_report.json"
    with open(report_file, "w") as f:
        json.dump(status, f, indent=2)
    
    print(f"Health report exported to: {report_file}")
    
    # Display summary
    print("\nReport Summary:")
    print(f"  Overall Status: {status['status']}")
    print(f"  Components:")
    for comp, details in status['components'].items():
        print(f"    - {comp}: {details['status']}")


async def custom_health_check_demo():
    """Demonstrate custom health check implementation."""
    print("\n=== Custom Health Check Demo ===\n")
    
    # Create custom checker with callbacks
    neo4j_checker = Neo4jHealthChecker()
    
    # Track health changes
    health_history = []
    
    def track_health(result):
        health_history.append({
            "timestamp": result.timestamp.isoformat(),
            "status": result.status.value,
            "latency": result.latency_ms
        })
    
    neo4j_checker.add_callback(track_health)
    
    # Perform multiple checks
    print("Performing 5 health checks with 1-second intervals...")
    for i in range(5):
        result = await neo4j_checker.perform_check()
        print(f"  Check {i+1}: {result.status.value} ({result.latency_ms:.1f}ms)")
        await asyncio.sleep(1)
    
    # Analyze history
    print("\nHealth History Analysis:")
    print(f"  Total checks: {len(health_history)}")
    
    healthy_count = sum(1 for h in health_history if h['status'] == 'healthy')
    print(f"  Healthy checks: {healthy_count}")
    
    avg_latency = sum(h['latency'] for h in health_history) / len(health_history)
    print(f"  Average latency: {avg_latency:.2f}ms")
    
    # Check for trends
    latencies = [h['latency'] for h in health_history]
    if latencies[-1] > latencies[0] * 1.5:
        print("  ⚠️  Warning: Latency is increasing!")
    elif latencies[-1] < latencies[0] * 0.7:
        print("  ✅ Good: Latency is decreasing!")


async def main():
    """Run all demonstrations."""
    print("Health Monitoring Examples")
    print("=" * 50)
    
    # Get configuration
    config = get_config()
    print(f"\nConfiguration:")
    print(f"  Neo4j URI: {config.neo4j.uri}")
    print(f"  Graphiti Endpoint: {config.graphiti.endpoint}")
    
    try:
        # Run demonstrations
        await simple_health_check()
        await detailed_component_checks()
        await continuous_monitoring_demo()
        await export_health_report()
        await custom_health_check_demo()
        
    except Exception as e:
        print(f"\nError during health check: {e}")
        print("\nMake sure Neo4j and Graphiti services are running!")
        return 1
    
    print("\n✅ All health monitoring examples completed!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)