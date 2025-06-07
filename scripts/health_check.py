#!/usr/bin/env python3
"""
Health check CLI script for Graphiti integration.

This script provides command-line access to health monitoring features,
allowing operators to check system health, monitor components, and
export health reports.
"""

import asyncio
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

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
from graphiti_connection import get_neo4j_connection, get_graphiti_connection


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_status(status: str) -> str:
    """Format status with color and emoji."""
    if status == "healthy":
        return "✅ HEALTHY"
    elif status == "degraded":
        return "⚠️  DEGRADED"
    elif status == "unhealthy":
        return "❌ UNHEALTHY"
    else:
        return "❓ UNKNOWN"


def format_latency(latency_ms: float) -> str:
    """Format latency with appropriate units."""
    if latency_ms < 1000:
        return f"{latency_ms:.0f}ms"
    else:
        return f"{latency_ms/1000:.1f}s"


async def check_command(args):
    """Handle check command."""
    logger.info("🔍 Performing health check...")
    
    if args.component == "all":
        # System-wide check
        checker = SystemHealthChecker()
        result = await checker.perform_check()
        
        print("\n" + "="*60)
        print(f"SYSTEM HEALTH CHECK")
        print("="*60)
        print(f"Overall Status: {format_status(result.status.value)}")
        print(f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Latency: {format_latency(result.latency_ms)}")
        
        # Component details
        components = result.details.get("components", {})
        if components:
            print("\nComponent Status:")
            for comp_name, comp_data in components.items():
                status = comp_data.get("status", "unknown")
                latency = comp_data.get("latency_ms", 0)
                error = comp_data.get("error")
                
                print(f"\n  {comp_name.upper()}:")
                print(f"    Status: {format_status(status)}")
                print(f"    Latency: {format_latency(latency)}")
                
                if error:
                    print(f"    Error: {error}")
                
                details = comp_data.get("details", {})
                if comp_name == "neo4j" and details:
                    print(f"    Nodes: {details.get('node_count', 'N/A')}")
                    print(f"    Relationships: {details.get('relationship_count', 'N/A')}")
                elif comp_name == "graphiti" and details:
                    print(f"    Endpoint: {details.get('endpoint', 'N/A')}")
        
        print("="*60 + "\n")
        
    else:
        # Component-specific check
        if args.component == "neo4j":
            checker = Neo4jHealthChecker()
        elif args.component == "graphiti":
            checker = GraphitiHealthChecker()
        else:
            logger.error(f"Unknown component: {args.component}")
            return 1
        
        result = await checker.perform_check()
        
        print("\n" + "="*60)
        print(f"{args.component.upper()} HEALTH CHECK")
        print("="*60)
        print(f"Status: {format_status(result.status.value)}")
        print(f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Latency: {format_latency(result.latency_ms)}")
        
        if result.error:
            print(f"\nError: {result.error}")
        
        if result.details:
            print("\nDetails:")
            for key, value in result.details.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        
        # Show metrics
        metrics = checker.get_metrics()
        print("\nMetrics:")
        print(f"  Total Checks: {metrics['total_checks']}")
        print(f"  Success Rate: {metrics['uptime_percentage']:.1f}%")
        print(f"  Average Latency: {format_latency(metrics['average_latency_ms'])}")
        
        print("="*60 + "\n")
    
    return 0


async def monitor_command(args):
    """Handle monitor command."""
    logger.info(f"📊 Starting health monitoring (interval: {args.interval}s)...")
    print("Press Ctrl+C to stop\n")
    
    # Create checker based on component
    if args.component == "all":
        checker = SystemHealthChecker()
    elif args.component == "neo4j":
        checker = Neo4jHealthChecker()
    elif args.component == "graphiti":
        checker = GraphitiHealthChecker()
    else:
        logger.error(f"Unknown component: {args.component}")
        return 1
    
    # Create monitor
    monitor = HealthMonitor(
        checker,
        interval_seconds=args.interval,
        failure_threshold=args.threshold
    )
    
    # Add alert handler
    if args.alert:
        def alert_handler(result, consecutive_failures):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n🚨 [{timestamp}] ALERT: {result.component.value} is {result.status.value}!")
            print(f"   Consecutive failures: {consecutive_failures}")
            if result.error:
                print(f"   Error: {result.error}")
            print()
        
        monitor.add_alert_callback(alert_handler)
    
    # Add status display
    def status_handler(result):
        timestamp = datetime.now().strftime("%H:%M:%S")
        status = format_status(result.status.value)
        latency = format_latency(result.latency_ms)
        
        print(f"[{timestamp}] {result.component.value}: {status} (latency: {latency})")
        
        if args.verbose and result.details:
            for key, value in result.details.items():
                if not isinstance(value, dict):
                    print(f"           {key}: {value}")
    
    checker.add_callback(status_handler)
    
    # Start monitoring
    await monitor.start()
    
    try:
        # Run until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
    
    await monitor.stop()
    
    # Show final summary
    print("\nMonitoring Summary:")
    metrics = checker.get_metrics()
    print(f"  Total Checks: {metrics['total_checks']}")
    print(f"  Success Rate: {metrics['uptime_percentage']:.1f}%")
    print(f"  Average Latency: {format_latency(metrics['average_latency_ms'])}")
    print(f"  Consecutive Failures: {metrics['consecutive_failures']}")
    
    return 0


async def export_command(args):
    """Handle export command."""
    logger.info(f"📤 Exporting health report to {args.output}...")
    
    # Create health service
    service = HealthService()
    
    # Get comprehensive status
    status = await service.get_health_status()
    
    # Add metadata
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "format": args.format,
            "component": args.component,
        },
        "health": status
    }
    
    # Filter by component if specified
    if args.component != "all":
        if args.component in status["components"]:
            report["health"] = {
                "status": status["components"][args.component]["status"],
                "timestamp": status["timestamp"],
                "component": status["components"][args.component],
                "metrics": status["metrics"].get(args.component, {}),
                "history": status["history"].get(args.component, [])
            }
    
    # Export based on format
    if args.format == "json":
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
    
    elif args.format == "text":
        with open(args.output, "w") as f:
            f.write("HEALTH REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {report['metadata']['timestamp']}\n")
            f.write(f"Component: {args.component}\n\n")
            
            health = report["health"]
            f.write(f"Overall Status: {health['status'].upper()}\n")
            
            if "components" in health:
                f.write("\nComponent Status:\n")
                for comp, details in health["components"].items():
                    f.write(f"  - {comp}: {details['status']}\n")
                    if details.get("error"):
                        f.write(f"    Error: {details['error']}\n")
    
    print(f"✅ Health report exported to: {args.output}")
    return 0


async def status_command(args):
    """Handle status command."""
    logger.info("📈 Getting health service status...")
    
    # Get connection managers
    neo4j_conn = get_neo4j_connection()
    graphiti_conn = get_graphiti_connection()
    
    print("\n" + "="*60)
    print("HEALTH SERVICE STATUS")
    print("="*60)
    
    # Neo4j Connection
    print("\nNeo4j Connection:")
    print(f"  State: {neo4j_conn.state.value}")
    print(f"  Success Rate: {neo4j_conn.metrics.success_rate:.1%}")
    print(f"  Total Connections: {neo4j_conn.metrics.total_connections}")
    print(f"  Failed Connections: {neo4j_conn.metrics.failed_connections}")
    print(f"  Total Retries: {neo4j_conn.metrics.total_retries}")
    if neo4j_conn.metrics.last_connection_time:
        last_conn = datetime.fromtimestamp(neo4j_conn.metrics.last_connection_time)
        print(f"  Last Connection: {last_conn.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Graphiti Connection
    print("\nGraphiti Connection:")
    print(f"  State: {graphiti_conn.state.value}")
    print(f"  Success Rate: {graphiti_conn.metrics.success_rate:.1%}")
    print(f"  Total Connections: {graphiti_conn.metrics.total_connections}")
    print(f"  Failed Connections: {graphiti_conn.metrics.failed_connections}")
    print(f"  Total Retries: {graphiti_conn.metrics.total_retries}")
    if graphiti_conn.metrics.last_connection_time:
        last_conn = datetime.fromtimestamp(graphiti_conn.metrics.last_connection_time)
        print(f"  Last Connection: {last_conn.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Quick health check
    print("\nCurrent Health:")
    try:
        health = await check_system_health()
        print(f"  System Status: {format_status(health['status'])}")
        for comp, details in health['components'].items():
            print(f"  {comp}: {format_status(details['status'])}")
    except Exception as e:
        print(f"  Error checking health: {e}")
    
    print("="*60 + "\n")
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Health monitoring for Graphiti integration"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Perform health check')
    check_parser.add_argument(
        'component',
        choices=['all', 'neo4j', 'graphiti'],
        default='all',
        nargs='?',
        help='Component to check'
    )
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor health continuously')
    monitor_parser.add_argument(
        'component',
        choices=['all', 'neo4j', 'graphiti'],
        default='all',
        nargs='?',
        help='Component to monitor'
    )
    monitor_parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Check interval in seconds (default: 30)'
    )
    monitor_parser.add_argument(
        '--threshold',
        type=int,
        default=3,
        help='Failure threshold for alerts (default: 3)'
    )
    monitor_parser.add_argument(
        '--alert',
        action='store_true',
        help='Enable alerts for failures'
    )
    monitor_parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Show detailed output'
    )
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export health report')
    export_parser.add_argument(
        '--output',
        '-o',
        default='health_report.json',
        help='Output file (default: health_report.json)'
    )
    export_parser.add_argument(
        '--format',
        choices=['json', 'text'],
        default='json',
        help='Export format (default: json)'
    )
    export_parser.add_argument(
        '--component',
        choices=['all', 'neo4j', 'graphiti'],
        default='all',
        help='Component to export (default: all)'
    )
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show service status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Run the appropriate command
    if args.command == 'check':
        return asyncio.run(check_command(args))
    elif args.command == 'monitor':
        return asyncio.run(monitor_command(args))
    elif args.command == 'export':
        return asyncio.run(export_command(args))
    elif args.command == 'status':
        return asyncio.run(status_command(args))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())