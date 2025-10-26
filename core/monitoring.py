"""
Production Monitoring System for RAG Pipeline
Implements comprehensive metrics collection, performance tracking, and alerting
"""

import time
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import contextmanager
import traceback
from enum import Enum
import uuid


class MetricType(Enum):
    """Types of metrics to track."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricData:
    """Data structure for metrics."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Optional[Dict[str, str]] = None
    unit: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation_name: str
    duration_ms: float
    success: bool
    timestamp: datetime
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    source: str
    metadata: Optional[Dict[str, Any]] = None
    resolved: bool = False


class MetricsCollector:
    """Thread-safe metrics collection system."""
    
    def __init__(self, max_history_size: int = 10000):
        self.metrics = defaultdict(deque)
        self.max_history_size = max_history_size
        self._lock = threading.Lock()
        
    def record_metric(self, 
                     name: str, 
                     value: float, 
                     metric_type: MetricType = MetricType.GAUGE,
                     labels: Optional[Dict[str, str]] = None,
                     unit: Optional[str] = None):
        """Record a metric with thread safety."""
        metric = MetricData(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            labels=labels or {},
            unit=unit
        )
        
        with self._lock:
            self.metrics[name].append(metric)
            # Maintain history size limit
            if len(self.metrics[name]) > self.max_history_size:
                self.metrics[name].popleft()
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, labels)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, unit: Optional[str] = None):
        """Set a gauge metric."""
        self.record_metric(name, value, MetricType.GAUGE, labels, unit)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        self.record_metric(name, value, MetricType.HISTOGRAM, labels)
    
    def get_metrics(self, name: str, since: Optional[datetime] = None) -> List[MetricData]:
        """Get metrics for a specific name."""
        with self._lock:
            metrics = list(self.metrics[name])
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            return metrics
    
    def get_latest_metric(self, name: str) -> Optional[MetricData]:
        """Get the latest metric value."""
        with self._lock:
            if name in self.metrics and self.metrics[name]:
                return self.metrics[name][-1]
            return None
    
    def get_metric_summary(self, name: str, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        metrics = self.get_metrics(name, since)
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
            "first_timestamp": metrics[0].timestamp,
            "latest_timestamp": metrics[-1].timestamp
        }


class PerformanceTracker:
    """Track performance metrics for operations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.performance_history = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    @contextmanager
    def track_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager to track operation performance."""
        start_time = time.time()
        success = True
        error_message = None
        
        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Record performance metric
            perf_metric = PerformanceMetrics(
                operation_name=operation_name,
                duration_ms=duration_ms,
                success=success,
                timestamp=datetime.now(),
                error_message=error_message,
                metadata=metadata
            )
            
            with self._lock:
                self.performance_history.append(perf_metric)
            
            # Record to metrics collector
            self.metrics_collector.record_metric(
                f"operation_duration_{operation_name}",
                duration_ms,
                MetricType.HISTOGRAM,
                labels={"success": str(success)},
                unit="ms"
            )
            
            self.metrics_collector.increment_counter(
                f"operation_count_{operation_name}",
                labels={"success": str(success)}
            )
    
    def get_operation_stats(self, operation_name: str, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        with self._lock:
            operations = [
                p for p in self.performance_history 
                if p.operation_name == operation_name and (not since or p.timestamp >= since)
            ]
        
        if not operations:
            return {}
        
        durations = [op.duration_ms for op in operations]
        success_count = sum(1 for op in operations if op.success)
        
        return {
            "total_operations": len(operations),
            "success_count": success_count,
            "failure_count": len(operations) - success_count,
            "success_rate": success_count / len(operations),
            "avg_duration_ms": sum(durations) / len(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "latest_timestamp": operations[-1].timestamp
        }


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, max_alerts: int = 1000):
        self.alerts = deque(maxlen=max_alerts)
        self.alert_handlers = []
        self._lock = threading.Lock()
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add a custom alert handler."""
        self.alert_handlers.append(handler)
    
    def create_alert(self, 
                    level: AlertLevel, 
                    message: str, 
                    source: str,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new alert."""
        alert = Alert(
            id=str(uuid.uuid4()),
            level=level,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.alerts.append(alert)
        
        # Trigger alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"Alert handler failed: {e}")
        
        return alert.id
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    return True
        return False
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            alerts = [a for a in self.alerts if not a.resolved]
            if level:
                alerts = [a for a in alerts if a.level == level]
            return list(alerts)


class SystemHealthMonitor:
    """Monitor overall system health."""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.health_checks = {}
        self.monitoring_enabled = True
        
    def register_health_check(self, name: str, check_function: Callable[[], bool], interval_seconds: int = 60):
        """Register a health check function."""
        self.health_checks[name] = {
            "function": check_function,
            "interval": interval_seconds,
            "last_check": None,
            "last_result": None
        }
    
    def run_health_checks(self):
        """Run all registered health checks."""
        if not self.monitoring_enabled:
            return
        
        current_time = datetime.now()
        
        for name, check_info in self.health_checks.items():
            last_check = check_info["last_check"]
            interval = timedelta(seconds=check_info["interval"])
            
            # Check if it's time to run this health check
            if last_check is None or current_time - last_check >= interval:
                try:
                    result = check_info["function"]()
                    check_info["last_check"] = current_time
                    check_info["last_result"] = result
                    
                    # Record health check metric
                    self.metrics_collector.set_gauge(
                        f"health_check_{name}",
                        1.0 if result else 0.0,
                        labels={"status": "pass" if result else "fail"}
                    )
                    
                    # Create alert if health check failed
                    if not result:
                        self.alert_manager.create_alert(
                            AlertLevel.WARNING,
                            f"Health check '{name}' failed",
                            "SystemHealthMonitor",
                            {"health_check": name}
                        )
                        
                except Exception as e:
                    logging.error(f"Health check '{name}' error: {e}")
                    self.alert_manager.create_alert(
                        AlertLevel.ERROR,
                        f"Health check '{name}' threw exception: {e}",
                        "SystemHealthMonitor",
                        {"health_check": name, "error": str(e)}
                    )
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health_status = {}
        
        for name, check_info in self.health_checks.items():
            health_status[name] = {
                "last_check": check_info["last_check"],
                "last_result": check_info["last_result"],
                "interval_seconds": check_info["interval"]
            }
        
        # Overall health
        recent_failures = len([
            check for check in health_status.values() 
            if check["last_result"] is False
        ])
        
        health_status["overall"] = {
            "healthy": recent_failures == 0,
            "failed_checks": recent_failures,
            "total_checks": len(self.health_checks)
        }
        
        return health_status


class RAGMonitoringSystem:
    """Main monitoring system for RAG pipeline."""
    
    def __init__(self, enable_alerts: bool = True):
        self.metrics_collector = MetricsCollector()
        self.performance_tracker = PerformanceTracker(self.metrics_collector)
        self.alert_manager = AlertManager() if enable_alerts else None
        self.health_monitor = SystemHealthMonitor(self.metrics_collector, self.alert_manager) if enable_alerts else None
        
        # Setup default alert handlers
        if enable_alerts:
            self.alert_manager.add_alert_handler(self._log_alert)
        
        # Register default health checks
        if self.health_monitor:
            self._register_default_health_checks()
    
    def _log_alert(self, alert: Alert):
        """Default alert handler that logs alerts."""
        log_level = {
            AlertLevel.INFO: logging.info,
            AlertLevel.WARNING: logging.warning,
            AlertLevel.ERROR: logging.error,
            AlertLevel.CRITICAL: logging.critical
        }
        
        log_level[alert.level](
            f"ALERT [{alert.level.value.upper()}] {alert.source}: {alert.message}"
        )
    
    def _register_default_health_checks(self):
        """Register default health checks."""
        
        def memory_check():
            """Basic memory usage check."""
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                return memory_percent < 90  # Alert if memory usage > 90%
            except ImportError:
                return True  # Skip if psutil not available
        
        def disk_check():
            """Basic disk usage check."""
            try:
                import psutil
                disk_percent = psutil.disk_usage('/').percent
                return disk_percent < 90  # Alert if disk usage > 90%
            except (ImportError, OSError):
                return True  # Skip if psutil not available or path invalid
        
        if self.health_monitor:
            self.health_monitor.register_health_check("memory_usage", memory_check, 300)  # 5 minutes
            self.health_monitor.register_health_check("disk_usage", disk_check, 300)  # 5 minutes
    
    # Convenience methods
    def track_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for tracking operations."""
        return self.performance_tracker.track_operation(operation_name, metadata)
    
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE, **kwargs):
        """Record a metric."""
        self.metrics_collector.record_metric(name, value, metric_type, **kwargs)
    
    def create_alert(self, level: AlertLevel, message: str, source: str, **kwargs):
        """Create an alert."""
        if self.alert_manager:
            return self.alert_manager.create_alert(level, message, source, **kwargs)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics_summary": {},
            "recent_performance": {},
            "active_alerts": [],
            "system_health": {}
        }
        
        # Recent metrics (last hour)
        since = datetime.now() - timedelta(hours=1)
        
        # Get performance data
        if hasattr(self.performance_tracker, 'performance_history'):
            recent_ops = [
                op for op in self.performance_tracker.performance_history 
                if op.timestamp >= since
            ]
            
            # Group by operation
            ops_by_name = defaultdict(list)
            for op in recent_ops:
                ops_by_name[op.operation_name].append(op)
            
            for op_name, ops in ops_by_name.items():
                success_count = sum(1 for op in ops if op.success)
                durations = [op.duration_ms for op in ops]
                
                dashboard_data["recent_performance"][op_name] = {
                    "total_operations": len(ops),
                    "success_rate": success_count / len(ops) if ops else 0,
                    "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                    "latest_timestamp": max(op.timestamp for op in ops) if ops else None
                }
        
        # Get active alerts
        if self.alert_manager:
            active_alerts = self.alert_manager.get_active_alerts()
            dashboard_data["active_alerts"] = [
                {
                    "id": alert.id,
                    "level": alert.level.value,
                    "message": alert.message,
                    "source": alert.source,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in active_alerts
            ]
        
        # Get system health
        if self.health_monitor:
            dashboard_data["system_health"] = self.health_monitor.get_system_health()
        
        return dashboard_data


# Global monitoring instance
_global_monitor: Optional[RAGMonitoringSystem] = None


def get_monitor() -> RAGMonitoringSystem:
    """Get the global monitoring instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = RAGMonitoringSystem()
    return _global_monitor


def initialize_monitoring(enable_alerts: bool = True) -> RAGMonitoringSystem:
    """Initialize the global monitoring system."""
    global _global_monitor
    _global_monitor = RAGMonitoringSystem(enable_alerts)
    return _global_monitor


# Decorators for easy monitoring
def monitor_performance(operation_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            with monitor.track_operation(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Initialize monitoring
    monitor = initialize_monitoring()
    
    # Example operations
    @monitor_performance("example_operation")
    def example_function():
        time.sleep(0.1)  # Simulate work
        return "success"
    
    # Test metrics
    print("=== Testing RAG Monitoring System ===")
    
    # Record some metrics
    monitor.record_metric("documents_processed", 10, MetricType.COUNTER)
    monitor.record_metric("cpu_usage", 45.5, MetricType.GAUGE, unit="%")
    
    # Test performance tracking
    for i in range(5):
        result = example_function()
    
    # Test alerts
    if monitor.alert_manager:
        monitor.create_alert(
            AlertLevel.WARNING,
            "High memory usage detected",
            "MemoryMonitor",
            metadata={"usage_percent": 85}
        )
    
    # Run health checks
    if monitor.health_monitor:
        monitor.health_monitor.run_health_checks()
    
    # Get dashboard data
    dashboard = monitor.get_dashboard_data()
    print("\nDashboard Data:")
    print(json.dumps(dashboard, indent=2, default=str))
    
    print("\n=== Monitoring System Test Complete ===")