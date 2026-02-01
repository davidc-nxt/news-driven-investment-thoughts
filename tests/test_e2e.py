#!/usr/bin/env python3
"""End-to-end test script for Investment Advisor CLI."""

import subprocess
import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def run_command(cmd: str, description: str) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    console.print(f"\n[bold blue]▶ {description}[/bold blue]")
    console.print(f"[dim]$ {cmd}[/dim]")
    
    start = time.time()
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        env={**subprocess.os.environ, "PYTHONWARNINGS": "ignore"}
    )
    elapsed = time.time() - start
    
    if result.returncode == 0:
        console.print(f"[green]✓ Passed[/green] ({elapsed:.2f}s)")
        return True, result.stdout
    else:
        console.print(f"[red]✗ Failed[/red] ({elapsed:.2f}s)")
        console.print(f"[red]{result.stderr}[/red]")
        return False, result.stderr


def main():
    console.print(Panel.fit(
        "[bold cyan]Investment Advisor CLI - End-to-End Test Suite[/bold cyan]",
        border_style="cyan"
    ))
    
    tests = []
    
    # Test 1: Check system status
    success, output = run_command(
        "invest status",
        "Test 1: System Status Check"
    )
    tests.append(("System Status", success, "Database connected" in output))
    
    # Test 2: List tickers
    success, output = run_command(
        "invest tickers list",
        "Test 2: List Tracked Tickers"
    )
    tests.append(("List Tickers", success, "NVDA" in output or success))
    
    # Test 3: Fetch data for a specific ticker
    success, output = run_command(
        "invest fetch -t AAPL",
        "Test 3: Fetch Data for AAPL"
    )
    tests.append(("Fetch AAPL Data", success, "Fetch complete" in output))
    
    # Test 4: Semantic search
    success, output = run_command(
        'invest search "technology innovation"',
        "Test 4: Semantic Search"
    )
    tests.append(("Semantic Search", success, "Search Results" in output))
    
    # Test 5: Generate investment advice
    success, output = run_command(
        "invest advise AAPL",
        "Test 5: Generate Investment Advice"
    )
    tests.append(("Investment Advice", success, "Investment Analysis" in output or "HOLD" in output or "BUY" in output or "SELL" in output))
    
    # Test 6: Filter search by ticker
    success, output = run_command(
        'invest search "earnings" -t AAPL',
        "Test 6: Filtered Search by Ticker"
    )
    tests.append(("Filtered Search", success, "Search Results" in output or "No results" in output))
    
    # Print summary
    console.print("\n")
    table = Table(title="E2E Test Results Summary")
    table.add_column("Test", style="white")
    table.add_column("Execution", style="cyan")
    table.add_column("Validation", style="magenta")
    
    passed = 0
    for name, exec_ok, valid_ok in tests:
        exec_status = "✓" if exec_ok else "✗"
        valid_status = "✓" if valid_ok else "✗"
        exec_style = "green" if exec_ok else "red"
        valid_style = "green" if valid_ok else "red"
        table.add_row(name, f"[{exec_style}]{exec_status}[/{exec_style}]", f"[{valid_style}]{valid_status}[/{valid_style}]")
        if exec_ok and valid_ok:
            passed += 1
    
    console.print(table)
    
    total = len(tests)
    if passed == total:
        console.print(f"\n[bold green]✓ All {total} tests passed![/bold green]")
        return 0
    else:
        console.print(f"\n[bold red]✗ {passed}/{total} tests passed[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
