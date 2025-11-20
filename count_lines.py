#!/usr/bin/env python3
"""Count lines of code in the deepagent framework"""

from pathlib import Path

root = Path('deepagent')
foundation_counts = {}
total = 0

for python_file in root.rglob('*.py'):
    if '__pycache__' in str(python_file):
        continue

    lines = len(python_file.read_text(encoding='utf-8').splitlines())
    total += lines

    # Determine foundation
    path_parts = python_file.parts
    if 'safety' in path_parts:
        if 'memory_firewall' in path_parts:
            foundation = 'Foundation #2: Memory Firewalls'
        else:
            foundation = 'Foundation #1: Action-Level Safety'
    elif 'provenance' in path_parts:
        foundation = 'Foundation #3: Identity & Provenance'
    elif 'sandbox' in path_parts:
        foundation = 'Foundation #4: Execution Sandboxing'
    elif 'behavioral' in path_parts:
        foundation = 'Foundation #5: Behavioral Monitoring'
    elif 'supervision' in path_parts:
        foundation = 'Foundation #6: Meta-Agent Supervision'
    elif 'audit' in path_parts:
        foundation = 'Foundation #7: Audit Logs & Forensics'
    elif 'purpose' in path_parts:
        foundation = 'Foundation #8: Purpose-Bound Agents'
    elif 'intent' in path_parts:
        foundation = 'Foundation #9: Global Intent & Context'
    elif 'deception' in path_parts:
        foundation = 'Foundation #10: Deception Detection'
    elif 'autonomy' in path_parts:
        foundation = 'Foundation #11: Risk-Adaptive Autonomy'
    elif 'governance' in path_parts:
        foundation = 'Foundation #12: Human Governance'
    else:
        foundation = 'Core/Other'

    if foundation not in foundation_counts:
        foundation_counts[foundation] = 0
    foundation_counts[foundation] += lines

print("SafeDeepAgent Framework - Line Count Summary")
print("=" * 60)
print()

for foundation in sorted(foundation_counts.keys()):
    print(f"{foundation}: {foundation_counts[foundation]} lines")

print()
print("=" * 60)
print(f"TOTAL: {total} lines of production code")
print("=" * 60)
