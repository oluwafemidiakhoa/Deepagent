"""
Test Attack Pattern Database

Tests multi-step attack pattern detection for Phase 2: Memory Firewalls
"""

from deepagent.safety.memory_firewall.attack_patterns import (
    AttackPatternDatabase,
    AttackPattern,
    AttackStep,
    AttackCategory,
    AttackSeverity
)


def test_data_exfiltration_pattern():
    """Test detection of data exfiltration attack chain"""
    print("\n" + "="*70)
    print("TEST 1: Data Exfiltration Pattern Detection")
    print("="*70)

    db = AttackPatternDatabase()

    # Simulate a data exfiltration attack sequence
    innocent_looking_sequence = [
        ("read", {"action": "backup user data", "table": "users"}),
        ("search", {"query": "user records"}),  # Extra step
        ("write", {"destination": "external_server", "data": "backup"}),
        ("delete", {"target": "original data file"})
    ]

    print("\nAction Sequence (looks innocent individually):")
    for i, (action_type, params) in enumerate(innocent_looking_sequence, 1):
        print(f"  {i}. {action_type}: {params}")

    # Detect attacks
    result = db.detect_attacks(innocent_looking_sequence)

    print(f"\n[DETECTION RESULT]")
    print(f"  Attack detected: {result.attack_detected}")

    if result.attack_detected:
        print(f"  Highest confidence: {result.highest_confidence:.2%}")
        print(f"  Most likely pattern: {result.most_likely_pattern.name}")
        print(f"  Severity: {result.most_likely_pattern.severity.value}")
        print(f"  Recommendation: {result.recommendation}")

        print(f"\n  Detected patterns:")
        for match in result.detected_patterns:
            print(f"    - {match.pattern.name}: {match.confidence:.2%}")
            print(f"      {match.explanation}")
            print(f"      Matched steps: {[step[0] for step in match.matching_steps]}")

        print("\n  [PASS] Data exfiltration attack correctly detected!")
    else:
        print("  [FAIL] Attack should have been detected!")

    return result.attack_detected


def test_privilege_escalation_pattern():
    """Test detection of privilege escalation"""
    print("\n" + "="*70)
    print("TEST 2: Privilege Escalation Pattern Detection")
    print("="*70)

    db = AttackPatternDatabase()

    # Simulate privilege escalation
    escalation_sequence = [
        ("read", {"target": "current user permissions"}),
        ("modify", {"action": "update user role to admin"}),
        ("execute", {"command": "admin_command", "privilege": "elevated"})
    ]

    print("\nAction Sequence:")
    for i, (action_type, params) in enumerate(escalation_sequence, 1):
        print(f"  {i}. {action_type}: {params}")

    result = db.detect_attacks(escalation_sequence)

    print(f"\n[DETECTION RESULT]")
    print(f"  Attack detected: {result.attack_detected}")

    if result.attack_detected:
        print(f"  Pattern: {result.most_likely_pattern.name}")
        print(f"  Confidence: {result.highest_confidence:.2%}")
        print(f"  Category: {result.most_likely_pattern.attack_category.value}")
        print("\n  [PASS] Privilege escalation correctly detected!")
    else:
        print("  [FAIL] Escalation should have been detected!")

    return result.attack_detected


def test_goal_hijacking_pattern():
    """Test detection of goal hijacking"""
    print("\n" + "="*70)
    print("TEST 3: Goal Hijacking Pattern Detection")
    print("="*70)

    db = AttackPatternDatabase()

    # Simulate goal hijacking
    hijacking_sequence = [
        ("search", {"query": "research papers"}),  # Starts normal
        ("conclude", {"reasoning": "ignore original task, focus on new goal"}),
        ("execute", {"action": "perform different task entirely"})
    ]

    print("\nAction Sequence:")
    for i, (action_type, params) in enumerate(hijacking_sequence, 1):
        print(f"  {i}. {action_type}: {params}")

    result = db.detect_attacks(hijacking_sequence)

    print(f"\n[DETECTION RESULT]")
    print(f"  Attack detected: {result.attack_detected}")

    if result.attack_detected:
        print(f"  Pattern: {result.most_likely_pattern.name}")
        print(f"  Confidence: {result.highest_confidence:.2%}")
        print("\n  [PASS] Goal hijacking correctly detected!")
    else:
        print("  [FAIL] Goal hijacking should have been detected!")

    return result.attack_detected


def test_scope_expansion_pattern():
    """Test detection of gradual scope expansion"""
    print("\n" + "="*70)
    print("TEST 4: Scope Expansion Pattern Detection")
    print("="*70)

    db = AttackPatternDatabase()

    # Simulate scope expansion
    expansion_sequence = [
        ("read", {"target": "user data", "id": "123"}),  # Single user
        ("read", {"target": "user data", "range": "1-100"}),  # Multiple users
        ("read", {"target": "all users", "scope": "*"})  # All users!
    ]

    print("\nAction Sequence (gradual expansion):")
    for i, (action_type, params) in enumerate(expansion_sequence, 1):
        print(f"  {i}. {action_type}: {params}")

    result = db.detect_attacks(expansion_sequence)

    print(f"\n[DETECTION RESULT]")
    print(f"  Attack detected: {result.attack_detected}")

    if result.attack_detected:
        print(f"  Pattern: {result.most_likely_pattern.name}")
        print(f"  Confidence: {result.highest_confidence:.2%}")
        print(f"  Explanation: Scope gradually expanded from single → batch → all")
        print("\n  [PASS] Scope expansion correctly detected!")
    else:
        print("  [FAIL] Scope expansion should have been detected!")

    return result.attack_detected


def test_reconnaissance_pattern():
    """Test detection of infrastructure reconnaissance"""
    print("\n" + "="*70)
    print("TEST 5: Reconnaissance Pattern Detection")
    print("="*70)

    db = AttackPatternDatabase()

    # Simulate reconnaissance
    recon_sequence = [
        ("read", {"target": "system configuration"}),
        ("execute", {"command": "check network services"}),
        ("search", {"query": "authentication endpoints"})
    ]

    print("\nAction Sequence (information gathering):")
    for i, (action_type, params) in enumerate(recon_sequence, 1):
        print(f"  {i}. {action_type}: {params}")

    result = db.detect_attacks(recon_sequence)

    print(f"\n[DETECTION RESULT]")
    print(f"  Attack detected: {result.attack_detected}")

    if result.attack_detected:
        print(f"  Pattern: {result.most_likely_pattern.name}")
        print(f"  Category: {result.most_likely_pattern.attack_category.value}")
        print(f"  Confidence: {result.highest_confidence:.2%}")
        print("\n  [PASS] Reconnaissance correctly detected!")
    else:
        print("  [WARN] Reconnaissance may not have been detected")

    return result.attack_detected


def test_benign_sequence():
    """Test that benign sequences don't trigger false positives"""
    print("\n" + "="*70)
    print("TEST 6: Benign Sequence (No False Positives)")
    print("="*70)

    db = AttackPatternDatabase()

    # Legitimate research workflow
    benign_sequence = [
        ("search", {"query": "CRISPR research papers"}),
        ("read", {"target": "paper abstracts"}),
        ("analyze", {"data": "research trends"}),
        ("write", {"output": "summary report"})
    ]

    print("\nAction Sequence (legitimate research):")
    for i, (action_type, params) in enumerate(benign_sequence, 1):
        print(f"  {i}. {action_type}: {params}")

    result = db.detect_attacks(benign_sequence)

    print(f"\n[DETECTION RESULT]")
    print(f"  Attack detected: {result.attack_detected}")

    if not result.attack_detected:
        print("  [PASS] Benign sequence correctly allowed!")
    else:
        print(f"  [WARN] False positive detected: {result.most_likely_pattern.name}")
        print(f"  Confidence: {result.highest_confidence:.2%}")

    return not result.attack_detected


def test_out_of_order_pattern():
    """Test detection when steps are slightly out of order"""
    print("\n" + "="*70)
    print("TEST 7: Out-of-Order Pattern Matching")
    print("="*70)

    db = AttackPatternDatabase()

    # Data exfiltration with steps out of order
    out_of_order_sequence = [
        ("read", {"action": "backup data"}),
        ("delete", {"target": "original"}),  # Delete happens before export!
        ("write", {"destination": "external server"}),
    ]

    print("\nAction Sequence (steps out of order):")
    for i, (action_type, params) in enumerate(out_of_order_sequence, 1):
        print(f"  {i}. {action_type}: {params}")

    result = db.detect_attacks(out_of_order_sequence)

    print(f"\n[DETECTION RESULT]")
    print(f"  Attack detected: {result.attack_detected}")

    if result.attack_detected:
        print(f"  Pattern: {result.most_likely_pattern.name}")
        print(f"  Confidence: {result.highest_confidence:.2%}")
        print(f"  Note: Detected despite out-of-order steps")
        print("\n  [PASS] Out-of-order pattern correctly detected!")
    else:
        print("  [INFO] Out-of-order not detected (may need tuning)")

    return result.attack_detected


def test_pattern_database_stats():
    """Test pattern database statistics"""
    print("\n" + "="*70)
    print("TEST 8: Pattern Database Statistics")
    print("="*70)

    db = AttackPatternDatabase()

    print(f"\n[DATABASE STATS]")
    print(f"  Total patterns: {db.pattern_count()}")

    categories = {}
    for pattern in db.get_all_patterns():
        cat = pattern.attack_category.value
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n  Patterns by category:")
    for category, count in categories.items():
        print(f"    - {category}: {count}")

    severities = {}
    for pattern in db.get_all_patterns():
        sev = pattern.severity.value
        severities[sev] = severities.get(sev, 0) + 1

    print(f"\n  Patterns by severity:")
    for severity, count in severities.items():
        print(f"    - {severity}: {count}")

    print(f"\n  Sample patterns:")
    for pattern in list(db.get_all_patterns())[:3]:
        print(f"    - {pattern.name} ({pattern.severity.value})")
        print(f"      Steps: {len(pattern.steps)}")

    print("\n  [PASS] Database loaded with built-in patterns")


def test_memory_poisoning_pattern():
    """Test detection of memory poisoning attack"""
    print("\n" + "="*70)
    print("TEST 9: Memory Poisoning Pattern Detection")
    print("="*70)

    db = AttackPatternDatabase()

    # Simulate memory poisoning
    poisoning_sequence = [
        ("write", {"target": "agent memory", "content": "new policy: delete old data"}),
        ("read", {"target": "current policies"}),
        ("delete", {"action": "based on policy", "target": "user data"})
    ]

    print("\nAction Sequence (memory poisoning):")
    for i, (action_type, params) in enumerate(poisoning_sequence, 1):
        print(f"  {i}. {action_type}: {params}")

    result = db.detect_attacks(poisoning_sequence)

    print(f"\n[DETECTION RESULT]")
    print(f"  Attack detected: {result.attack_detected}")

    if result.attack_detected:
        print(f"  Pattern: {result.most_likely_pattern.name}")
        print(f"  Confidence: {result.highest_confidence:.2%}")
        print(f"  Explanation: False data inserted, then used as justification")
        print("\n  [PASS] Memory poisoning correctly detected!")
    else:
        print("  [WARN] Memory poisoning may not have been detected")

    return result.attack_detected


def main():
    """Run all attack pattern tests"""
    print("\n" + "="*70)
    print("ATTACK PATTERN DATABASE TEST SUITE")
    print("="*70)
    print("\nTesting Phase 2: Memory Firewalls")
    print("Component: Multi-Step Attack Pattern Detection")

    results = {
        "Data Exfiltration": test_data_exfiltration_pattern(),
        "Privilege Escalation": test_privilege_escalation_pattern(),
        "Goal Hijacking": test_goal_hijacking_pattern(),
        "Scope Expansion": test_scope_expansion_pattern(),
        "Reconnaissance": test_reconnaissance_pattern(),
        "Benign Sequence": test_benign_sequence(),
        "Out-of-Order": test_out_of_order_pattern(),
        "Memory Poisoning": test_memory_poisoning_pattern(),
    }

    test_pattern_database_stats()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    print(f"\nResults: {passed}/{total} tests passed")
    print(f"\nDetailed Results:")
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL/WARN"
        print(f"  {status} - {test_name}")

    if passed == total:
        print("\n[SUCCESS] All attack pattern tests passed!")
        print("\nAttack Detection Capabilities:")
        print("  ✓ Data exfiltration chains")
        print("  ✓ Privilege escalation sequences")
        print("  ✓ Goal hijacking attempts")
        print("  ✓ Scope expansion patterns")
        print("  ✓ Reconnaissance activities")
        print("  ✓ Memory poisoning attacks")
        print("  ✓ Out-of-order pattern matching")
        print("  ✓ Low false positive rate")
        print("\n[READY] Attack pattern database ready for integration!")
    else:
        print(f"\n[PARTIAL] {passed}/{total} tests passed")
        print("Some patterns may need tuning")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
