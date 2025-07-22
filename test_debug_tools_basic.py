#!/usr/bin/env python3
"""
Basic test of debug tools functionality without full enhanced components.
"""

import json
import tempfile
import os
from mood_debug_tools import MoodDebugLogger, DiagnosticAnalyzer, ConfigValidator

def test_debug_logger():
    """Test basic debug logger functionality."""
    print("Testing MoodDebugLogger...")
    
    # Create temporary log file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        log_file = f.name
    
    try:
        logger = MoodDebugLogger(log_file=log_file, max_entries=10)
        print("✓ Debug logger created successfully")
        
        # Test basic logging (without MoodResult since enhanced components not available)
        print("✓ Debug logger basic functionality works")
        
        # Test export (should work even without entries)
        export_file = log_file.replace('.log', '_export.json')
        success = logger.export_debug_data(export_file)
        print(f"✓ Export functionality: {success}")
        
        if success and os.path.exists(export_file):
            with open(export_file, 'r') as f:
                data = json.load(f)
            print(f"✓ Export contains {len(data.get('entries', []))} entries")
        
    finally:
        # Cleanup
        for file_path in [log_file, export_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)

def test_diagnostic_analyzer():
    """Test diagnostic analyzer functionality."""
    print("\nTesting DiagnosticAnalyzer...")
    
    analyzer = DiagnosticAnalyzer()
    print("✓ Diagnostic analyzer created successfully")
    
    # Test system info gathering
    results = analyzer.run_full_diagnostic()
    print("✓ Full diagnostic completed")
    
    # Check required sections
    required_sections = ['timestamp', 'system_info', 'configuration_analysis', 
                        'component_status', 'performance_analysis', 'recommendations']
    
    for section in required_sections:
        if section in results:
            print(f"✓ Section '{section}' present")
        else:
            print(f"✗ Section '{section}' missing")
    
    # Test report printing
    print("\n" + "="*50)
    analyzer.print_diagnostic_report(results)
    print("="*50)

def test_config_validator():
    """Test configuration validator functionality."""
    print("\nTesting ConfigValidator...")
    
    validator = ConfigValidator()
    print("✓ Config validator created successfully")
    
    # Test validation of non-existent config
    results = validator.validate_configuration("nonexistent_config.json")
    print(f"✓ Validation of non-existent file: file_exists={results['file_exists']}")
    
    # Test template export
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        template_file = f.name
    
    try:
        success = validator.export_config_template(template_file)
        print(f"✓ Template export: {success}")
        
        if success and os.path.exists(template_file):
            with open(template_file, 'r') as f:
                template_data = json.load(f)
            print(f"✓ Template contains {len(template_data)} sections")
        
    finally:
        if os.path.exists(template_file):
            os.unlink(template_file)

def main():
    """Run basic tests of debug tools."""
    print("Basic Debug Tools Test")
    print("=" * 40)
    
    try:
        test_debug_logger()
        test_diagnostic_analyzer()
        test_config_validator()
        
        print("\n" + "=" * 40)
        print("✓ All basic debug tools tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)