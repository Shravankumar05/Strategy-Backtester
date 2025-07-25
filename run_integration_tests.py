import sys
import os
import subprocess
from pathlib import Path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def run_integration_tests():
    print("🚀 Running Trading Strategy Backtester Integration Tests")
    print("=" * 60)
    
    test_files = ["tests/integration/test_end_to_end_integration.py", "tests/integration/test_ui_integration.py"]
    all_passed = True
    results = {}
    
    for test_file in test_files:
        print(f"\n📋 Running {test_file}...")
        print("-" * 40)
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test_file, 
                "-v", 
                "--tb=short",
                "--no-header"
            ], capture_output=True, text=True, cwd=current_dir)
            
            if result.returncode == 0:
                print(f"✅ {test_file} - PASSED")
                results[test_file] = "PASSED"
            else:
                print(f"❌ {test_file} - FAILED")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                results[test_file] = "FAILED"
                all_passed = False
                
        except Exception as e:
            print(f"💥 {test_file} - ERROR: {e}")
            results[test_file] = f"ERROR: {e}"
            all_passed = False
    
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    for test_file, status in results.items():
        status_icon = "✅" if status == "PASSED" else "❌"
        print(f"{status_icon} {test_file}: {status}")
    
    if all_passed:
        print("ALL INTEGRATION TESTS PASSED!")
        print("✨ The system is ready for end-to-end usage.")
        return True
    else:
        print("SOME INTEGRATION TESTS FAILED")
        print("🔧 Please review the errors above and fix the issues.")
        return False

def test_basic_imports():
    """Test that all required modules can be imported."""
    print("\n🔍 Testing Basic Module Imports...")
    print("-" * 40)
    
    modules_to_test = [
        "backtester.data.yfinance_fetcher",
        "backtester.data.cache_manager", 
        "backtester.strategy.ma_crossover",
        "backtester.strategy.rsi_strategy",
        "backtester.simulation.engine",
        "backtester.simulation.config",
        "backtester.metrics.performance",
        "backtester.visualization.visualizer",
        "backtester.ui.utils.session_state"
    ]
    
    all_imports_ok = True
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            all_imports_ok = False
        except Exception as e:
            print(f"⚠️  {module_name}: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def test_app_startup():
    print("Testing App Startup...")
    print("-" * 40)
    
    try:
        from backtester.ui.app import main
        print("✅ Main app imports successfully")
        from backtester.ui.app import (validate_backtest_config, create_strategy_from_config, create_simulation_config)
        print("✅ Key integration functions exist")
        return True
        
    except Exception as e:
        print(f"❌ App startup failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Trading Strategy Backtester - Integration Test Suite")
    print("=" * 60)
    imports_ok = test_basic_imports()
    app_ok = test_app_startup()
    if imports_ok and app_ok:
        tests_ok = run_integration_tests()
        
        if tests_ok:
            print("INTEGRATION COMPLETE! :)")
            print("🚀 You can now run the app with: python main.py")
            sys.exit(0)
        else:
            print("INTEGRATION INCOMPLETE :(")
            sys.exit(1)
    else:
        print("BASIC SETUP FAILED :(")
        print("Please fix import/startup issues before running integration tests.")
        sys.exit(1)