import json
import os
from typing import List, Dict, Any
from evaluations.evalute import livecodebench_evaluate, livecodebench_execute_internal_test


def evaluate_livecodebench_results(results_path: str, output_path: str):
    """Evaluate LiveCodeBench results and generate summary"""
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error reading results file: {e}")
        return
    
    # Analyze evaluation results
    total_problems = len(results)
    passed_problems = 0
    failed_problems = 0
    error_problems = 0
    
    problem_details = []
    
    for result in results:
        problem_id = result.get('id', 'unknown')
        passed = result.get('passed', False)
        error = result.get('error', False)
        
        if error:
            error_problems += 1
            status = "ERROR"
        elif passed:
            passed_problems += 1
            status = "PASSED"
        else:
            failed_problems += 1
            status = "FAILED"
        
        problem_details.append({
            'id': problem_id,
            'status': status,
            'language': result.get('language', 'unknown'),
            'strategy': result.get('strategy', 'unknown')
        })
    
    # Generate summary
    summary = {
        'total_problems': total_problems,
        'passed_problems': passed_problems,
        'failed_problems': failed_problems,
        'error_problems': error_problems,
        'pass_rate': (passed_problems / total_problems * 100) if total_problems > 0 else 0,
        'problem_details': problem_details
    }
    
    # Save results
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"LiveCodeBench evaluation completed:")
        print(f"Total problems: {total_problems}")
        print(f"Passed: {passed_problems}")
        print(f"Failed: {failed_problems}")
        print(f"Errors: {error_problems}")
        print(f"Pass rate: {summary['pass_rate']:.2f}%")
    except Exception as e:
        print(f"Error saving summary: {e}")


def generate_livecodebench_report(results_path: str, report_path: str):
    """Generate detailed LiveCodeBench report"""
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error reading results file: {e}")
        return
    
    # Language performance analysis
    language_stats = {}
    strategy_stats = {}
    
    for result in results:
        language = result.get('language', 'unknown')
        strategy = result.get('strategy', 'unknown')
        passed = result.get('passed', False)
        
        # Language statistics
        if language not in language_stats:
            language_stats[language] = {'total': 0, 'passed': 0}
        language_stats[language]['total'] += 1
        if passed:
            language_stats[language]['passed'] += 1
        
        # Strategy statistics
        if strategy not in strategy_stats:
            strategy_stats[strategy] = {'total': 0, 'passed': 0}
        strategy_stats[strategy]['total'] += 1
        if passed:
            strategy_stats[strategy]['passed'] += 1
    
    # Generate report
    report = {
        'overview': {
            'total_problems': len(results),
            'evaluation_date': results[0].get('timestamp', 'unknown') if results else 'unknown'
        },
        'language_performance': language_stats,
        'strategy_performance': strategy_stats,
        'detailed_results': results
    }
    
    # Save results
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"LiveCodeBench report generated successfully: {report_path}")
    except Exception as e:
        print(f"Error saving report: {e}")


def validate_livecodebench_data(data_path: str) -> bool:
    """Validate LiveCodeBench data format"""
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return False
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    data = json.loads(line)
                    
                    # Required fields validation
                    required_fields = ['id', 'title', 'description', 'test_cases']
                    for field in required_fields:
                        if field not in data:
                            print(f"Line {line_num}: Missing required field '{field}'")
                            return False
                    
                    # Test cases format validation
                    test_cases = data.get('test_cases', [])
                    if not isinstance(test_cases, list) or len(test_cases) == 0:
                        print(f"Line {line_num}: Invalid test_cases format")
                        return False
                    
                    for tc in test_cases:
                        if not isinstance(tc, dict) or 'input' not in tc or 'output' not in tc:
                            print(f"Line {line_num}: Invalid test case format")
                            return False
        
        print("LiveCodeBench data format validation completed")
        return True
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return False
    except Exception as e:
        print(f"Error during data validation: {e}")
        return False


def analyze_livecodebench_performance(results_path: str) -> Dict[str, Any]:
    """Analyze LiveCodeBench performance metrics"""
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return {}
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error reading results file: {e}")
        return {}
    
    # Performance metrics
    metrics = {
        'total_problems': len(results),
        'success_rate': 0,
        'average_attempts': 0,
        'language_distribution': {},
        'difficulty_analysis': {},
        'platform_analysis': {}
    }
    
    if not results:
        return metrics
    
    total_attempts = 0
    successful_problems = 0
    
    for result in results:
        # Count attempts
        attempts = result.get('no_of_try', 1)
        total_attempts += attempts
        
        # Count successful problems
        if result.get('passed', False):
            successful_problems += 1
        
        # Language distribution
        language = result.get('language', 'unknown')
        metrics['language_distribution'][language] = metrics['language_distribution'].get(language, 0) + 1
        
        # Difficulty analysis (if available)
        difficulty = result.get('difficulty', 'unknown')
        if difficulty != 'unknown':
            if difficulty not in metrics['difficulty_analysis']:
                metrics['difficulty_analysis'][difficulty] = {'total': 0, 'passed': 0}
            metrics['difficulty_analysis'][difficulty]['total'] += 1
            if result.get('passed', False):
                metrics['difficulty_analysis'][difficulty]['passed'] += 1
        
        # Platform analysis (if available)
        platform = result.get('platform', 'unknown')
        if platform != 'unknown':
            if platform not in metrics['platform_analysis']:
                metrics['platform_analysis'][platform] = {'total': 0, 'passed': 0}
            metrics['platform_analysis'][platform]['total'] += 1
            if result.get('passed', False):
                metrics['platform_analysis'][platform]['passed'] += 1
    
    # Calculate final metrics
    metrics['success_rate'] = (successful_problems / len(results)) * 100 if results else 0
    metrics['average_attempts'] = total_attempts / len(results) if results else 0
    
    return metrics


def export_livecodebench_results(results_path: str, export_path: str, format: str = 'json'):
    """Export LiveCodeBench results in various formats"""
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error reading results file: {e}")
        return
    
    if format.lower() == 'csv':
        import csv
        try:
            with open(export_path, 'w', newline='', encoding='utf-8') as csvfile:
                if results:
                    fieldnames = results[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)
            print(f"Results exported to CSV: {export_path}")
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
    
    elif format.lower() == 'json':
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results exported to JSON: {export_path}")
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
    
    else:
        print(f"Unsupported export format: {format}")
