#!/usr/bin/env python3
"""
Command-line automated testing for the RAG system
Tests the entire pipeline on DFM Handbook documents
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from pages.automated_testing import AutomatedTestingSystem

def run_full_system_test():
    """Run the complete automated testing system"""
    print("=" * 80)
    print("🚀 RAG SYSTEM END-TO-END TESTING")
    print("=" * 80)

    # Initialize testing system
    testing_system = AutomatedTestingSystem()

    # Define test folders
    test_folders = [
        "/Users/spandankewte/Downloads/DFM Handbook data",
        "/opt/anaconda3/Phase-3-Final-master/data"
    ]

    print(f"📁 Test folders: {test_folders}")

    # Find all PDF files
    pdf_files = testing_system.find_test_documents(test_folders)
    print(f"📄 Found {len(pdf_files)} PDF files to process:")

    for i, pdf_file in enumerate(pdf_files, 1):
        folder_name = "DFM Handbook" if "DFM Handbook data" in pdf_file else "Additional Data"
        print(f"  {i}. [{folder_name}] {os.path.basename(pdf_file)}")

    if not pdf_files:
        print("❌ No PDF files found!")
        return False

    print(f"\n🔄 Starting automated testing on {len(pdf_files)} documents...")
    print("This may take several minutes per document...")

    # Clear database first
    print("🧹 Clearing existing database...")
    testing_system.pipeline.rag_system.clear_database()

    # Progress callback
    def progress_callback(current, total, result):
        progress = (current / total) * 100
        status = f"✅ {result['file_name']}" if result['success'] else f"❌ {result['file_name']}"
        print(f"[{progress:.1f}%] {status}")

    # Run the tests
    testing_system.run_automated_tests(test_folders, progress_callback)

    # Get results
    results = testing_system.test_results
    summary = testing_system.get_summary_stats()

    print(f"\n{'='*80}")
    print("📊 FINAL RESULTS SUMMARY")
    print(f"{'='*80}")

    print(f"Total Files: {summary['total_files']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Total RAG Chunks: {summary['total_rag_chunks']}")
    print(f"Total Rules Extracted: {summary['total_rules_extracted']}")
    print(f"Average Processing Time: {summary['avg_processing_time']:.2f}s")

    # Detailed results
    print(f"\n{'='*80}")
    print("📋 DETAILED RESULTS")
    print(f"{'='*80}")

    for i, result in enumerate(results, 1):
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"\n{i}. {result['file_name']} - {status}")

        if result['success']:
            print(f"   • RAG Chunks: {result['rag_chunks']}")
            print(f"   • Search Results: {result['search_results']}")
            print(f"   • Processing Time: {result['processing_time']:.2f}s")
            print(f"   • CSV Exported: {'✅' if result['csv_exported'] else '❌'}")

            # QA results summary
            qa_successful = sum(1 for qa in result['qa_results'] if 'error' not in qa)
            print(f"   • QA Tests: {qa_successful}/{len(result['qa_results'])} successful")

            # Show first QA result as example
            if result['qa_results']:
                first_qa = result['qa_results'][0]
                if 'error' not in first_qa:
                    answer_preview = first_qa['answer'][:100] + "..." if len(first_qa['answer']) > 100 else first_qa['answer']
                    print(f"   • Sample Answer: {answer_preview}")
        else:
            print(f"   • Error: {result.get('error', 'Unknown error')}")

    # Export results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"./test_results/full_system_test_{timestamp}.json"

    os.makedirs("./test_results", exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            'summary': summary,
            'detailed_results': results,
            'test_config': {
                'folders': test_folders,
                'total_files': len(pdf_files),
                'timestamp': datetime.now().isoformat()
            }
        }, f, indent=2)

    print(f"\n💾 Results exported to: {results_file}")

    # Check CSV files
    csv_files = [f for f in os.listdir("./test_results") if f.endswith("_rules.csv")]
    if csv_files:
        print(f"📊 Generated {len(csv_files)} CSV rule files:")
        for csv_file in csv_files:
            print(f"   • {csv_file}")
    else:
        print("⚠️ No CSV rule files were generated")

    # Final assessment
    success_rate = summary['success_rate']
    if success_rate >= 80:
        print("\n🎉 SYSTEM TEST PASSED!")
        print(f"   The RAG system successfully processed {success_rate:.1f}% of documents.")
        return True
    else:
        print(f"\n⚠️ SYSTEM TEST PARTIALLY SUCCESSFUL")
        print(f"   Only {success_rate:.1f}% of documents processed successfully.")
        print("   Check the detailed results above for issues.")
        return success_rate > 50  # Still consider it working if more than half succeed

if __name__ == "__main__":
    success = run_full_system_test()
    sys.exit(0 if success else 1)