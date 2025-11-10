"""
Ablation Study: Compare agent performance
Simple evaluation script for UCLA submission
"""

import sqlite3
import json
from typing import Dict, List

def analyze_rlhf_data(db_path: str = "rlhf_annotations.db") -> Dict:
    """Analyze RLHF annotations for ablation study."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            speech_act_type,
            AVG(rating) as avg_rating,
            AVG(pragmatic_coherence) as avg_coherence,
            COUNT(*) as count
        FROM rlhf_annotations
        GROUP BY speech_act_type
    """)
    
    results = cursor.fetchall()
    conn.close()
    
    analysis = {
        "by_speech_act": [
            {
                "type": row[0],
                "avg_rating": row[1],
                "avg_coherence": row[2],
                "count": row[3]
            }
            for row in results
        ]
    }
    
    return analysis

def generate_report():
    """Generate simple text report."""
    analysis = analyze_rlhf_data()
    
    report = "# Ablation Study Report\n\n"
    report += "## Performance by Speech Act Type\n\n"
    
    for item in analysis["by_speech_act"]:
        report += f"**{item['type']}**:\n"
        report += f"- Average Rating: {item['avg_rating']:.2f}\n"
        report += f"- Average Coherence: {item['avg_coherence']:.2f}\n"
        report += f"- Sample Count: {item['count']}\n\n"
    
    return report

if __name__ == "__main__":
    report = generate_report()
    print(report)
    
    # Save to file
    with open("ablation_report.md", "w") as f:
        f.write(report)

