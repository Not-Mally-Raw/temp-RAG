import csv
import json

def write_csv(output_path, rows):
    # FIX 1: Add encoding="utf-8" to prevent Windows crashes with symbols like ° or µ
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # FIX 2: Add 'status' and 'error' columns so you can debug failures
        headers = ["rule_text", "status", "rule_json", "equation", "reasoning", "error"]
        writer.writerow(headers)

        for r in rows:
            # FIX 3: Safe Data Handling
            # Use .get() to prevent KeyErrors if a field is missing (like in failed rows)
            
            # Handle rule_json: it might be a dict, a string, or None
            rule_data = r.get("rule_json", "")
            if isinstance(rule_data, (dict, list)):
                rule_data = json.dumps(rule_data)
            
            writer.writerow([
                r.get("rule_text", ""),
                r.get("status", "Unknown"),     # Writes "Success", "Failed", or "Skipped"
                rule_data,
                r.get("equation", ""),          # Empty string if no equation generated
                r.get("reasoning", ""),
                r.get("error", "")              # Captures error message if failed
            ])