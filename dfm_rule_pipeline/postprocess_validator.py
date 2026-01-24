import csv
import re
import os
from schema.feature_schema import features_dict

# CONFIGURATION
INPUT_CSV = r"output\dfm_final_results.csv"
OUTPUT_CSV = r"output\dfm_clean_production.csv"

def parse_schema_structure():
    """
    Parses features.py into a structured dictionary and flat sets for lookup.
    Returns:
        all_objects: Map of lower_case -> RealName (e.g. 'bend' -> 'Bend')
        all_attrs: Map of lower_case -> RealName (e.g. 'minradius' -> 'MinRadius')
    """
    all_objects = {} 
    all_attrs = {}   

    for domain, text in features_dict.items():
        current_obj = None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("Object:"):
                current_obj = line.split("Object:")[1].strip()
                all_objects[current_obj.lower()] = current_obj
                
            elif line.startswith("Attributes:") and current_obj:
                attrs = line.split("Attributes:")[1].split(",")
                for attr in attrs:
                    attr = attr.strip()
                    all_attrs[attr.lower()] = attr
    
    return all_objects, all_attrs

def build_synonym_map(all_objects, all_attrs):
    """
    Builds a targeted map for common DFM synonyms that don't match 1:1.
    This is the ONLY place for hardcoded engineering knowledge.
    """
    synonyms = {
        # Global Concepts
        "thickness": "ModuleParams.Thickness",
        "material_thickness": "ModuleParams.Thickness",
        "sheet_thickness": "ModuleParams.Thickness",
        ".thickness": "ModuleParams.Thickness",
        "mm": "",
        "in": "",
        
        # Injection Moulding / Die Cast
        "rib1": "Rib",
        "rib2": "Rib",
        # If Rib.Radius isn't in schema, map to closest physical match
        "rib.radius": "Rib.RadiusAtBot", 
        "rib.width": "Rib.ThicknessAtBot", 
        
        # Sheet Metal
        "lance": "OpenLance", # Assumption based on standard DFM
        "curl.radius": "Curl.Radius", # Fix lowercase
        "bend.radius": "Bend.MinRadius",
        
        # Turning
        "groove.width": "Groove.GrooveWidth", # Or FloorWidth based on schema
    }
    return synonyms

def algorithmic_clean(variable, all_objects, all_attrs, synonyms):
    """
    Reconstructs a valid 'Object.Attribute' from a dirty string 
    by finding the best valid object and attribute buried inside it.
    """
    # 1. Check Synonyms first (Global override)
    if variable.lower() in synonyms:
        return synonyms[variable.lower()]
    
    # 2. Handle common math/numbers (don't touch them)
    if re.match(r'^[\d\.]+$', variable):
        return variable
        
    # 3. Tokenize: Split by dots or underscores
    # "Bend.HoleChain.MinRadius" -> ['Bend', 'Hole', 'Chain', 'Min', 'Radius']
    tokens = re.split(r'[\._]', variable)
    
    found_object = None
    found_attr = None
    
    # 4. Scan tokens for known Objects
    for token in tokens:
        if token.lower() in all_objects:
            found_object = all_objects[token.lower()]
            break 
            
    # 5. Scan tokens for known Attributes
    for token in tokens:
        if token.lower() in all_attrs:
            found_attr = all_attrs[token.lower()]
            # Keep searching in case there's a more specific attribute later
            # (But usually the first match is decent for cleaning)
            
    # 6. Reconstruct Valid Path
    if found_object and found_attr:
        return f"{found_object}.{found_attr}"
    
    # 7. Fallback: If we found an attribute but no object, 
    # check if the variable ends with ".Attribute" (common LLM pattern)
    if not found_object and found_attr:
        # e.g. "Width" -> maybe valid if context implies it, but risky.
        # Check specific synonym map again for attribute-only keys
        if found_attr.lower() in synonyms:
            return synonyms[found_attr.lower()]

    return variable

def process_equation(eq, all_objects, all_attrs, synonyms):
    if not eq: return eq
    
    # Apply global synonyms to the whole string first (e.g. "Material.Thickness")
    for key, val in synonyms.items():
        # Case-insensitive replacement
        pattern = re.compile(re.escape(key), re.IGNORECASE)
        eq = pattern.sub(val, eq)

    # Regex to find variables: Words with dots/underscores, or just words
    # We replace them using the algorithmic cleaner
    def replacer(match):
        var = match.group(0)
        # Skip pure numbers
        if re.match(r'^\d+(\.\d+)?$', var): return var
        return algorithmic_clean(var, all_objects, all_attrs, synonyms)

    # Matches "Object.Attribute", "Object_Attribute", "Word.Word.Word"
    return re.sub(r'\b[a-zA-Z][a-zA-Z0-9_\.]+\b', replacer, eq)

def validate_row(row, all_objects, all_attrs):
    """
    Final Check: Does the equation consist ONLY of known Schema terms?
    """
    eq = row.get("equation", "")
    if not eq: return True, ""
    
    # Extract "Object.Attribute" patterns
    used_vars = re.findall(r'\b[A-Za-z0-9]+\.[A-Za-z0-9]+\b', eq)
    errors = []
    
    for var in used_vars:
        obj, attr = var.split('.', 1)
        
        # Check if Object is valid
        if obj.lower() not in all_objects:
            errors.append(f"Unknown Object: {obj}")
            continue
            
        # Check if Attribute is valid (Loose check: is it in the global attribute list?)
        # Strict check would be: is attr in features_dict[domain][obj]
        if attr.lower() not in all_attrs:
            errors.append(f"Unknown Attribute: {attr}")

    return (False, "; ".join(errors)) if errors else (True, "")

def main():
    print("🧠 Starting SCALABLE DFM Validator...")
    
    # 1. Learn the Schema
    all_objects, all_attrs = parse_schema_structure()
    synonyms = build_synonym_map(all_objects, all_attrs)
    print(f"    📚 Learned {len(all_objects)} Objects and {len(all_attrs)} Attributes.")
    
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Input file not found: {INPUT_CSV}")
        return

    cleaned_rows = []
    
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get("status", "")
            
            # Process success & review needed rows
            if "Failed" not in status and "Skipped" not in status:
                original_eq = row.get("equation", "")
                
                # 1. Clean / Reconstruct
                clean_eq = process_equation(original_eq, all_objects, all_attrs, synonyms)
                row["equation"] = clean_eq
                
                # 2. Validate
                is_valid, error_msg = validate_row(row, all_objects, all_attrs)
                
                if original_eq != clean_eq:
                    row["status"] = "Success (Standardized)"
                
                if not is_valid:
                    row["status"] = "Review Needed (Schema Gap)"
                    row["error"] = error_msg
                else:
                    # Clear previous warnings if now valid
                    if "Review" in status:
                        row["status"] = "Success (Standardized)"
                        row["error"] = ""
            
            cleaned_rows.append(row)

    headers = ["rule_text", "status", "rule_json", "equation", "reasoning", "error"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(cleaned_rows)
        
    print(f"✨ Cleanup Complete! Check {OUTPUT_CSV}")

if __name__ == "__main__":
    main()