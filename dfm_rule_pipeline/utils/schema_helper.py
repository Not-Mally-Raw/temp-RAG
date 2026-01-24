import re
from schema.feature_schema import features_dict

def generate_domain_summary() -> str:
    """
    Dynamically scans features_dict to create a DETAILED summary.
    Extracts Object Name AND its Attributes.
    
    Output Format:
    - Domain: Object1 (Attr, Attr...), Object2 (Attr...)
    """
    summary_lines = []
    
    for domain, schema_text in features_dict.items():
        # 1. Split schema into blocks starting with "Object:"
        # This handles the text format in features.py robustly
        object_blocks = schema_text.split("Object:")
        
        parsed_objects = []
        
        for block in object_blocks:
            if not block.strip(): 
                continue # Skip empty precursors
                
            lines = block.strip().splitlines()
            
            # Line 1 is always the Object Name (e.g., "PartBody")
            obj_name = lines[0].strip()
            
            # Find the "Attributes:" line
            attr_str = ""
            for line in lines:
                if "Attributes:" in line:
                    # Clean up "Attributes: Name, Type" -> "Name, Type"
                    attr_str = line.split("Attributes:")[1].strip()
                    break
            
            # Format: "PartBody (Material, Length, Width)"
            if attr_str:
                parsed_objects.append(f"{obj_name} ({attr_str})")
            else:
                parsed_objects.append(obj_name)
        
        # Join all objects for this domain
        # Example: **Model**: PartBody (Material, Length...), PartEdge (IsSharp)
        domain_line = f"**{domain}**: {', '.join(parsed_objects)}"
        summary_lines.append(domain_line)
        
    return "\n\n".join(summary_lines)