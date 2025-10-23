# Sample manufacturing documents for testing the RAG system

sample_manufacturing_process = """
Manufacturing Quality Control Standard Operating Procedure

1. SCOPE AND APPLICATION
This document establishes quality control requirements for automotive component manufacturing processes. All production lines must implement these standards to ensure consistent product quality and compliance with industry regulations.

2. QUALITY CONTROL CHECKPOINTS
- Incoming material inspection must verify dimensional tolerances within ±0.05mm
- Surface finish requirements: Ra ≤ 1.6μm for critical surfaces
- Heat treatment verification required for all steel components
- Final inspection includes dimensional check, visual inspection, and functional testing

3. PROCESS PARAMETERS
- Machining operations: spindle speed 2000-3000 RPM, feed rate 0.1-0.3 mm/rev
- Welding parameters: current 150-200A, voltage 18-22V, travel speed 5-8 mm/s
- Assembly torque specifications: M8 bolts 25±2 Nm, M10 bolts 40±3 Nm

4. DOCUMENTATION REQUIREMENTS
- Maintain process control charts for all critical dimensions
- Record lot traceability for all materials
- Document any deviations from standard parameters
- Weekly calibration of all measuring equipment

5. NON-CONFORMANCE HANDLING
- Immediate containment of suspect parts
- Root cause analysis within 24 hours
- Corrective action implementation within 72 hours
- Customer notification for safety-related issues within 2 hours
"""

sample_safety_guidelines = """
Workplace Safety Guidelines for Manufacturing Environment

PERSONAL PROTECTIVE EQUIPMENT (PPE)
- Safety glasses required in all production areas
- Steel-toed boots mandatory for floor operations
- Hearing protection required in areas exceeding 85 dB
- Cut-resistant gloves for material handling operations
- Respiratory protection for painting and coating operations

MACHINE SAFETY PROTOCOLS
- Lockout/Tagout procedures mandatory before maintenance
- Emergency stops must be tested weekly
- Guards must be in place before equipment operation
- Maximum lifting limit: 50 lbs without assistance
- Confined space entry requires permit and attendant

CHEMICAL HANDLING
- Material Safety Data Sheets (MSDS) readily available
- Proper storage segregation of incompatible chemicals
- Spill kits located within 50 feet of chemical storage
- Eye wash stations tested monthly
- Ventilation systems inspected quarterly

INCIDENT REPORTING
- All injuries reported within 1 hour
- Near-miss events documented and investigated
- Monthly safety meetings mandatory for all personnel
- Annual safety training required for equipment operators
"""

sample_design_specification = """
Product Design Specification: Automotive Brake Component

MATERIAL REQUIREMENTS
- Base material: Cast iron ASTM A48 Class 30
- Minimum tensile strength: 30,000 psi
- Hardness range: 170-220 HB
- Chemical composition: Carbon 3.0-3.4%, Silicon 1.8-2.3%

DIMENSIONAL SPECIFICATIONS
- Overall diameter: 280 ± 1.0 mm
- Thickness variation: ≤ 0.05 mm TIR
- Surface roughness: Ra ≤ 0.8 μm on friction surfaces
- Parallelism between faces: 0.02 mm maximum

MANUFACTURING PROCESSES
- Casting followed by rough machining
- Stress relief heat treatment at 550°C for 2 hours
- Finish machining to final dimensions
- Final inspection and quality verification

TESTING REQUIREMENTS
- Dimensional inspection on 100% of parts
- Hardness testing on sample basis (1 per 50 parts)
- Surface finish verification on critical surfaces
- Non-destructive testing for internal defects

PACKAGING AND SHIPPING
- Individual wrapping to prevent surface damage
- Batch documentation with each shipment
- Temperature-controlled storage recommended
- Maximum stacking height: 10 layers
"""

sample_maintenance_procedure = """
Preventive Maintenance Schedule - CNC Machining Center

DAILY MAINTENANCE (8 hours operation)
- Check coolant level and quality
- Verify spindle warm-up procedure completion
- Inspect tool wear and replace as needed
- Clean chip accumulation from work area
- Check hydraulic fluid levels

WEEKLY MAINTENANCE
- Lubricate all grease points per manufacturer specifications
- Check belt tension and alignment
- Inspect electrical connections for signs of wear
- Calibrate touch probe system
- Review and clear any alarm history

MONTHLY MAINTENANCE
- Replace air filters in electrical cabinet
- Check accuracy using certified test equipment
- Inspect and clean way covers
- Verify emergency stop function
- Update tool life database

QUARTERLY MAINTENANCE
- Perform complete geometric accuracy check
- Replace hydraulic filters
- Check spindle bearing condition
- Inspect and test safety interlocks
- Professional calibration of temperature sensors

ANNUAL MAINTENANCE
- Complete electrical system inspection
- Replace coolant system components
- Spindle rebuild evaluation
- Software backup and updates
- Comprehensive safety system audit

DOCUMENTATION
- Record all maintenance activities in logbook
- Maintain parts inventory for critical components
- Track equipment downtime and failure analysis
- Schedule maintenance during planned production breaks
"""

sample_procurement_spec = """
Supplier Quality Requirements - Fastener Components

SUPPLIER QUALIFICATION
- ISO 9001:2015 certification mandatory
- Automotive industry experience minimum 5 years
- Statistical process control implementation
- First article inspection capability
- Continuous improvement program demonstration

MATERIAL SPECIFICATIONS
- Grade 8.8 steel per ISO 898-1
- Zinc plating thickness: 8-15 microns
- Hydrogen embrittlement relief required
- Certificate of compliance with each lot
- Traceability to steel mill source

QUALITY REQUIREMENTS
- Process capability index (Cpk) ≥ 1.33
- Defect rate target: < 50 PPM
- First article inspection for new lots
- Statistical sampling per MIL-STD-105E
- 100% dimensional inspection on critical features

DELIVERY REQUIREMENTS
- Just-in-time delivery capability
- Electronic data interchange (EDI) capability
- Advance shipping notice (ASN) required
- Emergency response time: < 4 hours
- Packaging to prevent damage during transport

AUDIT AND MONITORING
- Annual supplier audits required
- Monthly quality performance reviews
- Corrective action response within 48 hours
- Continuous cost reduction initiatives
- Environmental compliance verification

TESTING PROTOCOL
- Tensile strength verification
- Torque-tension relationship testing
- Salt spray corrosion resistance
- Thread gauge verification
- Magnetic particle inspection for critical applications
"""

# Dictionary containing all sample documents
sample_documents = {
    "manufacturing_process": sample_manufacturing_process,
    "safety_guidelines": sample_safety_guidelines,
    "design_specification": sample_design_specification,
    "maintenance_procedure": sample_maintenance_procedure,
    "procurement_spec": sample_procurement_spec
}

# Additional metadata for each document
document_metadata = {
    "manufacturing_process": {
        "document_type": "procedure",
        "manufacturing_domain": "quality_control",
        "industry": "automotive",
        "compliance_level": "mandatory"
    },
    "safety_guidelines": {
        "document_type": "guideline",
        "manufacturing_domain": "safety",
        "industry": "general_manufacturing",
        "compliance_level": "regulatory"
    },
    "design_specification": {
        "document_type": "specification",
        "manufacturing_domain": "design",
        "industry": "automotive",
        "compliance_level": "mandatory"
    },
    "maintenance_procedure": {
        "document_type": "procedure",
        "manufacturing_domain": "maintenance",
        "industry": "general_manufacturing",
        "compliance_level": "recommended"
    },
    "procurement_spec": {
        "document_type": "specification",
        "manufacturing_domain": "procurement",
        "industry": "automotive",
        "compliance_level": "mandatory"
    }
}