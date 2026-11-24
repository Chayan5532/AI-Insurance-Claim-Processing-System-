"""
One-off helper script to generate small dummy PDF documents for testing the
Streamlit app. Not part of the main application -- run manually if you want
fresh sample files:

    python generate_samples.py
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

OUT_DIR = "sample_documents"
os.makedirs(OUT_DIR, exist_ok=True)
styles = getSampleStyleSheet()


def make_pdf(filename, title, body_lines):
    doc = SimpleDocTemplate(os.path.join(OUT_DIR, filename), pagesize=letter)
    story = [Paragraph(title, styles["Title"]), Spacer(1, 16)]
    for line in body_lines:
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 8))
    doc.build(story)
    print(f"Created {filename}")


# 1. Sample genuine policy document
make_pdf(
    "sample_policy_genuine.pdf",
    "Motor Insurance Policy Certificate",
    [
        "Policy Number: POL-2026-88213",
        "Policyholder Name: Rahul Sharma",
        "Vehicle Registration Number: DL8CAF1234",
        "Vehicle Make/Model: Hyundai Creta 2023",
        "Policy Type: Comprehensive Cover",
        "Policy Validity: 01-Jan-2026 to 31-Dec-2026",
        "Own Damage Sum Insured: Rs. 8,50,000",
        "Coverage Includes: Accidental damage, fire, theft, third-party liability",
        "Exclusions: Driving without a valid licence, drunk driving, "
        "normal wear and tear, mechanical breakdown",
        "Deductible (Compulsory Excess): Rs. 2,000",
        "Issuing Office: SecureLife General Insurance, New Delhi",
    ],
)

# 2. Sample repair estimate
make_pdf(
    "sample_repair_estimate.pdf",
    "Vehicle Repair Estimate",
    [
        "Workshop: City Motors Service Center, New Delhi",
        "Vehicle Registration Number: DL8CAF1234",
        "Date of Estimate: 14-Aug-2026",
        "Damage Description: Rear bumper damage, tail lamp assembly broken, "
        "minor dent on boot door",
        "Estimated Parts Cost: Rs. 18,500",
        "Estimated Labour Cost: Rs. 4,500",
        "Total Estimated Repair Cost: Rs. 23,000",
        "Estimated Repair Duration: 3-4 working days",
    ],
)

# 3. Sample invoice
make_pdf(
    "sample_invoice.pdf",
    "Repair Invoice",
    [
        "Invoice Number: INV-99231",
        "Workshop: City Motors Service Center, New Delhi",
        "Vehicle Registration Number: DL8CAF1234",
        "Date: 17-Aug-2026",
        "Parts Replaced: Rear bumper, tail lamp assembly",
        "Labour Charges: Rs. 4,500",
        "Parts Charges: Rs. 18,500",
        "Total Amount Paid: Rs. 23,000",
        "Payment Mode: Card",
    ],
)

# 4. Sample RC (Registration Certificate)
make_pdf(
    "sample_rc.pdf",
    "Vehicle Registration Certificate (RC)",
    [
        "Registration Number: DL8CAF1234",
        "Owner Name: Rahul Sharma",
        "Vehicle Class: Motor Car (LMV)",
        "Maker/Model: Hyundai Creta",
        "Registration Date: 15-Mar-2023",
        "Registered RTO: Dwarka, New Delhi",
        "Chassis Number: MALC381CLJM123456",
        "Engine Number: G4FGHK123456",
    ],
)

# 5. Suspicious/fraud-risk-flavoured policy (mismatched vehicle number, expired)
make_pdf(
    "sample_policy_suspicious.pdf",
    "Motor Insurance Policy Certificate",
    [
        "Policy Number: POL-2024-11209",
        "Policyholder Name: Vikram Mehta",
        "Vehicle Registration Number: HR26XY5678",
        "Vehicle Make/Model: Maruti Suzuki Swift 2021",
        "Policy Type: Third-Party Only",
        "Policy Validity: 01-Jan-2024 to 31-Dec-2024",
        "Own Damage Sum Insured: Not Applicable (Third-Party Only)",
        "Coverage Includes: Third-party liability only",
        "Exclusions: Own vehicle damage is NOT covered under this policy",
        "Issuing Office: SecureLife General Insurance, Gurugram",
    ],
)

print("\nAll sample documents created in the 'sample_documents' folder.")
