import os
import re
import hashlib
from pathlib import Path
import pdfplumber
import pandas as pd

SALT = "CDU_PRT691_SERVICE_HAUS_2026"


def hash_identifier(val: str, prefix: str) -> str:
    if not val or val.strip().upper() in ["UNKNOWN", "NONE", ""]:
        return f"{prefix}_UNKNOWN"
    clean = val.strip().upper()
    return f"{prefix}_{hashlib.sha256((clean + SALT).encode()).hexdigest()[:8]}"


def extract_odometer(text: str):
    patterns = [
        r"(?:SERVICED\s+AT|KMS?\s*[:]?|VEHICLE\s+KMS|AT)\s*[:]?\s*(\d{4,6})",
        r"(\d{4,6})\s*KM",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 1000 <= val <= 999999:
                return val
    return None


def extract_oil_grade(text: str):
    m = re.search(
        r"(0W-20|0W-30|0W-40|5W-30|5W-40|10W-30|15W-40|FULL\s+SYNTHETIC)",
        text,
        re.IGNORECASE,
    )
    return m.group(1).upper() if m else "None"


def parse_invoice_text(block: str) -> dict:
    if "Invoice number" not in block and "Tax invoice" not in block:
        return None

    # 1. Invoice Number
    inv_m = re.search(r"(?:Invoice number|Invoice no)[:\s]+(\d+)", block, re.IGNORECASE)
    invoice_id = inv_m.group(1) if inv_m else None
    if not invoice_id:
        return None

    # 2. Issue Date Extraction (robust to multiline table format)
    # Match standard DD/MM/YYYY dates
    dates = re.findall(r"(\b\d{2}/\d{2}/\d{4}\b)", block)
    issue_date = dates[0] if dates else None

    # 3. Registration / PO Extraction
    # Look for the text right between table headers or preceding the invoice number
    rego = "UNKNOWN"
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    for i, line in enumerate(lines):
        if "Purchase order no" in line:
            # Often the value appears 1 or 2 lines below or right after a pipe
            for offset in [1, 2]:
                if i + offset < len(lines):
                    candidate = lines[i + offset].replace("|", "").strip()
                    # Filter out metadata keywords
                    if candidate and not any(
                        k in candidate for k in ["Invoice", "Due", "Issue", "Tax"]
                    ):
                        rego = candidate.split()[0]  # Take the plate token
                        break
            break
        elif re.search(r"\|\s*([A-Z0-9]{5,8})\s*\|\s*\d{8}", line):
            # Formats like: | FTD51C | 00002559 |
            m = re.search(r"\|\s*([A-Z0-9]{5,8})\s*\|\s*\d{8}", line)
            rego = m.group(1)
            break

    # 4. Bill To / Customer
    client = "UNKNOWN"
    for i, line in enumerate(lines):
        if "Bill to" in line and i + 1 < len(lines):
            candidate = lines[i + 1].replace("|", "").strip()
            if candidate:
                client = candidate
            break

    # 5. Financial Total
    total_m = re.search(r"Total Amount.*?\$\s*([\d,]+\.\d{2})", block)
    total_aud = float(total_m.group(1).replace(",", "")) if total_m else 0.0

    # Categorization flags
    b_upper = block.upper()
    fleet_keywords = [
        "DETECTOR",
        "MINING",
        "PLUMBING",
        "SCAFFOLD",
        "SHUTTLE",
        "CORP",
        "HIRE",
        "CIVIL",
        "TRANSPORT",
        "PTY",
        "LTD",
        "LIMITED",
        "CONSTRUCTION",
        "PAINT",
    ]
    client_type = (
        "Fleet" if any(k in client.upper() for k in fleet_keywords) else "Retail"
    )

    warning_keywords = [
        "NEEDS",
        "ATTENTION",
        "KEEP AN EYE",
        "NEXT SERVICE",
        "LEAK FROM",
        "MAY NEED",
    ]
    has_warning = 1 if any(k in b_upper for k in warning_keywords) else 0

    return {
        "invoice_id": invoice_id,
        "issue_date": pd.to_datetime(issue_date, format="%d/%m/%Y", errors="coerce"),
        "raw_rego": rego,
        "vehicle_token": hash_identifier(rego, "VEH"),
        "client_token": hash_identifier(client, "CUST"),
        "client_type": client_type,
        "odometer_km": extract_odometer(block),
        "oil_grade": extract_oil_grade(block),
        "total_aud": total_aud,
        "has_warning": has_warning,
        "cat_routine": (
            1 if ("SERVICE" in b_upper or "OIL AND FILTER" in b_upper) else 0
        ),
        "cat_brakes": (
            1
            if any(w in b_upper for w in ["BRAKE", "PAD", "ROTOR", "CALIPER", "DISKS"])
            else 0
        ),
        "cat_cooling": (
            1
            if any(
                w in b_upper
                for w in [
                    "RADIATOR",
                    "WATERPUMP",
                    "WATER PUMP",
                    "COOLANT",
                    "THERMOSTAT",
                ]
            )
            else 0
        ),
        "cat_drivetrain": (
            1
            if any(
                w in b_upper
                for w in [
                    "TRANSMISSION",
                    "GEARBOX",
                    "CLUTCH",
                    "TCM",
                    "DIFFERENTIAL",
                    "DIFF",
                    "SHAFT",
                ]
            )
            else 0
        ),
        "cat_electrical": (
            1
            if any(
                w in b_upper
                for w in [
                    "BATTERY",
                    "ALTERNATOR",
                    "STARTER",
                    "COIL PACK",
                    "SPARK PLUG",
                    "GLOBE",
                ]
            )
            else 0
        ),
        "cat_compliance": (
            1
            if any(
                w in b_upper
                for w in [
                    "PINKSLIP",
                    "BLUESLIP",
                    "BROWN SLIP",
                    "REGO RENEWAL",
                    "DEFECT CLEARANCE",
                ]
            )
            else 0
        ),
    }


def batch_process(
    folder_path: str = "./invoices", output_csv: str = "workshop_service_data.csv"
):
    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {folder_path}")
        return

    records = []
    for pdf_file in pdf_files:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                full_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                blocks = re.split(r"(?=SH\s*SERVICE|Tax\s+invoice)", full_text)
                for block in blocks:
                    parsed = parse_invoice_text(block)
                    if parsed:
                        records.append(parsed)
        except Exception as e:
            print(f"Error reading {pdf_file.name}: {e}")

    df = pd.DataFrame(records).drop_duplicates(subset=["invoice_id"])

    # Drop records that couldn't find a valid date
    df = df.dropna(subset=["issue_date"]).sort_values("issue_date")

    df.to_csv(output_csv, index=False)
    print(f"Extraction successful: {len(df)} invoices saved with valid dates.")
    print(f"Unique vehicles found: {df['vehicle_token'].nunique()}")


if __name__ == "__main__":
    batch_process("./invoices")
