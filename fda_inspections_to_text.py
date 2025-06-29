import json
from datetime import datetime

# Sample JSON data (replace with your actual JSON list or load from file)
inspection_data = [
    {
        "legal_name": "3m Company",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "Cordova, Illinois, United States",
        "fei_number": "1484185",
        "classification": "VAI",
        "posted_citations": "Yes",
        "fiscal_year": 2011,
        "inspection_end_date": "2011-04-13T00:00:00.000Z",
        "inspection_id": "720685",
        "additional_details": "-"
    },
    {
        "legal_name": "3m Company - Health Care Business",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "3m Center 2510 Conway Ave, Bldg 275-5w-06 Saint Paul, Minnesota 55144 United States",
        "fei_number": "2110898",
        "classification": "VAI",
        "posted_citations": "Yes",
        "fiscal_year": 2019,
        "inspection_end_date": "2019-02-11T00:00:00.000Z",
        "inspection_id": "1081355",
        "additional_details": "Postmarket Adverse Drug Experience (PADE)"
    },
    {
        "legal_name": "3m Espe Dental Products",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "2510 Conway Ave E Bldg 260-02a-11 Saint Paul, Minnesota 55144 United States",
        "fei_number": "3005174370",
        "classification": "NAI",
        "posted_citations": "No",
        "fiscal_year": 2014,
        "inspection_end_date": "2014-03-12T00:00:00.000Z",
        "inspection_id": "872250",
        "additional_details": "-"
    },
    {
        "legal_name": "3m Company IRB",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "3M Center Bldg 220-6E-03 Saint Paul, Minnesota 55144 United States",
        "fei_number": "3009666037",
        "classification": "VAI",
        "posted_citations": "No",
        "fiscal_year": 2013,
        "inspection_end_date": "2012-10-10T00:00:00.000Z",
        "inspection_id": "803442",
        "additional_details": "IRB"
    },
    {
        "legal_name": "3m Company",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "Brookings, South Dakota, United States",
        "fei_number": "1717046",
        "classification": "VAI",
        "posted_citations": "Yes",
        "fiscal_year": 2018,
        "inspection_end_date": "2018-05-25T00:00:00.000Z",
        "inspection_id": "1060963",
        "additional_details": "-"
    },
    {
        "legal_name": "3m Company - Health Care Business",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "3m Center 2510 Conway Ave, Bldg 275-5w-06 Saint Paul, Minnesota 55144 United States",
        "fei_number": "2110898",
        "classification": "VAI",
        "posted_citations": "Yes",
        "fiscal_year": 2017,
        "inspection_end_date": "2017-05-17T00:00:00.000Z",
        "inspection_id": "1017823",
        "additional_details": "Postmarket Adverse Drug Experience (PADE)"
    },
    {
        "legal_name": "3m Company",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "2510 Conway Ave E Saint Paul, Minnesota 55144 United States",
        "fei_number": "2126770",
        "classification": "NAI",
        "posted_citations": "No",
        "fiscal_year": 2013,
        "inspection_end_date": "2013-04-18T00:00:00.000Z",
        "inspection_id": "828412",
        "additional_details": "-"
    },
    {
        "legal_name": "3m Company",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "Cordova, Illinois, United States",
        "fei_number": "1484185",
        "classification": "NAI",
        "posted_citations": "No",
        "fiscal_year": 2022,
        "inspection_end_date": "2022-09-22T00:00:00.000Z",
        "inspection_id": "1182209",
        "additional_details": "-"
    },
    {
        "legal_name": "3m Company",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "Cordova, Illinois, United States",
        "fei_number": "1484185",
        "classification": "NAI",
        "posted_citations": "No",
        "fiscal_year": 2012,
        "inspection_end_date": "2012-06-05T00:00:00.000Z",
        "inspection_id": "787941",
        "additional_details": "-"
    },
    {
        "legal_name": "3m Company",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "2510 Conway Ave E Saint Paul, Minnesota 55144 United States",
        "fei_number": "2126770",
        "classification": "NAI",
        "posted_citations": "No",
        "fiscal_year": 2019,
        "inspection_end_date": "2018-10-19T00:00:00.000Z",
        "inspection_id": "1075144",
        "additional_details": "Bioanalytical BA/BE"
    },
    {
        "legal_name": "3m Company",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "Brookings, South Dakota, United States",
        "fei_number": "1717046",
        "classification": "NAI",
        "posted_citations": "No",
        "fiscal_year": 2013,
        "inspection_end_date": "2012-12-05T00:00:00.000Z",
        "inspection_id": "813433",
        "additional_details": "-"
    },
    {
        "legal_name": "3m Company",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "Cordova, Illinois, United States",
        "fei_number": "1484185",
        "classification": "NAI",
        "posted_citations": "No",
        "fiscal_year": 2016,
        "inspection_end_date": "2016-08-19T00:00:00.000Z",
        "inspection_id": "984843",
        "additional_details": "-"
    },
    {
        "legal_name": "3m Company",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "2510 Conway Ave E Saint Paul, Minnesota 55144 United States",
        "fei_number": "2126770",
        "classification": "NAI",
        "posted_citations": "No",
        "fiscal_year": 2012,
        "inspection_end_date": "2012-04-20T00:00:00.000Z",
        "inspection_id": "778370",
        "additional_details": "-"
    },
    {
        "legal_name": "3m Company",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "Cordova, Illinois, United States",
        "fei_number": "1484185",
        "classification": "NAI",
        "posted_citations": "No",
        "fiscal_year": 2010,
        "inspection_end_date": "2010-03-05T00:00:00.000Z",
        "inspection_id": "649164",
        "additional_details": "-"
    },
    {
        "legal_name": "3m Company",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "2510 Conway Ave E Saint Paul, Minnesota 55144 United States",
        "fei_number": "2126770",
        "classification": "VAI",
        "posted_citations": "Yes",
        "fiscal_year": 2015,
        "inspection_end_date": "2014-12-15T00:00:00.000Z",
        "inspection_id": "907895",
        "additional_details": "-"
    },
    {
        "legal_name": "3m Company",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "2510 Conway Ave E Saint Paul, Minnesota 55144 United States",
        "fei_number": "2126770",
        "classification": "VAI",
        "posted_citations": "No",
        "fiscal_year": 2010,
        "inspection_end_date": "2009-10-30T00:00:00.000Z",
        "inspection_id": "634005",
        "additional_details": "-"
    },
    {
        "legal_name": "3m Espe Dental Products",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "2510 Conway Ave E Bldg 260-02a-11 Saint Paul, Minnesota 55144 United States",
        "fei_number": "3005174370",
        "classification": "VAI",
        "posted_citations": "No",
        "fiscal_year": 2011,
        "inspection_end_date": "2010-11-19T00:00:00.000Z",
        "inspection_id": "698056",
        "additional_details": "-"
    },
    {
        "legal_name": "3m Espe Dental Products",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "2510 Conway Ave E Bldg 260-02a-11 Saint Paul, Minnesota 55144 United States",
        "fei_number": "3005174370",
        "classification": "VAI",
        "posted_citations": "Yes",
        "fiscal_year": 2009,
        "inspection_end_date": "2009-03-13T00:00:00.000Z",
        "inspection_id": "579363",
        "additional_details": "-"
    },
    {
        "legal_name": "3m Company",
        "enterprise_name": "3M",
        "country": "United States",
        "firm_address": "2510 Conway Ave E Saint Paul, Minnesota 55144 United States",
        "fei_number": "2126770",
        "classification": "VAI",
        "posted_citations": "Yes",
        "fiscal_year": 2020,
        "inspection_end_date": "2020-03-20T00:00:00.000Z",
        "inspection_id": "1123567",
        "additional_details": "-"
    }
]

# Output file path
output_file = "dataset/txt/fda_inspections.txt"

# Function to format each record into readable text
def format_inspection(record):
    # Convert date to readable format
    try:
        end_date = datetime.fromisoformat(record["inspection_end_date"].replace("Z", "")).strftime("%B %d, %Y")
    except:
        end_date = record["inspection_end_date"]

    # Posted citations message
    citations_msg = "Citations were posted as a result of the inspection." if record.get("posted_citations", "").lower() == "yes" else "No citations were posted."

    # Additional details
    additional = record.get("additional_details", "").strip()
    additional_msg = additional if additional and additional != "-" else "No additional details were provided."

    # Format paragraph
    text = (
        f'In fiscal year {record["fiscal_year"]}, an FDA inspection (ID: {record["inspection_id"]}) was conducted '
        f'at the firm "{record["legal_name"]}", also known as "{record["enterprise_name"]}", located in {record["firm_address"]}. '
        f'The inspection ended on {end_date}. The facility, identified by FEI number {record["fei_number"]}, received a classification of '
        f'"{record["classification"]}". {citations_msg} {additional_msg}.\n'
    )
    return text

# Write formatted text to file
with open(output_file, "w", encoding="utf-8") as f:
    for record in inspection_data:
        formatted = format_inspection(record)
        f.write(formatted + "\n")

print(f"Text file '{output_file}' created successfully.")
