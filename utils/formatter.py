# utils/formatter.py
import json


def extract_fields(tokens):
    structured = {
        "province": "",
        "city": "",
        "district": "",
        "town": "",
        "extra": ""
    }
    for token, tag in tokens:
        if tag == "PROV":
            structured["province"] = token
        elif tag == "CITY":
            structured["city"] = token
        elif tag == "DISTRICT":
            structured["district"] = token
        elif tag == "TOWN":
            structured["town"] = token
        elif tag == "EXTRA":
            structured["extra"] = token
    return structured

def format_address(structured):
    return str(structured)