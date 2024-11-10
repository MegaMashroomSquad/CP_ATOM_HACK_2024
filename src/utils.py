from docx2python import docx2python
import re
import string
from fuzzywuzzy import fuzz


def parse_docx_text_SSTS(file_path):
    docx_content = docx2python(file_path)

    text_content = docx_content.text

    sections = {}
    section_index = None
    current_section = []
    pre_section_content = []

    for line in text_content.split("\n"):
        line = line.strip()
        if not line:
            continue

        pattern = r"\((?:[^()]*[&|][^()]*)+\)"
        pattern2 = r"\((?:[a-zA-Z](?:[|&][a-zA-Z])*)\)"
        match = re.search(pattern2, line)
        if match:
            if current_section and section_index is not None:
                sections[section_index] = " ".join(current_section)
            elif not section_index:
                sections["pre_section"] = " ".join(pre_section_content)
            section_index = line
            current_section = [line]
        else:
            if section_index is None:
                pre_section_content.append(line)
            else:
                current_section.append(line)

    if current_section and section_index is not None:
        sections[f"{section_index}."] = " ".join(current_section)

    return sections


def structure_text_hmi(text):
    structured_text = {"preconditions": [], "main_scenario": [], "postconditions": []}

    current_section = None

    import string

    scenario_counter = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if fuzz.partial_ratio("Preconditions", line) > 95:
            current_section = "preconditions"
        elif fuzz.partial_ratio("Main Scenario", line) > 95:
            current_section = "main_scenario"
        elif fuzz.partial_ratio("Postconditions", line) > 95:
            current_section = "postconditions"
        elif fuzz.partial_ratio("Alternative Scenario", line) > 95:
            scenario_counter += 1
            current_section = (
                f"alternative_scenario_{string.ascii_uppercase[scenario_counter - 1]}"
            )
            structured_text[current_section] = (
                []
            )  # Start a new list for each alternative scenario
        elif current_section:
            structured_text[current_section].append(line.strip())

    return structured_text


def structure_text_to_dict_hmi(text_content):
    structured_text = structure_text_hmi(text_content)
    section_dict = {}

    for section, lines in structured_text.items():
        if section == "alternative_scenarios":
            section_text = []
            for scenario in lines:
                scenario_text = "\n".join(scenario)
                section_text.append(scenario_text)
            section_dict[section] = section_text
        else:
            section_text = "\n".join(lines)
            section_dict[section] = section_text

    return section_dict
