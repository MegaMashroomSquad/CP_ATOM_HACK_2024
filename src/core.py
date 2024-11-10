from dataclasses import dataclass
from typing import List, Tuple, Union
import re
import numpy as np
import torch
from docx2python import docx2python
from sklearn.neighbors import KNeighborsClassifier
from transformers import pipeline

from src.model import Model
from src.scheme import Fact, FactComparison
from src.utils import parse_docx_text_SSTS, structure_text_to_dict_hmi
from src.timer import timer


class DocumentProcessor:
    @staticmethod
    def replace_boolean_algebra_with_translation(model: Model, text: str) -> str:
        pattern = r"\((?:[^()]*[&|][^()]*)+\)"
        pattern2 = r"\((?:[a-zA-Z](?:[|&][a-zA-Z])*)\)"

        lines = text.split("\n")

        for i, line in enumerate(lines):

            matches = re.findall(pattern2, line)
            for match in matches:
                translation = model.translate_logical_expression(match)
                translation = "(" + translation + ")"
                line = line.replace(match, translation)
            lines[i] = line

        return "\n".join(lines)

    @staticmethod
    @timer
    def process_docx_file(
        model: Model, file_path, file_type: str = "SSTS"
    ) -> List[Fact]:
        docx_content = docx2python(file_path)
        text_content = docx_content.text
        first_line = text_content.split("\n", 1)[0]

        if file_type == "SSTS":
            sections = parse_docx_text_SSTS(file_path)
        elif file_type == "HMI":
            sections = structure_text_to_dict_hmi(text_content)
        else:
            raise ValueError("File type must be either 'SSTS' or 'HMI'")

        facts = []
        for section, content in sections.items():
            translated_content = (
                DocumentProcessor.replace_boolean_algebra_with_translation(
                    model, content
                )
            )
            fact = Fact(
                text=translated_content, doc_name=first_line, section_name=section
            )
            facts.append(fact)
        return facts


class InconsistencyClassifier:
    @staticmethod
    @timer
    def find_disrepancies(model: Model, rules_data: List[Fact], impl_data: List[Fact]):
        def get_facts(doc):
            content = docx2python(doc).text
            splitting_prompt = f"""You will see a document with list of rules and instructions. I need you to split into many little logical fragments. 
                Basically, I want you to give me a list of facts and list of keywords from a whole document. Output should be like:
                1) <fact 1>
                2) <fact 2>
                3) <fact 3>
                etc...
                Write numbers and facts and only them
                My document to split: {content}
            """
            res = [
                Fact(text=x, doc_name="", section_name="")
                for x in model.generate(splitting_prompt, max_new_tokens=1000).split(
                    "\n"
                )
            ]
            return res

        # print("RULES LIST")

        # for rule in rules_data:
        #     print(rule)

        # print("IMPL LIST")

        # for impl in impl_data:
        #     print(impl)

        res = []
        for rule in rules_data:
            for impl in impl_data:
                system_prompt, user_prompt = model.build_prompt(rule, impl)
                ans = model.generate(user_prompt, system_prompt, max_new_tokens=1)
                if ans.lower() == "yes":
                    system_prompt_descr, user_prompt_descr = (
                        model.build_prompt_description(rule, impl)
                    )
                    descr = (
                        model.generate(
                            user_prompt_descr, system_prompt_descr, max_new_tokens=100
                        )
                        .strip()
                        .strip("\n")
                    )
                    system_prompt_severity, user_prompt_severity = (
                        model.build_prompt_severeness(descr)
                    )
                    severe = model.generate(
                        user_prompt_severity, system_prompt_severity, max_new_tokens=1
                    )

                    res.append(
                        {
                            "factA": rule,
                            "factB": impl,
                            "has_diff": True,
                            "diff": descr,
                            "severeness_level": severe,
                        }
                    )

        return res


class HallucinationDetector:
    pass


class ComplianceClassifier:
    @staticmethod
    @timer
    def predict_compliance(
        # model: Model,
        data: List[FactComparison],
        method: str = "statistical",
        clf: KNeighborsClassifier = None,
    ) -> List[int]:
        if method == "statistical":
            # Use severeness levels directly as features
            features = []
            for fc in data:
                # Create histogram of severeness levels (0-5)
                severity_hist = np.zeros(6)  # 0-5 severity levels
                severity_hist[fc.severeness_level] += 1
                features.append(severity_hist)

            X = np.array(features)
            results = clf.predict(X).tolist()

            # Example training data with histograms
            # Each row represents counts of severity levels [0,1,2,3,4,5]
            # example_X = np.array(
            #     [
            #         [2, 1, 1, 0, 0, 0],  # 2 level-0s, 1 level-1, 1 level-2 -> level 4
            #         [0, 1, 1, 1, 2, 1],  # more severe issues -> level 2
            #         [3, 1, 0, 0, 0, 0],  # mostly minor issues -> level 5
            #         [0, 0, 1, 2, 2, 3],  # many severe issues -> level 1
            #         [1, 2, 1, 1, 0, 0],  # mixed moderate issues -> level 3
            #     ]
            # )
            # example_y = np.array([4, 2, 5, 1, 3])

            # clf = KNeighborsClassifier(n_neighbors=3, weights="distance")
            # clf.fit(example_X, example_y)

        else:
            raise ValueError("Method must be either 'llm' or 'statistical'")
        return results


def preprocces_document(
    model: Model, rules_document_path: str, impl_document_path: str
) -> Tuple[List[Fact], List[Fact]]:
    rules_data = DocumentProcessor.process_docx_file(
        model, rules_document_path, file_type="HMI"
    )
    impl_data = DocumentProcessor.process_docx_file(
        model, impl_document_path, file_type="SSTS"
    )
    fact_comparisons = InconsistencyClassifier.find_disrepancies(
        model, rules_data, impl_data
    )
    return fact_comparisons


def tag_document(
    model: Model,
    rules_document_path: str,
    impl_document_path: str,
    compliance_knn_model: KNeighborsClassifier,
) -> Union[int, None]:
    """
    Compliance level classification
    """

    fact_comparisons = preprocces_document(
        model, rules_document_path, impl_document_path
    )
    compliance_levels = ComplianceClassifier.predict_compliance(
        fact_comparisons, compliance_knn_model
    )
    return compliance_levels
