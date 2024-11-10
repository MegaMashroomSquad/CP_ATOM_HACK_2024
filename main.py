from src.core import tag_document, preprocces_document
from src.model import Model
from src.timer import timer
import numpy as np


def main():
    model = Model()
    # tag_document(model, "rules.docx", "impl.docx", None)
    # hmi_prefix = "train Атом/train data/HMI/UC-"
    # ssts_prefix = "train Атом/train data/SSTS/SSTS-"

    # doc_ids = [
    #     6583,
    #     8604,
    #     8692,
    #     8800,
    #     11467,
    #     25957,
    #     26160,
    #     26161,
    #     26771,
    #     28561,
    #     30371,
    #     31523
    # ]

    hmi_prefix = "case_data/train Атом/train data/HMI/UC-"
    ssts_prefix = "case_data/train Атом/train data/SSTS/SSTS-"

    doc_id = 8800

    hmi_file = f"{hmi_prefix}{doc_id}.docx"
    ssts_file = f"{ssts_prefix}{doc_id}.docx"

    print(preprocces_document(model, hmi_file, ssts_file))
    timer.get_stats()


if __name__ == "__main__":
    main()
