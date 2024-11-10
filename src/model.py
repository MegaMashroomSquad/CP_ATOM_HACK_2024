from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from src.scheme import Fact


class Model:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Model, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.device = "cuda"
        self.model_name = "Qwen/Qwen2.5-14B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="cuda",
            cache_dir="./llm_weights",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate(self, prompt, system_prompt=None, max_new_tokens=1):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant."
                    if system_prompt is None
                    else system_prompt
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.00000001,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        return response

    def translate_logical_expression(self, expression):
        prompt = f"""EXAMPLES: 
            TASK: Convert (a|b) into natural english. RESPONSE: a or b should be true.
            TASK: Convert (a&b) into natural english. RESPONSE: a and b both should be true.
            TASK: Convert ((a&c)|(b&d)) into natural english. RESPONSE: both a and c should be true OR both b and d should be true.
            
            TASK: Convert the logical expression {expression} into natural English. Provide only the translation in a single clear sentence. RESPONSE:"""
        return self.generate(prompt, max_new_tokens=64)

    def build_prompt(self, factA: Fact, factB: Fact):
        system_prompt = "You are a compliance analyst tasked with verifying whether an implementation complies with a given regulation."
        user_prompt = f"""
            You will be provided with two statements:
            
            - Statement A (Regulation): Contains information from a regulation, policy, or specification outlining conditions and requirements.

            - Statement B (Implementation): Contains a description of an implementation that may or may not comply with the regulation specified in Statement A.

            Determine whether there are any discrepancies or inconsistencies between the regulation and the implementation.

            Statement A text:
            {factA.text}

            Statement B text:
            {factB.text}

            Write only one word "Yes" or "No"
            Disrepancy Found: 
        """
        return system_prompt, user_prompt

    def build_prompt_description(self, factA: Fact, factB: Fact):
        system_prompt = "You are a compliance analyst tasked with verifying whether an implementation complies with a given regulation."
        user_prompt = f"""
            You will be provided with two statements:

            - Statement A (Regulation): Contains information from a regulation, policy, or specification outlining conditions and requirements.

            - Statement B (Implementation): Contains a description of an implementation that may or may not comply with the regulation specified in Statement A.


            Determine whether there are any discrepancies or inconsistencies between the regulation and the implementation.

            Also try to make your answer concise, write only the main diff and do not any other information

            Statement A text:
            {factA.text}

            Statement B text:
            {factB.text}

            Your concise discrepancy: 
        """
        return system_prompt, user_prompt

    def build_prompt_severeness(self, discrepancy: str):
        system_prompt = "You are a compliance analyst tasked with verifying whether an implementation complies with a given regulation."
        user_prompt = f"""
            You will be provided with a description of discrepancies between an official regulation and an existing implementation.

            Your task is to evaluate how important this discrepancy is, based on the following scale:

            - 0: Not a serious difference at all
            - 1: Tolerable difference
            - 2: Important difference

            Please analyze the description and output only the number 0, 1, or 2 corresponding to the importance level of the discrepancy. Do not provide any explanation or additional information—just output the number.
            Discrepancy: {discrepancy}
            Your number: 
        """

        return system_prompt, user_prompt
