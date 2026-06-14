from transformers import MarianMTModel, MarianTokenizer


class MarianTranslator:
    def __init__(self):
        model_name = "Helsinki-NLP/opus-mt-de-en"

        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs)

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )


translator = MarianTranslator()
