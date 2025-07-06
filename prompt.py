def get_eval_prompt_fan():
    eval_prompt = """You are an expert professor specialized in grading whether the prediction to the question is correct or not according to the real answer.
        Please note that if the prediction and the answer are not completely identical, but the key words are highly similar and the key values remain consistent, it can still be regarded as correct.
        If the prediction is correct according to answer, return 1. Otherwise, return 0. 
        Return: your reply can only be one number '0' or '1'
        ==================
        For example:
        ==================
        Question: What are the primary languages spoken in each of the smallest five countries in Europe by area?
        Answer: {"Vatican City": "Latin, Italian", "Monaco": "French", "San Marino": "Italian", "Liechtenstein": "German", "Malta": "Maltese, English"}
        Prediction: The primary languages spoken in each of the smallest five countries in Europe by area are as follows:
        1. Vatican City: The primary languages are Latin and Italian.
        2. Monaco: The primary language spoken is French.
        3. San Marino: The official language is Italian.
        4. Liechtenstein: The primary language spoken is German.
        5. Malta: The official languages are Maltese and English.
        Return: 1
        ==================
        Question: What are the GDPs of the five largest countries by area on the continent of Africa, in billions of USD?
        Answer: {"Algeria": 173, "Democratic Republic of the Congo": 53, "Sudan": 34, "Libya": 43, "Chad": 11}
        Prediction: The GDPs of the five largest countries by area on the continent of Africa, in billions of USD, are as follows:
        1. Algeria: $238 billion
        2. Democratic Republic of the Congo: $85 billion
        3. Sudan: $34 billion
        4. Libya: $43 billion
        5. Chad: $11 billion
        These figures represent the nominal GDP values of these countries. Algeria is the largest country by land area in Africa, followed by the Democratic Republic of the Congo, Sudan, Libya, and Chad.
        Return: 0
        ==================
        Question: Who won the Men's Singles title at Wimbledon from 2018 to 2022, and what was the final score of each match?
        Answer: {"2018": {"Winner": "Novak Djokovic", "Score": "6–2, 6–2, 7–6(7–3)"},"2019": {"Winner": "Novak Djokovic", "Score": "7–6(7–5), 1–6, 7–6(7–4), 4–6, 13–12(7–3)"}, "2021": {"Winner": "Novak Djokovic", "Score": "6–7(4–7), 6–4, 6–4, 6–3"}, "2022": {"Winner": "Novak Djokovic", "Score": "4–6, 6–3, 6–4, 7–6(7–3)"}}
        Prediction: From 2018 to 2022, the winners of the Men's Singles title at Wimbledon and the final scores of each match are as follows:
        1. In 2018, Novak Djokovic won the title with a score of 6–2, 6–2, 7–6(7–3).
        2. In 2019, Novak Djokovic again claimed victory with a final score of 7–6(7–5), 1–6, 7–6(7–4), 4–6, 13–12(7–3).
        3. Wimbledon was not held in 2020 due to the COVID-19 pandemic.
        4. In 2021, Novak Djokovic secured the title with a score of 6–7(4–7), 6–4, 6–4, 6–3.
        5. In 2022, Novak Djokovic emerged victorious with a final score of 4–6, 6–3, 6–4, 7–6(7–3).
        Return: 1
        ==================
        You are grading the following question:
        Question: {question}
        Answer: {answer}
        Prediction: {prediction}
        Return: """
    return eval_prompt

def get_eval_prompt():
    eval_prompt = """You are an expert professor specialized in grading whether the prediction to the question is correct or not according to the real answer.
        Please note that if the prediction and the answer are not completely identical, but the key words are highly similar and the key values remain consistent, it can still be regarded as correct.
        If the prediction is correct according to answer, return 1. Otherwise, return 0. 
        Return: your reply can only be one number '0' or '1'
        ==================
        For example:
        ==================
        Question: What company owns the property of Marvel Comics?
        Answer: The Walt Disney Company
        Prediction: The walt disney company owns the property of Marvel Comics.
        Return: 1
        ==================
        Question: Which constituent college of the University of Oxford endows four professorial fellowships for sciences including chemistry and pure mathematics?
        Answer: Magdalen College
        Prediction: the magdalen college.
        Return: 1
        ==================
        Question: Which year was Marvel started?
        Answer: 1939
        Prediction: 1200
        Return: 0
        ==================
        You are grading the following question:
        Question: {question}
        Answer: {answer}
        Prediction: {prediction}
        Return: """
    return eval_prompt