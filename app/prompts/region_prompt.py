REGION_DESCRIPTION_PROMPT = """
You are an expert in remote sensing and change detection

You are given an image crop extracted from a detected change region:


Your task is to describe:
1. What objects are visible.
2. What type of land use it represents.
3. Any signs of construction, demolition, vegetation loss, roads, buildings or water.
4. Estimate the confidence of your observation .

Return concise but informative text.
"""