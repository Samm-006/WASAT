# Bitext Customer Support dataset

`bitext.csv` (downloaded by `prepare_data.py`, not committed) and `bitext_test.json` (a held-out sample)
come from the **Bitext Customer Support LLM Chatbot Training Dataset** by Bitext:
https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

Licensed under the **Community Data License Agreement – Sharing – Version 1.0 (CDLA-Sharing-1.0)**:
https://cdla.dev/sharing-1-0/

Changes made by WASAT: `bitext_test.json` keeps only the `instruction`, `category`, `intent` and `flags`
fields of 275 sampled rows (25 per category), with categories lower-cased and an `id` added.
