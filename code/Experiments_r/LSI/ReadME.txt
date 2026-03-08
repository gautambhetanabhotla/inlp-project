For using the trained model to run evaluation/predictions:

1. Place the "ILSI/" folder inside the same directory as "lsi.py".
2. Create a folder "InLegalBERT" inside "ILSI/" folder.
3. Place the "pytorch_model.bin" file inside "ILSI/InLegalBERT/".
4. In the training arguments (inside "lsi.py"), set "do_train = False"
5. From the main directory, run "python lsi.py ILSI law-ai/InLegalBERT InLegalBERT"