# Acknowledgments

This project gratefully acknowledges the following open-source projects and research groups whose work makes this tool possible.

## NLP Libraries & Models
- **[spaCy](https://spacy.io)** *(MIT License)*  
  Industrial-strength NLP library used for tokenization, POS tagging, dependency parsing, and sentence segmentation.
- **[gr-nlp-toolkit](https://github.com/nlpaueb/gr-nlp-toolkit)** *(Apache 2.0 License)*  
  By the NLP Group at Athens University of Economics and Business (AUEB). Used for Modern Greek NER, POS tagging, and dependency parsing. We especially thank the researchers and authors of the toolkit presented at COLING 2025: Lefteris Loukas, Nikolaos Smyrnioudis, Chrysa Dikonomaki, Spiros Barbakos, Anastasios Toumazatos, John Koutsikakis, Manolis Kyriakakis, Mary Georgiou, Stavros Vassos, John Pavlopoulos, and Ion Androutsopoulos.
- **[OdyCy (grc_odycy_joint_trf)](https://github.com/chcaa/grc_odycy_joint_trf)** *(MIT License)*  
  Ancient/Polytonic Greek transformer model for spaCy by the Aarhus Center for Humanities Computing. Used for robust sentence parsing.
- **[Hugging Face Transformers](https://github.com/huggingface/transformers)** *(Apache 2.0 License)*  
  Used for model loading and subword tokenization.
- **[nlpaueb/bert-base-greek-uncased-v1](https://huggingface.co/nlpaueb/bert-base-greek-uncased-v1)**  
  By AUEB NLP Group. Greek BERT model used for subword tokenization in the Modern Greek pipeline.
- **[UGARIT/grc-ner-bert](https://huggingface.co/UGARIT/grc-ner-bert)**  
  Ancient Greek NER model used for named entity recognition on Ancient Greek texts.
- **[PyTorch](https://pytorch.org)** *(BSD-style License)*  
  Deep learning framework powering the transformer models.

## OCR Engine
- **[Marker](https://github.com/datalab-to/marker)** *(Apache 2.0 License)*  
  High-accuracy PDF to markdown converter by Datalab.
- **[Surya OCR](https://github.com/datalab-to/surya)** *(Apache 2.0 License)*  
  OCR engine powering Marker's text recognition by Datalab. Note: The model weights for these tools may have their own separate commercial use restrictions. See their respective repositories.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** *(MIT License)*  
  Efficient LLM inference engine used by Surya for vision model execution on Apple Metal/CUDA.

## Web Framework & Frontend
- **[Flask](https://flask.palletsprojects.com)** *(BSD-3-Clause License)*  
  Python micro web framework used for the backend server.
- **[Werkzeug](https://werkzeug.palletsprojects.com)** *(BSD-3-Clause License)*  
  WSGI toolkit used by Flask.
- **[Material Design 3](https://m3.material.io)**  
  Design system by Google, providing UI design principles and color tokens used across the application.
- **[Bootstrap](https://getbootstrap.com)** *(MIT License)*
- **[Leaflet](https://leafletjs.com/)** *(BSD-2-Clause License)*
- **[vis-network](https://github.com/visjs/vis-network)** *(Apache 2.0 / MIT License)*
- **[OpenStreetMap](https://www.openstreetmap.org/)** *(ODbL)*
