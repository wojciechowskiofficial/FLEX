# Towards faithful and plausible natural language explanations for image classification: a pipeline approach
*F*aithful *L*anguage *EX*planations.

To run our method please run `python run_pipeline_on_val.py`. Remember to adjust the script's arguments. Also, remember to set the GPT4 API key.

For citing us, please use the following BibTeX entry:
```bibtex
@inproceedings{wojciechowski-etal-2024-faithful,
    title = "Faithful and Plausible Natural Language Explanations for Image Classification: A Pipeline Approach",
    author = "Wojciechowski, Adam  and
      Lango, Mateusz  and
      Dusek, Ondrej",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.130",
    doi = "10.18653/v1/2024.findings-emnlp.130",
    pages = "2340--2351",
    abstract = "Existing explanation methods for image classification struggle to provide faithful and plausible explanations. This paper addresses this issue by proposing a post-hoc natural language explanation method that can be applied to any CNN-based classifier without altering its training process or affecting predictive performance. By analysing influential neurons and the corresponding activation maps, the method generates a faithful description of the classifier{'}s decision process in the form of a structured meaning representation, which is then converted into text by a language model. Through this pipeline approach, the generated explanations are grounded in the neural network architecture, providing accurate insight into the classification process while remaining accessible to non-experts. Experimental results show that the NLEs constructed by our method are significantly more plausible and faithful than baselines. In particular, user interventions in the neural network structure (masking of neurons) are three times more effective.",
}
```

## Acknowledgements
This work was co-funded by the European Union (ERC, NG-NLG, 101039303).

<img src="img/LOGO_ERC-FLAG_FP.png" alt="erc-logo" height="150"/> 
