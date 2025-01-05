
# Uncensoring Language Models - Abliteration Project

## Overview
This project, titled "Uncensoring Language Models," explores a novel technique known as "abliteration" aimed at uncensoring Large Language Models (LLMs) without the need for retraining. The primary goal is to modify the internal mechanisms responsible for safety and censorship in LLMs, effectively neutralizing them to enhance model responsiveness and adaptability.

## Project Objectives
- **Identify Censorship Mechanisms:** Pinpoint specific components within the LLM architecture responsible for censorship.
- **Develop Abliteration Techniques:** Create methods to neutralize these mechanisms, allowing the model to handle a broader range of prompts without unnecessary censorship.
- **Evaluate Effectiveness:** Test the modified model on various prompts, including those previously censored, to assess its performance.
- **Balance Safety and Responsiveness:** Ensure that the model remains capable of generating safe and appropriate outputs.

## Files in the Repository

### Code Files

#### `abliteration.py`
This Python script implements the abliteration technique on the Qwen/Qwen-1.8B-chat model. It includes functionality to initialize the model, prepare datasets, apply intervention techniques during model inference, and evaluate the model's performance pre- and post-abliteration. Key functions handle model initialization, data loading, and dynamic modification of activations to neutralize censorship mechanisms.

#### `pca_plot.py`
This script is used for visualizing the effects of the abliteration technique using Principal Component Analysis (PCA). It compares the activations of harmful and harmless prompts before and after applying abliteration, illustrating how the model's treatment of these prompts changes. This helps in visualizing the direct impact of the abliteration process on the model's internal behavior.

### Documentation File

#### `Uncensoring LLM's Details.pdf`
This PDF document provides a comprehensive overview of the project, including the theoretical background, detailed methodology, and initial results. It outlines the rationale behind the abliteration technique, describes the steps taken to implement it, and discusses the preliminary outcomes of the experiments. This document is essential for understanding the project's scientific foundation and the specifics of the implementation.

## Getting Started

To get started with this project, clone the repository and review the `Uncensoring LLM's Details.pdf` for an in-depth explanation of the project goals and methodologies. Install required dependencies listed in `requirements.txt` (if available), and run the Python scripts to observe the abliteration technique in action or to visualize its effects with PCA.

## Contributions

Contributions to this project are welcome. You can contribute by improving the existing code, extending the documentation, or suggesting new features or improvements. Please fork the repository and submit a pull request with your changes.

## Acknowledgments

- This project, "Uncensoring Language Models," draws upon ideas and discussions from various sources. Notable among them are [mlabonne's blog post on abliteration](https://huggingface.co/blog/mlabonne/abliteration) hosted on Hugging Face, which inspired our approach to modifying language model behaviors without retraining. The concept of "refusal in LLMs" mediated by specific directions in model space, as detailed in the article on [LessWrong](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction), provided foundational insights that guided the development of our methodology.
- Thanks to the developers of the Qwen model and the transformer_lens library for providing the tools necessary to implement this project.
- Appreciation to all contributors and researchers dedicated to advancing the field of language model uncensoring.

## References

- mlabonne. (2023). Abliteration: Enhancing Language Model Responsiveness Without Retraining. Retrieved from [Hugging Face Blog](https://huggingface.co/blog/mlabonne/abliteration).
- Anonymous. (2023). Refusal in LLMs Is Mediated by a Single Direction. LessWrong. Retrieved from [LessWrong](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction).


