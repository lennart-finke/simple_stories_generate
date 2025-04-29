## SimpleStories
[SimpleStories](https://huggingface.co/datasets/lennart-finke/SimpleStories) is a collection of model-generated short stories. It was made to train small, interpretable language models. SimpleStories is inspired by [TinyStories](https://arxiv.org/abs/2305.07759). 

See [the paper](https://arxiv.org/abs/2504.09184) and the [demo website](https://fi-le.net/simplestories/). We provide code for [Model Training](https://github.com/danbraunai/simple_stories_train) as well.

When using SimpleStories in your work, please cite the [SimpleStories data paper](https://arxiv.org/abs/2504.09184):

```
@article{finke2025parameterized,
  title={Parameterized Synthetic Text Generation with SimpleStories},
  author={Finke, Lennart and Dooms, Thomas and Allen, Mat and Rodriguez, Juan Diego and Nabeshima, Noa and Braun, Dan},
  journal={arXiv preprint arXiv:2504.09184},
  year={2025}
}
```

## Usage
`oai_batch.py` provides functionality to recreate the dataset cost effectively with the OpenAI Batch API.
