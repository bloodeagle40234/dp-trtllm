# Copyright 2024 Kota Tsuyuzaki
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig


class DummyPlug(PreTrainedModel):
    """
    This DummpyPlug class is intended to be used as Pre-Trained model
    for transformers pipeline.
    """
    def __init__(self, pretrained_model_name_or_path, **kwargs):
        self.engine_path = pretrained_model_name_or_path
        self.runner = self._get_trt_instance()
        self.config = PretrainedConfig.from_dict(
            self.runner.session._model_config.__dict__)

    @property
    def model_id(self):
        return self.engine_path

    @property
    def _modules(self):
        return {}

    @classmethod
    def from_dir(cls, engine_path):
        return cls(engine_path)

    def generate(self, **kwargs):
        batch_input_ids = kwargs.pop("input_ids")
        kwargs["end_id"] = kwargs.pop("eos_token")
        kwargs["pad_id"] = kwargs.pop("pad_token")
        result = self.runner.generate(batch_input_ids, **kwargs)
        return result[0]

    def to(self, device):
        # tensorrt-llm only works on CUDA so this api
        # (like as model.to("cuda")) should be just mock.
        pass

    def _get_trt_instance(self):
        tensorrt_llm.logger.set_level("info")
        runtime_rank = tensorrt_llm.mpi_rank()
        torch.cuda.set_device(0)
        runner = ModelRunner.from_dir(
            engine_dir=self.engine_path, rank=runtime_rank)
        return runner
