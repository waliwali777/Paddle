#!/bin/bash
# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e
cd ..

paddle train \
    --job=test \
    --config='translation/gen.conf' \
    --save_dir='data/wmt14_model' \
    --use_gpu=false \
    --num_passes=13 \
    --test_pass=12 \
    --trainer_count=1 \
    2>&1 | tee 'translation/gen.log'
paddle usage -l 'translation/gen.log' -e $? -n "seqToseq_translation_gen" >/dev/null 2>&1
