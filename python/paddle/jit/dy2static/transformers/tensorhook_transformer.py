# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import collections

from paddle.utils import gast

from ..utils import ast_to_source_code
from .base import BaseTransformer


def get_loads(node: gast.AST):
    for child in gast.walk(node):
        if isinstance(
            child, (gast.Name, gast.Attribute, gast.Subscript)
        ) and isinstance(child.ctx, gast.Load):
            yield child


class RegisterHookTransformer(BaseTransformer):
    def __init__(self, root):
        self.register_hook_pos_map = collections.defaultdict(list)
        self.assignment_pos_map = collections.defaultdict(list)
        self.root = root

    def transform(self):
        """
        Main function to transform AST.
        """
        self.visit(self.root)

    def visit_FunctionDef(self, node: gast.FunctionDef):
        func_body = node.body
        regisiter_hook_nodes = [
            n
            for n in func_body
            for stmt in gast.walk(n)
            if isinstance(stmt, gast.Attribute) and stmt.attr == "register_hook"
        ]
        # Analyze the register_hook nodes name dependency
        dependents = {}
        for n in regisiter_hook_nodes:
            if n not in func_body:
                continue
            for load_node in get_loads(n):
                load_name = ast_to_source_code(load_node)
                if load_name not in dependents:
                    dependents[load_name] = []
                dependents[load_name].append(n)

        # Reorder the register_hook nodes, insert it before the dependent nodes
        idx = 0
        body = list(func_body)
        while idx < len(body):
            stmt = body[idx]
            loads = get_loads(stmt)
            for load_node in loads:
                load_name = ast_to_source_code(load_node)
                if load_name in dependents:
                    dep_nodes = dependents[load_name]
                    for dep_node in dep_nodes:
                        dep_idx = body.index(dep_node)
                        if dep_idx <= idx:
                            continue
                        body.remove(dep_node)
                        body.insert(idx, dep_node)
                        idx += 1
            idx += 1
        node.body = body
        return node
