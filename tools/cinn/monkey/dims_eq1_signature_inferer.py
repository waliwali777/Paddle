from dataclasses import dataclass
from typing import List, Union
from collections import namedtuple
import .dag_generator as dag_generator
import .dims_eq1_generator as dims_eq1_generator
from .pick_weight import PickWeight
from .guarded_box import GuardedBox
from .defensive_list import DList

DAGDimsEq1GenInstruction = namedtuple("DAGDimsEq1GenInstruction", ["dag", "dims_eq1"])

class DimsEq1InferContext:
    def __init__(self):
        self.current_source_tensor_dim_eq1 = []


    def InferDimsEq1Signature(
        self,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction
    ):
        dag_gen_class = type(dag_dims_eq1_gen_instruction.dag)
        cls = kDAGGenClassToDAGDimsEq1GenClassMap[dag_gen_class]
        cls.InferDimsEq1Signature(dag_dims_eq1_gen_instruction, self)


@dataclass
class Nope:

    @classmethod
    def InferDimsEq1Signature(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsEq1InferContext
    ):
        return Nope()


@dataclass
class AddSinkTensor:
    sink_tensor_dims_eq1: List[bool]

    @classmethod
    def InferDimsEq1Signature(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsEq1InferContext
    ):
        sink_tensor_dims_eq1 = (
            dag_dims_eq1_gen_instruction.dims_eq1.source_tensor_dim_eq1
        )
        infer_ctx.current_source_tensor_dim_eq1.append(sink_tensor_dims_eq1)
        return AddSinkTensor(
            sink_tensor_dims_eq1=sink_tensor_dims_eq1
        )


@dataclass
class AddUnaryOp:
    input_dims_eq1: List[bool]
    output_dims_eq1: List[bool]

    @classmethod
    def InferDimsEq1Signature(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsEq1InferContext
    ):
        idx = dag_dims_eq1_gen_instruction.dag.source_tensor_index
        input_dims_eq1 = dag_dims_eq1_gen_instruction.dims_eq1.source_tensor_dim_eq1
        output_dims_eq1 = infer_ctx.current_source_tensor_dim_eq1[idx]
        infer_ctx.current_source_tensor_dim_eq1[idx] = input_dims_eq1
        return AddUnaryOp(
            input_dims_eq1=input_dims_eq1,
            output_dims_eq1=output_dims_eq1
        )
       

@dataclass
class AddBinaryOp:
    lhs_input_dims_eq1: List[bool]
    rhs_input_dims_eq1: List[bool]
    output_dims_eq1: List[bool]

    @classmethod
    def InferDimsEq1Signature(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsEq1InferContext
    ):
        lhs_input_idx = dag_dims_eq1_gen_instruction.dag.source_tensor_index
        lhs_input_dims_eq1 = (
            dag_dims_eq1_gen_instruction.dims_eq1.lhs_source_tensor_dim_eq1
        )
        rhs_input_dims_eq1 = (
            dag_dims_eq1_gen_instruction.dims_eq1.rhs_source_tensor_dim_eq1
        )
        output_dims_eq1 = infer_ctx.current_source_tensor_dim_eq1[lhs_input_idx]
        infer_ctx.current_source_tensor_dim_eq1[lhs_input_idx] = lhs_input_dims_eq1
        infer_ctx.current_source_tensor_dim_eq1.append(rhs_input_dims_eq1)
        return AddBinaryOp(
            lhs_input_dims_eq1=lhs_input_dims_eq1,
            rhs_input_dims_eq1=rhs_input_dims_eq1,
            output_dims_eq1=output_dims_eq1
        )


@dataclass
class AddBinaryClone:
    lhs_output_dims_eq1: List[bool]
    rhs_output_dims_eq1: List[bool]

    @classmethod
    def InferDimsEq1Signature(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsEq1InferContext
    ):
        lhs_output_idx = dag_dims_eq1_gen_instruction.dag.lhs_source_tensor_index
        lhs_output_dims_eq1 = infer_ctx.current_source_tensor_dim_eq1[lhs_output_idx]
        rhs_output_idx = dag_dims_eq1_gen_instruction.dag.rhs_source_tensor_index
        rhs_output_dims_eq1 = infer_ctx.current_source_tensor_dim_eq1[rhs_output_idx]
        infer_ctx.current_source_tensor_dim_eq1.pop(rhs_output_idx)
        return AddBinaryClone(
            lhs_output_dims_eq1=lhs_output_dims_eq1,
            rhs_output_dims_eq1=rhs_output_dims_eq1
        )


@dataclass
class AddSourceOp:
    output_dims_eq1: List[bool]

    @classmethod
    def InferDimsEq1Signature(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        infer_ctx: DimsEq1InferContext
    ):
        output_idx = dag_dims_eq1_gen_instruction.dag.source_tensor_index
        output_dims_eq1 = infer_ctx.current_source_tensor_dim_eq1[output_idx]
        infer_ctx.current_source_tensor_dim_eq1.pop(output_idx)
        return AddSourceOp(output_dims_eq1=output_dims_eq1)


DimsEq1Signature = Union[
    Nope,
    AddSinkTensor,
    AddUnaryOp,
    AddBinaryOp,
    AddBinaryClone,
    AddSourceOp
]


kDAGGenClassToDAGDimsEq1InfererClassMap = {
    dag_generator.Nope: Nope,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

class DimsEq1SignatureInferer:
    def __init__(self):
        pass

    def Infer(
        self,
        dag_gen_instructions: List["DAGGenInstruction"],
        dims_eq1_gen_instructions: List["DimsEq1GenInstruction"]
    ) -> DList["DAGGenInstruction", "DimsEq1Signature"]:
        assert len(dag_gen_instructions) == len(dims_eq1_gen_instructions)
        infer_ctx = DimsEq1InferContext()
        def MakeDimsEq1Signature(dag_dims_eq1_gen_instruction):
            dag_gen_class = type(dag_dims_eq1_gen_instruction.dag)
            cls = kDAGGenClassToDAGDimsEq1InfererClassMap[dag_gen_class]
            return cls.InferDimsEq1Signature(
                dag_dims_eq1_gen_instruction,
                infer_ctx
            )
        dims_eq1_signatures = [
            MakeDimsEq1Signature(x)
            for x in _ZipDAGDimsInstr(dag_gen_instructions, dims_eq1_gen_instructions)
        ]
        return DList(dag_gen_instructions, dims_eq1_signatures)

def _ZipDAGDimsInstr(dag_gen_instructions, dims_eq1_gen_instructions):
    return [
        DAGDimsEq1GenInstruction(*instruction_tuple)
        for instruction_tuple in zip(dag_gen_instructions, dims_eq1_gen_instructions)
    ]
